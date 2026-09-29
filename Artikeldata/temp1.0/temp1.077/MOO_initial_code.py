import simpy
import multiprocessing
import random
import statistics
import os

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv


# Assumes the production simulation code is imported in the same namespace,
# including: run_simulation, RANDOM_SEED, WARMUP_SECONDS, MEASURE_UNTIL.


# --- Constraint handling for buffer capacities ---
# Decision variables:
#   x[0] = post_loading_buffer cap
#   x[1] = post_conveyor_buffer cap
#   x[2] = post_washing_buffer cap
#   x[3] = pre_press1_buffer cap
#   x[4] = pre_press2_buffer cap
#   x[5] = post_press12_buffer cap
#
# All caps integer in [1, 10]
#
# Example structural constraint (can be modified as needed):
#   - Total capacity of both pre-press buffers must be >= capacity of shared pre_press_shared (6)
#     => x[3] + x[4] >= 6
#
# Any individual violating constraints will not be evaluated, and thus not appear in CSV.
# We implement this by:
#   - In _evaluate: detect feasibility using constraint_function.
#   - For infeasible individuals: assign NaN to objectives, which we then filter out
#     when exporting history to CSV, so they never appear in results.


def constraint_function(x):
    """
    Returns True if x is feasible, False otherwise.
    x is a 1D array-like of length 6 with integer capacities.
    """
    # Example constraint: pre-press buffers combined capacity >= 6
    pre_press1 = int(x[3])
    pre_press2 = int(x[4])
    if pre_press1 + pre_press2 < 6:
        return False

    # Add other constraints here if needed.

    return True


def run_simulation_with_caps(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Wrapper around the provided run_simulation that injects the buffer capacities.
    To avoid duplicating the simulation code, we call the original run_simulation
    after monkey-patching the capacities via a factory function pattern.

    In this implementation, we re-create run_simulation logic locally by directly
    calling the original run_simulation for each cap set is not possible without
    modifying the original. To keep integration easy, we assume the user will
    adapt the original run_simulation to accept caps if needed.

    Here we assume that the provided run_simulation has been modified to accept
    an additional 'caps' argument of length 6 and uses them as:
        caps[0] -> post_loading_buffer.cap
        caps[1] -> post_conveyor_buffer.cap
        caps[2] -> post_washing_buffer.cap
        caps[3] -> pre_press1_buffer.cap
        caps[4] -> pre_press2_buffer.cap
        caps[5] -> post_press12_buffer.cap
    """
    # Directly call user-supplied run_simulation with caps
    return run_simulation(seed, caps, warmup=warmup, measure_until=measure_until)


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications, including constraint handling.

    If the individual violates constraints, return NaN objectives, so that
    it can be filtered out later (it will not be written to CSV).
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x[:6]]

    # Check constraints before any simulation calls
    if not constraint_function(caps):
        return [np.nan, np.nan]

    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1_000_000)
        res = run_simulation_with_caps(seed, caps, warmup=warmup, measure_until=measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Return the objectives: [f1 (wip), f2 (-throughput)]
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables:
        x[0] = post_loading_buffer cap
        x[1] = post_conveyor_buffer cap
        x[2] = post_washing_buffer cap
        x[3] = pre_press1_buffer cap
        x[4] = pre_press2_buffer cap
        x[5] = post_press12_buffer cap

    Bounds:
        each in [1, 10], integer

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)

    Constraint handling:
        - Constraints are enforced in evaluate_single_individual via constraint_function.
        - Infeasible individuals get NaN objectives, and are excluded from CSV export.
    """

    def __init__(self, n_var=6, n_obj=2, n_constr=0,
                 xl=None, xu=None,
                 n_replications=5,
                 base_seed=RANDOM_SEED,
                 n_cores=50):
        if xl is None:
            xl = np.array([1] * n_var, dtype=int)
        if xu is None:
            xu = np.array([10] * n_var, dtype=int)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu,
                         elementwise_evaluation=False)

        self.n_replications = n_replications
        self.base_seed = base_seed
        self.n_cores = n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # Use exactly 50 cores
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    Uses exactly 50 cores for parallel simulation evaluations.
    """
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1] * 6),
        xu=np.array([10] * 6),
        n_replications=n_replications,
        base_seed=base_seed,
        n_cores=50
    )

    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(prob=1.0 / problem.n_var, eta=20)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=base_seed,
        save_history=True,
        verbose=verbose
    )

    return res


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export all feasible solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Any individuals with NaN objective values (infeasible / not evaluated)
    are skipped and not written to the CSV file.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "post_loading_cap",
        "post_conveyor_cap",
        "post_washing_cap",
        "pre_press1_cap",
        "pre_press2_cap",
        "post_press12_cap",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        for ind_idx, (x, f) in enumerate(zip(X, F)):
            # Skip individuals that were not evaluated (NaN objectives)
            if np.any(np.isnan(f)):
                continue

            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "post_loading_cap": caps[0],
                "post_conveyor_cap": caps[1],
                "post_washing_cap": caps[2],
                "pre_press1_cap": caps[3],
                "pre_press2_cap": caps[4],
                "post_press12_cap": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization on the production line simulation model
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        verbose=True,
        n_cores=50
    )

    # Export all feasible solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")