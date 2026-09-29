import multiprocessing
import random
import statistics
import os
import csv

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# NOTE:
# This MOO code expects the production simulation functions and constants
# (run_simulation, WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED) to be available
# in the same Python namespace (for easy integration place this code into the
# same module as the simulation or import them before running).
#
# Decision variables: the 6 DelayBuffer capacities (integers 1..10)
# Objectives: minimize WIP, maximize throughput (returned as negative for pymoo)
# Population size: 50
# Generations: 50
# Use exactly 50 CPU cores for parallel evaluation

PENALTY_OBJ = 1e6  # large penalty for infeasible/failed evaluations


def evaluate_single_individual(args):
    """
    Evaluate one individual (vector of buffer capacities) by running n_replications
    of the provided simulation. Return (F, G) where:
      - F is a list/array of objective values [wip, -throughput]
      - G is a single constraint value (<=0 feasible, >0 infeasible)
    If any simulation run fails (raises), this returns a positive G and penalized F.
    """
    x, n_replications, base_seed, warmup, measure_until = args

    # Ensure integer capacities
    caps = [int(v) for v in x[:6]]

    throughputs = []
    wips = []

    try:
        # Use a local RNG to derive distinct seeds per replication
        local_rng = random.Random()
        local_rng.seed(base_seed + sum(caps))

        for r in range(n_replications):
            seed = base_seed + r + local_rng.randint(0, 2_000_000_000)
            # Expect run_simulation to be available in the same namespace
            res = run_simulation(seed, warmup=warmup, measure_until=measure_until)
            # The provided simulation signature must be compatible:
            # - If your simulation accepts buffer capacities as arguments,
            #   modify run_simulation accordingly and pass caps here.
            throughputs.append(res["overall"]["throughput"])
            wips.append(res["overall"]["wip"])

        avg_throughput = statistics.mean(throughputs)
        avg_wip = statistics.mean(wips)

        # Objectives: minimize wip, minimize -throughput (so pymoo treats it as minimization)
        F = [avg_wip, -avg_throughput]
        G = -1.0  # feasible (<=0)

    except Exception:
        # On any error treat as infeasible and apply heavy penalty to objectives
        F = [PENALTY_OBJ, PENALTY_OBJ]
        G = 1.0  # infeasible (>0)

    return F, G


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for six buffer capacities using NSGA-II.

    Decision variables (integers 1..10):
      x[0] = post_loading_buffer cap
      x[1] = post_conveyor_buffer cap
      x[2] = post_washing_buffer cap
      x[3] = pre_press1_buffer cap
      x[4] = pre_press2_buffer cap
      x[5] = post_press12_buffer cap

    Objectives:
      f1 = average WIP (minimize)
      f2 = -average throughput (minimize, since pymoo minimizes objectives)

    Constraints:
      One generic constraint output is provided. Feasible if G <= 0.
      evaluate_single_individual returns G > 0 for infeasible/failed evaluations.
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 n_constr=1,
                 xl=None,
                 xu=None,
                 n_replications=5,
                 base_seed=None,
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
        self.base_seed = base_seed if base_seed is not None else RANDOM_SEED
        self.n_cores = n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # Use exactly self.n_cores worker processes
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        F = np.zeros((n_individuals, self.n_obj), dtype=float)
        G = np.zeros((n_individuals, self.n_constr), dtype=float)

        for i, (f_vals, g_val) in enumerate(results):
            F[i, :] = f_vals
            G[i, 0] = g_val

        out["F"] = F
        out["G"] = G


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=None,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem using the provided
    simulation run_simulation in the same namespace.
    """

    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        n_constr=1,
        xl=np.array([1] * 6),
        xu=np.array([10] * 6),
        n_replications=n_replications,
        base_seed=base_seed if base_seed is not None else RANDOM_SEED,
        n_cores=n_cores
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
        seed=problem.base_seed,
        save_history=True,
        verbose=verbose
    )

    return res


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export feasible solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file. Infeasible solutions
    (G > 0) are omitted.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "post_loading_buffer",
        "post_conveyor_buffer",
        "post_washing_buffer",
        "pre_press1_buffer",
        "pre_press2_buffer",
        "post_press12_buffer",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")  # constraint violations; shape (n_pop, n_constr) or None

        for ind_idx, (x, f) in enumerate(zip(X, F)):
            is_feasible = True
            if G is not None:
                # If any constraint > 0 -> infeasible
                try:
                    is_feasible = np.all(G[ind_idx] <= 0)
                except Exception:
                    is_feasible = True

            if not is_feasible:
                continue  # skip infeasible individual

            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "post_loading_buffer": caps[0],
                "post_conveyor_buffer": caps[1],
                "post_washing_buffer": caps[2],
                "pre_press1_buffer": caps[3],
                "pre_press2_buffer": caps[4],
                "post_press12_buffer": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Ensure the multiprocessing start method is compatible across platforms
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Already set
        pass

    # Run NSGA-II optimization with specified configuration:
    # - population size 50
    # - 50 generations
    # - exactly 50 cores used during evaluation
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=3,
        base_seed=RANDOM_SEED,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")