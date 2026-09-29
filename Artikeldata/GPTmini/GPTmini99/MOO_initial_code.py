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

# The user should ensure the simulation module with the function
# run_simulation(seed, caps, warmup, measure_until) and the constants
# WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED are available in the
# Python path and importable as `production_sim`.
#
# Example expected signature in the simulation module:
#   def run_simulation(seed: int, caps: Sequence[int], warmup: int, measure_until: int) -> dict
#
# The returned dict must follow the structure used below:
#   res["overall"]["throughput"], res["overall"]["wip"], res["overall"]["produced_parts"]
#
# The MOO code below will call run_simulation(...) with `caps` being a list
# of integers (one per buffer). Adjust the simulation module if necessary
# so it accepts and applies these capacities.
from production_sim import run_simulation, WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [avg_wip, -avg_throughput].
    """
    x, n_replications, base_seed, warmup, measure_until = args
    # Ensure integer capacities
    caps = [int(v) for v in x]

    throughputs = []
    wips = []

    # Local RNG to generate independent seeds per replication
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1_000_000)
        res = run_simulation(seed, caps, warmup, measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Objectives: minimize wip, maximize throughput -> pymoo minimizes so negate throughput
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for all buffer capacities using NSGA-II.

    Decision variables: all delay buffer capacities (integers in [1,10]).
    The number of variables (n_var) should match the number of delay buffers
    in the simulation model. For the provided production simulation this is 6:
      post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
      pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 xl=None,
                 xu=None,
                 n_replications=5,
                 base_seed=RANDOM_SEED,
                 n_cores=50):
        if xl is None:
            xl = np.array([1] * n_var, dtype=int)
        if xu is None:
            xu = np.array([10] * n_var, dtype=int)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=xl,
                         xu=xu,
                         elementwise_evaluation=False)

        self.n_replications = n_replications
        self.base_seed = base_seed
        # Force exactly the requested core count (the user requested exactly 50 cores).
        self.n_cores = int(n_cores)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=int)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # Use exactly self.n_cores worker processes
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
    """

    # Number of decision variables equals number of delay buffers in the main simulation (6)
    n_vars = 6

    problem = BufferCapacityProblem(
        n_var=n_vars,
        n_obj=2,
        xl=np.array([1] * n_vars),
        xu=np.array([10] * n_vars),
        n_replications=n_replications,
        base_seed=base_seed,
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
        seed=base_seed,
        save_history=True,
        verbose=verbose
    )

    return res


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export all solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Prepare header for 6 buffer capacities
    fieldnames = [
        "gen",
        "ind",
    ] + [f"buffer{i+1}_cap" for i in range(6)] + [
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
            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
            }
            for i in range(6):
                row[f"buffer{i+1}_cap"] = caps[i]
            row["wip"] = wip
            row["throughput"] = throughput
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Force exactly 50 cores for evaluation as requested
    CORES = 50

    # Run NSGA-II optimization on the simulation model with pop_size=50 and n_gen=50
    result = run_nsga2_optimization(pop_size=50, n_gen=50, n_replications=5,
                                    base_seed=RANDOM_SEED, verbose=True, n_cores=CORES)

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")