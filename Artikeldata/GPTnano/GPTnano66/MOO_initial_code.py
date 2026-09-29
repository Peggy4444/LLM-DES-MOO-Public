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

# Integration constants (align with existing simulation model defaults)
RANDOM_SEED = 66
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME

# Expose a minimal API to the outer simulation model when integrated
# run_simulation(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL)
# is assumed to be provided by the existing simulation code base.

# Feasibility helper for constraints (buffer capacities in 1..10)
def is_feasible_caps(x):
    try:
        caps = [int(v) for v in x[:3]]
    except Exception:
        return False
    return all(1 <= c <= 10 for c in caps)

def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x[:3]]

    # If infeasible caps, return a flag that will be filtered out later
    if not is_feasible_caps(x):
        return [np.nan, np.nan]

    throughputs = []
    wips = []

    # Create a local random instance to prevent multiple CPU processes 
    # from pulling from the same global random state and generating identical seeds
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        # Call to the external simulation model; assumed to exist in integration
        res = run_simulation(seed, caps, warmup=warmup, measure_until=measure_until)
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
        x[0] = buffer1 cap
        x[1] = buffer2 cap
        x[2] = buffer3 cap

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self, n_var=3, n_obj=2, n_constr=0,
                 xl=None, xu=None,
                 n_replications=5,
                 base_seed=RANDOM_SEED,
                 n_cores=1):
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

        # 1. Package the arguments for each individual
        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # 2. Map the tasks across all available CPU cores
        # Using a context manager (with) ensures processes are cleaned up properly
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        # 3. Filter out infeasible evaluated individuals (nan indicates infeasible)
        feasible_results = [r for r in results if not (r[0] != r[0] or r[1] != r[1])]
        if not feasible_results:
            out["F"] = np.array([[np.nan, np.nan] for _ in range(n_individuals)], dtype=float)
            return

        # 4. Assign the compiled results matrix back to out["F"]
        # Note: We align with the number of individuals by filtering and reconstructing the array
        F = np.array(feasible_results, dtype=float)

        # If some individuals were infeasible (nan rows), we keep them as NaNs in F to be ignored by the optimizer
        # Construct a full F matrix with NaNs for infeasible points
        full_F = np.empty((n_individuals, F.shape[1]), dtype=float)
        full_F[:] = np.nan
        idx = 0
        for i in range(n_individuals):
            if not (results[i][0] != results[i][0] or results[i][1] != results[i][1]):
                full_F[i] = F[idx]
                idx += 1

        out["F"] = full_F

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
    problem = BufferCapacityProblem(
        n_var=3,
        n_obj=2,
        xl=np.array([1, 1, 1]),
        xu=np.array([10, 10, 10]),
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
    Filters out infeasible points (caps outside 1..10 are considered infeasible).
    """

    # 1. Define the target directory
    output_dir = "result"
    
    # 2. Safely create the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Join the directory name and the filename together
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "buffer1_cap",
        "buffer2_cap",
        "buffer3_cap",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    def _caps_from_x(x):
        return [int(v) for v in x[:3]]

    def _is_valid_row(x, f):
        if x is None or f is None:
            return False
        if not is_feasible_caps(x):
            return False
        caps = _caps_from_x(x)
        if not all(1 <= c <= 10 for c in caps):
            return False
        if not np.isfinite(f).all():
            return False
        return True

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            continue
        for ind_idx, (x, f) in enumerate(zip(X, F)):
            if not _is_valid_row(x, f):
                continue
            caps = [int(v) for v in x[:3]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "buffer1_cap": caps[0],
                "buffer2_cap": caps[1],
                "buffer3_cap": caps[2],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization on the buffer capacity optimization problem
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        verbose=True,
        n_cores=50  # Exactly 50 cores
    )

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")