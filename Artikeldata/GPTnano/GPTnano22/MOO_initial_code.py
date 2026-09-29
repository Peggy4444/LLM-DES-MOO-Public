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

# The simulation definitions (run_simulation, DelayBuffer, Machine, etc.)
# are assumed to be defined in the integrated simulation module.
# The constants WARMUP_SECONDS and MEASURE_UNTIL are also assumed to be defined there.

def evaluate_single_individual(args):
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x[:3]]
    # Constraint: buffer capacities must be within 1..10
    if any((c < 1 or c > 10) for c in caps):
        return [np.nan, np.nan]
    throughputs = []
    wips = []
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))
    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        res = run_simulation(seed, caps, warmup, measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])
    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)
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

        # Prepare tasks for parallel evaluation
        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # Evaluate in parallel
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
    Export all valid solutions from every generation (excluding invalid points)
    with their KPIs and decision variables to a CSV file.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
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

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            continue
        for ind_idx, (x, f) in enumerate(zip(X, F)):
            # Skip invalid objectives (due to constraints)
            try:
                if not (np.isfinite(float(f[0])) and np.isfinite(float(f[1]))):
                    continue
            except Exception:
                continue
            caps = [int(v) for v in x[:3]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as negative throughput
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
    # Run NSGA-II optimization on the simulation model
    result = run_nsga2_optimization(pop_size=50, n_gen=50, n_replications=5, verbose=True, n_cores=50)

    # Export all valid solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")