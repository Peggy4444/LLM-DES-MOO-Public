import simpy
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


# Assumes the simulation code is imported in the same namespace:
# - RANDOM_SEED
# - REPLICATIONS
# - run_replication()
# - BUFFERS dict exists and is used by ProductionSystem


def run_simulation_with_caps(seed, caps):
    """
    Run one replication of the original production system simulation
    with modified buffer capacities given by `caps`.

    caps: list or array of 6 integers in [1,10] corresponding to:
        [PostLoading, PostConveyor, PostWashing,
         PrePress1, PrePress2, PostPress1and2]
    """
    # Set global random seed for this replication
    random.seed(seed)

    # Backup original capacities
    original_caps = {
        key: BUFFERS[key]['cap']
        for key in [
            'PostLoading',
            'PostConveyor',
            'PostWashing',
            'PrePress1',
            'PrePress2',
            'PostPress1and2'
        ]
    }

    # Apply new capacities
    BUFFERS['PostLoading']['cap'] = int(caps[0])
    BUFFERS['PostConveyor']['cap'] = int(caps[1])
    BUFFERS['PostWashing']['cap'] = int(caps[2])
    BUFFERS['PrePress1']['cap'] = int(caps[3])
    BUFFERS['PrePress2']['cap'] = int(caps[4])
    BUFFERS['PostPress1and2']['cap'] = int(caps[5])

    # Run one replication using the existing function
    th, w, e = run_replication(seed)

    # Restore original capacities to avoid side effects
    for k, v in original_caps.items():
        BUFFERS[k]['cap'] = v

    return th, w


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [avg_wip, -avg_throughput].
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    throughputs = []
    wips = []

    # Local RNG to generate distinct seeds per replication
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1_000_000)
        th, w = run_simulation_with_caps(seed, caps)
        throughputs.append(th)
        wips.append(w)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integer in [1,10]):
        x[0] = PostLoading cap
        x[1] = PostConveyor cap
        x[2] = PostWashing cap
        x[3] = PrePress1 cap
        x[4] = PrePress2 cap
        x[5] = PostPress1and2 cap

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 n_replications=REPLICATIONS,
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

        # Prepare tasks for parallel evaluation
        tasks = [
            (X[i], self.n_replications, self.base_seed)
            for i in range(n_individuals)
        ]

        # Use exactly self.n_cores processes
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=REPLICATIONS,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    """
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1] * 6),
        xu=np.array([10] * 6),
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
    Only feasible (evaluated) individuals are exported.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "PostLoading_cap",
        "PostConveyor_cap",
        "PostWashing_cap",
        "PrePress1_cap",
        "PrePress2_cap",
        "PostPress1and2_cap",
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
            # Skip individuals that were not evaluated (contain NaN)
            if np.any(np.isnan(f)):
                continue

            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "PostLoading_cap": caps[0],
                "PostConveyor_cap": caps[1],
                "PostWashing_cap": caps[2],
                "PrePress1_cap": caps[3],
                "PrePress2_cap": caps[4],
                "PostPress1and2_cap": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Use exactly 50 cores
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")