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


# Assumes the simulation code is imported in the same namespace and provides:
# - run_replication(rep) -> (throughput_per_hour, mean_wip, energy_per_part)
# - BUF_CONF, RANDOM_SEED, REPLICATIONS


def run_simulation_for_caps(seed, caps, n_replications=REPLICATIONS):
    """
    Run the given production line simulation for a specific set of buffer capacities
    over multiple replications and return average throughput and WIP.
    """
    # caps is a list/array of 6 integers in order:
    # [PostLoading, PostConveyor, PostWashing, PrePress1, PrePress2, PostPress12]
    caps = [int(v) for v in caps[:6]]

    # Backup original capacities
    original_caps = {
        name: BUF_CONF[name]["cap"]
        for name in BUF_CONF.keys()
    }

    # Apply new capacities
    buf_names = ["PostLoading", "PostConveyor", "PostWashing", "PrePress1", "PrePress2", "PostPress12"]
    for name, cap in zip(buf_names, caps):
        BUF_CONF[name]["cap"] = cap

    throughputs = []
    wips = []

    # Local RNG to generate distinct seeds per replication
    local_rng = random.Random(seed)

    for r in range(n_replications):
        rep_seed_offset = local_rng.randint(0, 10**9)
        # run_replication uses RANDOM_SEED + rep internally, so we temporarily
        # adjust RANDOM_SEED to incorporate our offset
        global RANDOM_SEED
        old_rs = RANDOM_SEED
        RANDOM_SEED = old_rs + rep_seed_offset

        th, w, _ = run_replication(r)
        throughputs.append(th)
        wips.append(w)

        RANDOM_SEED = old_rs

    # Restore original capacities
    for name in BUF_CONF.keys():
        BUF_CONF[name]["cap"] = original_caps[name]

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [f1, f2] = [avg_wip, -avg_throughput].
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Constraint example (if any) could be checked here.
    # If violated, we would skip evaluation and return a large penalty.
    # Since no explicit constraint is given beyond variable bounds, we evaluate directly.

    # Create a local seed based on base_seed and decision variables
    seed = base_seed + sum(caps)

    avg_throughput, avg_wip = run_simulation_for_caps(seed, caps, n_replications=n_replications)

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
        x[5] = PostPress12 cap

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

        tasks = [
            (X[i], self.n_replications, self.base_seed)
            for i in range(n_individuals)
        ]

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

    Only feasible/evaluated points are exported.
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
        "PostPress12_cap",
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
            # If F contains NaN or inf, skip (treat as non-evaluated / infeasible)
            if not np.all(np.isfinite(f)):
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
                "PostPress12_cap": caps[5],
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
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50
    )

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")