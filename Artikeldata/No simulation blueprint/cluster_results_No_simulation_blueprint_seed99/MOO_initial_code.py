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


# Assumes the simulation code (including run_replication, BUFFERS, RANDOM_SEED, etc.)
# is imported or present in the same module.


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed = args

    # Map decision variables to buffer capacities (order must match BUFFERS)
    caps = [int(v) for v in x]

    # Apply capacities to global BUFFERS dict (copy to avoid side effects across individuals)
    original_buffers = BUFFERS.copy()
    try:
        # BUFFERS is an ordered dict-like; ensure same order as decision variables
        new_buffers = {}
        for (name, (_, pt)), cap in zip(original_buffers.items(), caps):
            new_buffers[name] = (cap, pt)

        # Temporarily override global BUFFERS
        globals()['BUFFERS'] = new_buffers

        throughputs = []
        wips = []

        # Local RNG to decorrelate seeds
        local_rng = random.Random()
        local_rng.seed(base_seed + sum(caps))

        for r in range(n_replications):
            seed = base_seed + r + local_rng.randint(0, 1000000)
            # run_replication uses RANDOM_SEED + rep internally, so we adjust RANDOM_SEED
            globals()['RANDOM_SEED'] = seed
            res = run_replication(r)
            throughputs.append(res['throughput_rate'])
            wips.append(res['wip'])

        avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
        avg_wip = statistics.mean(wips) if wips else 0.0

        # Objectives: f1 = WIP (min), f2 = -throughput (max throughput)
        return [avg_wip, -avg_throughput]

    finally:
        # Restore original BUFFERS
        globals()['BUFFERS'] = original_buffers


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables: capacities of all buffers in BUFFERS (1–10, integer).
    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self,
                 n_obj=2,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 n_replications=REPLICATIONS,
                 base_seed=RANDOM_SEED,
                 n_cores=50):
        self.buffer_names = list(BUFFERS.keys())
        n_var = len(self.buffer_names)

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
        X = np.asarray(X, dtype=int)
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
    n_var = len(BUFFERS.keys())

    problem = BufferCapacityProblem(
        n_obj=2,
        xl=np.array([1] * n_var, dtype=int),
        xu=np.array([10] * n_var, dtype=int),
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

    buffer_names = list(BUFFERS.keys())
    fieldnames = ["gen", "ind"] + [f"{name}_cap" for name in buffer_names] + ["wip", "throughput"]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")

        for ind_idx, (x, f) in enumerate(zip(X, F)):
            if f is None or any(np.isnan(f)):
                continue

            caps = [int(v) for v in x]
            wip = float(f[0])
            throughput = float(-f[1])

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "wip": wip,
                "throughput": throughput
            }

            for name, cap in zip(buffer_names, caps):
                row[f"{name}_cap"] = cap

            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")