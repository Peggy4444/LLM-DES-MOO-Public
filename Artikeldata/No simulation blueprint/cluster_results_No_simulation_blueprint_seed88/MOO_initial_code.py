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


# Decision variables: all 6 buffer capacities in BUFFERS (1–10, integer)
BUFFER_NAMES = [
    "PostLoading",
    "PostConveyor",
    "PostWashing",
    "PrePress1",
    "PrePress2",
    "PostPresses",
]


def run_simulation_with_caps(seed, caps):
    """
    Run the given production line simulation with modified buffer capacities.

    caps: list/array of 6 integers (1–10) in the order of BUFFER_NAMES.
    Returns:
        throughput_per_hour, mean_wip, energy_per_part
    """
    # Import everything from the existing simulation module namespace.
    # Assumes this MOO file is in the same module or that the simulation
    # code has been executed so that the following names exist:
    #   RANDOM_SEED, SIM_TIME, WARMUP, REPLICATIONS,
    #   MACHINES, BUFFERS, Machine, Buffer,
    #   part_generator, run_replication
    global RANDOM_SEED, SIM_TIME, WARMUP, REPLICATIONS
    global MACHINES, BUFFERS, Machine, Buffer
    global part_generator, run_replication

    # Set the capacities in BUFFERS according to caps
    for i, bname in enumerate(BUFFER_NAMES):
        BUFFERS[bname]["cap"] = int(caps[i])

    # Now run the same logic as in the original script but with modified BUFFERS
    throughputs = []
    wips = []
    energies = []

    for r in range(REPLICATIONS):
        # run_replication uses RANDOM_SEED + rep_id internally
        th, wip, en = run_replication(r)
        throughputs.append(th)
        wips.append(wip)
        energies.append(en)

    mean_throughput = statistics.mean(throughputs) if throughputs else 0.0
    mean_wip = statistics.mean(wips) if wips else 0.0
    mean_energy = statistics.mean(energies) if energies else 0.0

    return mean_throughput, mean_wip, mean_energy


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications (handled inside run_simulation_with_caps).

    Returns objectives: [f1 (wip), f2 (-throughput)]
    """
    x, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Local RNG to decorrelate seeds if needed
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    # Use a derived seed for this individual (if you want to influence global RANDOM_SEED)
    seed = base_seed + local_rng.randint(0, 10**6)

    # Optionally, you could adjust RANDOM_SEED here if desired, but run_replication
    # already uses RANDOM_SEED + rep_id. We keep base behavior and only pass seed
    # to keep the interface consistent.
    throughput, wip, _ = run_simulation_with_caps(seed, caps)

    return [wip, -throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (integer, 1–10):
        x[0] = PostLoading cap
        x[1] = PostConveyor cap
        x[2] = PostWashing cap
        x[3] = PrePress1 cap
        x[4] = PrePress2 cap
        x[5] = PostPresses cap

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
                 base_seed=88,
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

        self.base_seed = base_seed
        self.n_cores = n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.base_seed)
            for i in range(n_individuals)
        ]

        # Use exactly self.n_cores processes
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    base_seed=88,
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

    Only feasible points are exported. In this setup, all points within bounds
    are feasible, and no objective is evaluated for infeasible points.
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
        "PostPresses_cap",
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
            # If any objective is NaN, treat as infeasible and skip
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
                "PostPresses_cap": caps[5],
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
        base_seed=88,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")