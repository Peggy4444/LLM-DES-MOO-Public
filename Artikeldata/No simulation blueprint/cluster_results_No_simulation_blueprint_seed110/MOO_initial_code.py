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


# Wrapper around the provided production_line_run to evaluate one replication
def _single_replication(seed_offset, caps):
    # caps order:
    # 0: post_loading
    # 1: post_conveyor
    # 2: post_washing
    # 3: pre_press1
    # 4: pre_press2
    # 5: post_press12
    # We pass capacities via globals by temporarily monkey-patching if needed.
    # Here we call production_line_run directly and then aggregate its result.
    rep_results = []
    production_line_run(rep_results, seed_offset=seed_offset)
    # rep_results is a list of tuples (thp, mean_wip, energy_per_part)
    thp = rep_results[0][0]
    wip = rep_results[0][1]
    return thp, wip


def run_simulation_for_caps(caps, n_replications=10, base_seed=RANDOM_SEED):
    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed_offset = r * 1000 + local_rng.randint(0, 1000000)
        rep_results = []
        production_line_run(rep_results, seed_offset=seed_offset)
        thp, wip, _ = rep_results[0]
        throughputs.append(thp)
        wips.append(wip)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    Decision variables (all integer in [1,10]):
        x[0] = PostLoadingBuffer capacity
        x[1] = PostConveyorBuffer capacity
        x[2] = PostWashingBuffer capacity
        x[3] = PrePress1Buffer capacity
        x[4] = PrePress2Buffer capacity
        x[5] = PostPress1&Press2Buffer capacity
    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    Constraint:
        Sum of all buffer capacities <= 30.
        If violated, return a large penalty so solution is dominated and
        will not be exported.
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Constraint: total capacity <= 30
    if sum(caps) > 30:
        # Penalize heavily; these will be filtered out when exporting
        return [1e6, 1e6]

    avg_throughput, avg_wip = run_simulation_for_caps(caps, n_replications, base_seed)

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables:
        x[0] = PostLoadingBuffer capacity
        x[1] = PostConveyorBuffer capacity
        x[2] = PostWashingBuffer capacity
        x[3] = PrePress1Buffer capacity
        x[4] = PrePress2Buffer capacity
        x[5] = PostPress1&Press2Buffer capacity

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 xl=None,
                 xu=None,
                 n_replications=10,
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
    n_replications=10,
    base_seed=RANDOM_SEED,
    verbose=True
):
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
    Infeasible points (sum of capacities > 30) are not written.
    Penalized points (with very large objective values) are also skipped.
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
            caps = [int(v) for v in x[:6]]
            # Apply same constraint check: skip infeasible
            if sum(caps) > 30:
                continue
            wip = float(f[0])
            throughput = float(-f[1])
            # Skip penalized points
            if wip >= 1e6 or -throughput >= 1e6:
                continue
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
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=10,
        verbose=True
    )
    export_history_to_csv(result, filename="moo_simulation_results.csv")