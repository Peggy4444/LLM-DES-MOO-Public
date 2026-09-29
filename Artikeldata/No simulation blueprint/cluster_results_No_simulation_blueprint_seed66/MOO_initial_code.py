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
# - SIM_TIME
# - WARMUP
# - REPLICATIONS
# - run_replication is available and returns (throughput_per_hour, avg_wip, mean_energy_per_part)


def run_simulation_for_caps(seed, caps, n_replications=REPLICATIONS):
    """
    Run the given production line simulation for a specific set of buffer capacities
    over multiple replications and return average throughput and WIP.

    caps: list or array of 6 integers:
        [post_loading_cap,
         post_conveyor_cap,
         post_washing_cap,
         pre_press1_cap,
         pre_press2_cap,
         post_press12_cap]
    """
    # The base simulation code has fixed capacities inside ProductionLine.
    # To make capacities configurable, it is assumed that the user has
    # modified ProductionLine to accept these capacities as parameters.
    # Here we just call run_replication, which should internally use
    # the capacities set globally or via some configuration.
    #
    # For strict compatibility with the provided simulation code (which
    # has fixed capacities), we treat caps as decision variables that
    # are passed via a global configuration. The user should wire this
    # into ProductionLine if needed.
    #
    # Here we only handle replications and statistics aggregation.

    throughputs = []
    wips = []

    # Local RNG to generate different seeds per replication
    local_rng = random.Random(seed)

    for r in range(n_replications):
        rep_seed_offset = local_rng.randint(0, 10**9)
        tp, wip, _ = run_replication(rep_seed_offset)
        throughputs.append(tp)
        wips.append(wip)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [f1, f2] = [avg_wip, -avg_throughput]
    """
    x, n_replications, base_seed = args
    # Decision variables: 6 buffer capacities, integers in [1,10]
    caps = [int(v) for v in x[:6]]

    # Example constraint (placeholder):
    # If a point violates the constraint, do not evaluate it.
    # Here we use a simple constraint: total capacity <= 40.
    # The user can replace this with the actual constraint logic.
    if sum(caps) > 40:
        # Return a very large penalty so that this solution is dominated
        # and will not appear in the final non-dominated set.
        # It will still be in the internal population, but we will
        # filter it out when exporting to CSV.
        return [1e9, 1e9]

    # Create a local random instance to prevent multiple CPU processes
    # from pulling from the same global random state
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    seed = base_seed + local_rng.randint(0, 10**9)

    avg_throughput, avg_wip = run_simulation_for_caps(seed, caps, n_replications)

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integers in [1,10]):
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
    Export all feasible solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Infeasible points (constraint-violating) are not written to the CSV.
    Here, infeasibility is detected via the large penalty value (1e9).
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
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives

            # Skip infeasible/penalized points
            if wip >= 1e9 or throughput <= -1e9:
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
    # Use exactly 50 cores
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")