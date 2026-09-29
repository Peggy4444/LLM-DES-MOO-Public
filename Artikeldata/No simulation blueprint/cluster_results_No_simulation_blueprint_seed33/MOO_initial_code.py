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


# Assumes the simulation code with:
# - RANDOM_SEED
# - SIM_TIME
# - WARMUP
# - REPLICATIONS
# - ProductionSystem
# - run_replication(seed)  OR we re-implement a replication wrapper here
# is already imported in the same namespace.


def run_simulation(seed, caps):
    """
    Run one full simulation replication for a given set of buffer capacities.

    caps: list/array of 6 integers in [1,10] for:
        0: post_loading_buffer
        1: post_conveyor_buffer
        2: post_washing_buffer
        3: pre_press1_buffer
        4: pre_press2_buffer
        5: post_press_buffer
    """
    random.seed(seed)
    env = simpy.Environment()

    # Create system with default interarrival
    system = ProductionSystem(env, interarrival=60.0)

    # Override buffer capacities with decision variables
    caps = list(caps)[:6]
    (
        cap_post_loading,
        cap_post_conveyor,
        cap_post_washing,
        cap_pre_press1,
        cap_pre_press2,
        cap_post_press,
    ) = caps

    system.post_loading_buffer.capacity = int(cap_post_loading)
    system.post_loading_buffer.store.capacity = int(cap_post_loading)

    system.post_conveyor_buffer.capacity = int(cap_post_conveyor)
    system.post_conveyor_buffer.store.capacity = int(cap_post_conveyor)

    system.post_washing_buffer.capacity = int(cap_post_washing)
    system.post_washing_buffer.store.capacity = int(cap_post_washing)

    system.pre_press1_buffer.capacity = int(cap_pre_press1)
    system.pre_press1_buffer.store.capacity = int(cap_pre_press1)

    system.pre_press2_buffer.capacity = int(cap_pre_press2)
    system.pre_press2_buffer.store.capacity = int(cap_pre_press2)

    system.post_press_buffer.capacity = int(cap_post_press)
    system.post_press_buffer.store.capacity = int(cap_post_press)

    env.run(until=SIM_TIME)

    # KPIs over full run (consistent with provided simulation code)
    th = system.throughput_count * 3600.0 / SIM_TIME
    avg_wip = system.wip_time_integral / SIM_TIME

    return {
        "throughput": th,
        "wip": avg_wip,
        "produced_parts": system.throughput_count,
    }


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [f1, f2] = [avg_wip, -avg_throughput].
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    throughputs = []
    wips = []

    # Local RNG to decorrelate seeds across processes
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1_000_000)
        res = run_simulation(seed, caps)
        throughputs.append(res["throughput"])
        wips.append(res["wip"])

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integer in [1,10]):
        x[0] = post_loading_buffer capacity
        x[1] = post_conveyor_buffer capacity
        x[2] = post_washing_buffer capacity
        x[3] = pre_press1_buffer capacity
        x[4] = pre_press2_buffer capacity
        x[5] = post_press_buffer capacity

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(
        self,
        n_var=6,
        n_obj=2,
        n_constr=0,
        xl=None,
        xu=None,
        n_replications=REPLICATIONS,
        base_seed=RANDOM_SEED,
        n_cores=50,
    ):
        if xl is None:
            xl = np.array([1] * n_var, dtype=int)
        if xu is None:
            xu = np.array([10] * n_var, dtype=int)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
        )

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

        # Use exactly 50 cores
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=REPLICATIONS,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50,
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
        n_cores=n_cores,
    )

    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(prob=1.0 / problem.n_var, eta=20)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=base_seed,
        save_history=True,
        verbose=verbose,
    )

    return res


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export all solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Only feasible points are exported (all here are feasible since bounds are enforced).
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
        "post_press_cap",
        "wip",
        "throughput",
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
            throughput = float(-f[1])

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "post_loading_cap": caps[0],
                "post_conveyor_cap": caps[1],
                "post_washing_cap": caps[2],
                "pre_press1_cap": caps[3],
                "pre_press2_cap": caps[4],
                "post_press_cap": caps[5],
                "wip": wip,
                "throughput": throughput,
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
        n_cores=50,
    )

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")