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

# Note: This MOO code expects the simulation functions and constants
# (run_simulation, WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED) to be
# available in the same Python module or imported into the namespace
# where this code is integrated. Do not redefine run_simulation here.

PENALTY_OBJ = 1e9  # large penalty for infeasible individuals


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    If the candidate is infeasible (violates bounds or integrality), the
    simulation is NOT executed and a large penalty objective is returned.
    """
    x, n_replications, base_seed, warmup, measure_until = args
    # Expecting x length to match number of buffer decision variables (6)
    caps = [int(round(v)) for v in x]

    # Feasibility check: all capacities must be integers in [1,10]
    if any((c < 1 or c > 10) for c in caps):
        # Infeasible: return large penalty objectives and skip simulation
        return [PENALTY_OBJ, PENALTY_OBJ]

    throughputs = []
    wips = []

    # Local RNG to generate distinct seeds per replication without collisions across processes
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        # The run_simulation function must be defined in the simulation module
        res = run_simulation(seed, caps, warmup=warmup, measure_until=measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Return objectives: minimize WIP, minimize -throughput (so throughput is maximized)
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables: capacities for the six DelayBuffer objects in the simulation:
        x[0] = post_loading_buffer cap
        x[1] = post_conveyor_buffer cap
        x[2] = post_washing_buffer cap
        x[3] = pre_press1_buffer cap
        x[4] = pre_press2_buffer cap
        x[5] = post_press12_buffer cap

    All are integer in [1, 10].
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 xl=None,
                 xu=None,
                 n_replications=5,
                 base_seed=0,
                 n_cores=1):
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

        # Package arguments for each individual
        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # Map tasks across available CPU cores using exactly self.n_cores processes
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
    Uses IntegerRandomSampling and integer decision variables in [1,10].
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

    Solutions that were infeasible and therefore received penalty objectives
    are excluded from the CSV (rows with objectives >= PENALTY threshold).
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "post_loading_buffer",
        "post_conveyor_buffer",
        "post_washing_buffer",
        "pre_press1_buffer",
        "pre_press2_buffer",
        "post_press12_buffer",
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
            # Skip infeasible/penalized individuals
            if f[0] >= PENALTY_OBJ * 0.1 or f[1] >= PENALTY_OBJ * 0.1:
                continue

            caps = [int(round(v)) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "post_loading_buffer": caps[0],
                "post_conveyor_buffer": caps[1],
                "post_washing_buffer": caps[2],
                "pre_press1_buffer": caps[3],
                "pre_press2_buffer": caps[4],
                "post_press12_buffer": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization with population size 50 and 50 generations,
    # using exactly 50 CPU cores for parallel simulation evaluations.
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        base_seed=RANDOM_SEED,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")