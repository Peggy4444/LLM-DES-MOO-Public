import multiprocessing
import random
import statistics
import os
import math
import csv

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# The simulation module must provide these symbols:
#   - run_simulation(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL)
#   - WARMUP_SECONDS
#   - MEASURE_UNTIL
#   - RANDOM_SEED
# Please make sure the simulation code is available in the Python path and
# that run_simulation accepts caps (list/tuple of 6 integers) as its 2nd argument.
#
# Example when integrating:
#   from production_sim import run_simulation, WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED
#
# This file intentionally DOES NOT import the simulation module so it can be
# integrated in different ways by the user/environment.

# Configurable defaults (will be used if the simulation module defines nothing)
try:
    WARMUP_SECONDS  # noqa: F821
except NameError:
    WARMUP_SECONDS = 86400
try:
    MEASURE_UNTIL  # noqa: F821
except NameError:
    MEASURE_UNTIL = 691200
try:
    RANDOM_SEED  # noqa: F821
except NameError:
    RANDOM_SEED = 44

def evaluate_single_individual(args):
    """
    Evaluate one individual (caps) by running multiple simulation replications.
    This function intentionally expects run_simulation to be available in the
    global namespace (provided by the simulation code integration).

    args = (x, n_replications, base_seed, warmup, measure_until)
    where x is an array-like of length 6 (buffer capacities).
    """
    x, n_replications, base_seed, warmup, measure_until = args
    # Ensure integer capacities (we expect callers to only pass integer candidates)
    caps = [int(round(float(v))) for v in x[:6]]

    # Validate caps (feasibility): must be in [1, 10]
    if any((c < 1 or c > 10) for c in caps):
        # Do not run simulation for infeasible individuals.
        # Return a large penalty so the optimizer treats it as poor but we will
        # filter infeasible points from the CSV later using the constraint values.
        return [1e9, 1e9]

    throughputs = []
    wips = []

    # Use a local RNG to create distinct seeds per replication and avoid collisions
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 2**20)
        # run_simulation must be provided by the integrated simulation module
        res = run_simulation(seed, caps, warmup=warmup, measure_until=measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 1e9

    # Pymoo minimizes objectives; we want:
    #   - minimize wip
    #   - maximize throughput -> minimize -throughput
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Optimize all buffer capacities (6 delay buffers) in the provided simulation.

    Decision variables (integers 1..10):
      x[0] = post_loading_buffer cap
      x[1] = post_conveyor_buffer cap
      x[2] = post_washing_buffer cap
      x[3] = pre_press1_buffer cap
      x[4] = pre_press2_buffer cap
      x[5] = post_press12_buffer cap

    Objectives:
      f1 = average WIP (minimize)
      f2 = -average throughput (minimize -> maximizes throughput)

    Constraint:
      All variables must be integer in [1, 10]. Infeasible individuals are not
      evaluated and are marked as infeasible (G > 0). They will be excluded
      when exporting the CSV history.
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 n_constr=1,
                 xl=None,
                 xu=None,
                 n_replications=3,
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
        # Force exactly 50 cores as requested
        self.n_cores = 50 if n_cores != 50 else n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        # Precompute feasibility per individual (integers and in [1,10])
        feasible_mask = np.full(n_individuals, True, dtype=bool)
        for i in range(n_individuals):
            xi = X[i, :self.n_var]
            # Check integer closeness and bounds
            ints = np.round(xi)
            if not np.all(np.isclose(ints, xi, atol=1e-8)):
                feasible_mask[i] = False
                continue
            if not np.all((ints >= 1) & (ints <= 10)):
                feasible_mask[i] = False

        # Prepare outputs
        F = np.full((n_individuals, self.n_obj), 1e9, dtype=float)
        # Constraint vector: G <= 0 means feasible. We set positive for infeasible.
        G = np.zeros((n_individuals, self.n_constr), dtype=float)

        # Create tasks only for feasible individuals
        tasks = []
        idx_map = []
        for i in range(n_individuals):
            if feasible_mask[i]:
                x_row = X[i]
                tasks.append((x_row, self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL))
                idx_map.append(i)
            else:
                # mark constraint violation
                G[i, 0] = 1.0

        # If there are feasible tasks, evaluate them in a process pool
        if tasks:
            # Enforce exactly 50 processes as requested. If the machine doesn't
            # have 50 cores, multiprocessing will still attempt to spawn 50 processes.
            pool_processes = 50
            with multiprocessing.Pool(processes=pool_processes) as pool:
                results = pool.map(evaluate_single_individual, tasks)

            # Fill in F for feasible individuals
            for local_idx, res in enumerate(results):
                global_idx = idx_map[local_idx]
                F[global_idx, :] = res

        out["F"] = F
        out["G"] = G


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=3,
    base_seed=RANDOM_SEED,
    verbose=True
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    Uses population size 50, for n_gen generations.
    Forces the internal worker pool to use exactly 50 processes when
    evaluating individuals.
    """
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1, 1, 1, 1, 1, 1]),
        xu=np.array([10, 10, 10, 10, 10, 10]),
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

    Infeasible individuals (where any constraint > 0) are skipped and not written.
    """
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "buffer_post_loading",
        "buffer_post_conveyor",
        "buffer_post_washing",
        "buffer_pre_press1",
        "buffer_pre_press2",
        "buffer_post_press12",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G") if "G" in pop.keys() else None

        for ind_idx, x in enumerate(X):
            # Skip infeasible if G exists and any constraint > 0
            if G is not None and np.any(G[ind_idx] > 0):
                continue

            f = F[ind_idx]
            caps = [int(round(float(v))) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "buffer_post_loading": caps[0],
                "buffer_post_conveyor": caps[1],
                "buffer_post_washing": caps[2],
                "buffer_pre_press1": caps[3],
                "buffer_pre_press2": caps[4],
                "buffer_post_press12": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization with the requested configuration:
    # population size 50, 50 generations, force using exactly 50 worker processes
    result = run_nsga2_optimization(pop_size=50, n_gen=50, n_replications=3, base_seed=RANDOM_SEED, verbose=True)

    # Export feasible solutions only
    export_history_to_csv(result, filename="moo_simulation_results.csv")