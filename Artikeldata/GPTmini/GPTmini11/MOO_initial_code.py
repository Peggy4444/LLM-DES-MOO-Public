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

# Note: This MOO script is intended to be executed in the same environment/module
# as the simulation code provided separately. It relies on names like:
#   - run_simulation(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL)
#   - WARMUP_SECONDS, MEASURE_UNTIL
#   - RANDOM_SEED
# and helper classes/functions (DelayBuffer, Machine, part_generator, reset_machine_stats, kwh_per_sec, etc.)
# being available in the import namespace (e.g., by running from the same directory/module).


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns a tuple (F, G) where:
      - F is a list of two objectives [wip, -throughput]
      - G is a list of constraint values (<=0 feasible). We use 1.0 to indicate infeasible, 0.0 feasible.
    """
    x, n_replications, base_seed, warmup, measure_until = args

    # Ensure we have exactly 6 decision variables (all buffer capacities)
    # Cast/round and clip to integer bounds [1,10]
    caps_raw = list(x)
    if len(caps_raw) < 6:
        # Pad with minimum capacity if fewer provided
        caps_raw = (caps_raw + [1] * 6)[:6]

    # Convert to integer capacities
    caps = []
    for v in caps_raw[:6]:
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = None
        caps.append(iv)

    # Validate capacities: must be integers in [1,10]
    feasible = True
    for iv in caps:
        if iv is None or iv < 1 or iv > 10:
            feasible = False
            break

    # If infeasible, don't run simulations. Return a large objective penalty and non-zero constraint.
    if not feasible:
        F = [1e6, 1e6]
        G = [1.0]  # >0 indicates constraint violation in pymoo (infeasible)
        return (F, G)

    # Otherwise perform simulation replications
    throughputs = []
    wips = []

    # Use a local RNG to generate distinct seeds for each replication
    local_rng = random.Random()
    # Mix capacities into the seed to provide deterministic yet distinct seeds across different designs
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1_000_000)
        res = run_simulation(seed, caps, warmup, measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Objectives: minimize WIP, minimize -throughput (i.e., maximize throughput)
    F = [avg_wip, -avg_throughput]
    G = [0.0]  # feasible
    return (F, G)


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for all buffer capacities (6 buffers).
    Decision variables:
        x[0..5] = capacities for the 6 delay buffers (integers in [1,10])

    Objectives:
        f1 = average WIP (minimize)
        f2 = -average throughput (minimize => maximizes throughput)
    """

    def __init__(self, n_var=6, n_obj=2, n_constr=1,
                 xl=None, xu=None,
                 n_replications=5,
                 base_seed=0,
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

        # 1. Package the arguments for each individual
        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # 2. Map the tasks across the specified CPU cores
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        # 3. Build F and G arrays from results
        F = np.zeros((n_individuals, self.n_obj), dtype=float)
        G = np.zeros((n_individuals, self.n_constr), dtype=float)

        for i, (f, g) in enumerate(results):
            # Ensure shapes
            F[i, :] = np.array(f, dtype=float)
            G[i, :] = np.array(g, dtype=float)

        out["F"] = F
        out["G"] = G


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=0,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.

    Defaults set to population size 50 and 50 generations, and uses exactly 50 cores.
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

    Infeasible individuals (G > 0) are excluded from the CSV.
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
        "buffer4_cap",
        "buffer5_cap",
        "buffer6_cap",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")  # may be None if problem had no constraints

        n_pop = X.shape[0]
        for ind_idx in range(n_pop):
            # Check feasibility: all constraints must be <= 0
            feasible = True
            if G is not None:
                # If any G value > 0 -> infeasible
                if np.any(G[ind_idx] > 0):
                    feasible = False

            if not feasible:
                continue  # skip infeasible individuals

            x = X[ind_idx]
            f = F[ind_idx]
            caps = [int(round(float(v))) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # objectives stored as -throughput

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "buffer1_cap": caps[0],
                "buffer2_cap": caps[1],
                "buffer3_cap": caps[2],
                "buffer4_cap": caps[3],
                "buffer5_cap": caps[4],
                "buffer6_cap": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Ensure we use exactly 50 cores as requested
    N_CORES = 50

    # Run NSGA-II optimization: population size 50, 50 generations
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        base_seed=RANDOM_SEED,
        verbose=True,
        n_cores=N_CORES
    )

    # Export only feasible solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")