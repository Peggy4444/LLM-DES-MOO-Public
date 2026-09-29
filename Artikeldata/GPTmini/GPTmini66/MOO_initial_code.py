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

# The simulation functions and classes (DelayBuffer, Machine, part_generator, reset_machine_stats,
# WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED, run_simulation) are expected to be available in the
# same module / namespace where this MOO script is integrated. This script calls run_simulation(seed, caps, warmup, measure_until)
# and therefore the simulation must provide a compatible function signature:
#   run_simulation(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL)
#
# Decision variables correspond to all buffer capacities (discrete integers 1..10).
# This MOO script expects the simulation's run_simulation to accept 'caps' (an iterable of ints)
# and to build the production model using those buffer capacities.

# -----------------------------------------------------------------------------
# Simulation evaluation wrapper and worker
# -----------------------------------------------------------------------------
def evaluate_single_individual(args):
    """
    Evaluate a single individual (vector of buffer capacities) over multiple replications.
    Returns [avg_wip, -avg_throughput] (pymoo minimizes objectives).
    If evaluation fails or individual is invalid, returns [np.nan, np.nan].
    """
    x, n_replications, base_seed, warmup, measure_until = args
    try:
        # Ensure capacities are integers
        caps = [int(v) for v in x]
    except Exception:
        return [np.nan, np.nan]

    throughputs = []
    wips = []

    # local RNG to produce distinct seeds
    local_rng = random.Random()
    # combine base_seed and capacities to vary seedspace deterministically per solution
    local_rng.seed(int(base_seed) + sum(caps))

    for r in range(n_replications):
        seed = int(base_seed) + r + local_rng.randint(0, 1_000_000)
        try:
            # The simulation function is expected to accept (seed, caps, warmup, measure_until)
            res = run_simulation(seed, caps, warmup=warmup, measure_until=measure_until)
            throughputs.append(res["overall"]["throughput"])
            wips.append(res["overall"]["wip"])
        except Exception:
            # If any replication crashes, mark this individual invalid
            return [np.nan, np.nan]

    # If no successful replication, invalid
    if not throughputs or not wips:
        return [np.nan, np.nan]

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    return [avg_wip, -avg_throughput]

# -----------------------------------------------------------------------------
# Problem definition for pymoo
# -----------------------------------------------------------------------------
class BufferCapacityProblem(Problem):
    """
    Multi-objective problem:
      - decision variables: capacities for all delay buffers (integers 1..10)
      - objectives: f1 = avg WIP (min), f2 = -avg throughput (min in pymoo)
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 xl=None,
                 xu=None,
                 n_replications=5,
                 base_seed=66,
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

        self.n_replications = int(n_replications)
        self.base_seed = int(base_seed)
        self.n_cores = int(n_cores)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X: population matrix shape (n_individuals, n_var)
        We skip evaluating invalid individuals (those with values outside bounds or non-integers)
        and we mark them as NaN in the objectives. These NaN entries will be filtered out
        when exporting results.
        """
        X = np.asarray(X)
        n_individuals = X.shape[0]

        # Validate individuals (must be integers and within bounds)
        valid_mask = []
        for i in range(n_individuals):
            row = X[i]
            # check integerness (allow floats that are integer-valued)
            is_int_vals = all(math.isfinite(v) and float(v).is_integer() for v in row)
            in_bounds = all((row >= self.xl) & (row <= self.xu))
            valid_mask.append(bool(is_int_vals and in_bounds))
        valid_mask = np.array(valid_mask, dtype=bool)

        tasks = []
        indices = []
        for i in range(n_individuals):
            if valid_mask[i]:
                # package the task
                tasks.append((X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL))
                indices.append(i)

        # Prepare result array filled with NaNs
        F = np.full((n_individuals, self.n_obj), np.nan, dtype=float)

        if tasks:
            # Map across processes using exact pool size specified
            # Use context manager to ensure proper cleanup
            with multiprocessing.Pool(processes=self.n_cores) as pool:
                results = pool.map(evaluate_single_individual, tasks)

            # Fill results back in order
            for idx, res in zip(indices, results):
                try:
                    F[idx, 0] = float(res[0])
                    F[idx, 1] = float(res[1])
                except Exception:
                    F[idx, :] = np.nan

        out["F"] = F

# -----------------------------------------------------------------------------
# NSGA-II run helper
# -----------------------------------------------------------------------------
def run_nsga2_optimization(pop_size=50,
                           n_gen=50,
                           n_replications=5,
                           base_seed=66,
                           verbose=True,
                           n_cores=50):
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1, 1, 1, 1, 1, 1]),
        xu=np.array([10, 10, 10, 10, 10, 10]),
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

# -----------------------------------------------------------------------------
# CSV export (skips invalid / NaN solutions)
# -----------------------------------------------------------------------------
def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export solutions from every generation (initial population included) to CSV.
    Invalid solutions (objective contains NaN) are omitted.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        # buffer capacities (6 variables)
        "post_loading_buffer_cap",
        "post_conveyor_buffer_cap",
        "post_washing_buffer_cap",
        "pre_press1_buffer_cap",
        "pre_press2_buffer_cap",
        "post_press12_buffer_cap",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")

        # F may contain NaNs for invalid individuals; skip those
        for ind_idx, (x, f) in enumerate(zip(X, F)):
            # skip invalid
            if np.any(np.isnan(f)):
                continue
            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "post_loading_buffer_cap": caps[0],
                "post_conveyor_buffer_cap": caps[1],
                "post_washing_buffer_cap": caps[2],
                "pre_press1_buffer_cap": caps[3],
                "pre_press2_buffer_cap": caps[4],
                "post_press12_buffer_cap": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------------------------------------------------------
# Main execution (guarded for multiprocessing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Force exactly 50 cores as requested
    N_CORES = 50

    # NSGA-II parameters as requested: population size 50, 50 generations
    POP_SIZE = 50
    N_GEN = 50

    # Number of replications per individual evaluation (can be tuned)
    N_REPLICATIONS = 5

    # Base random seed (expected to be defined in simulation as well)
    BASE_SEED = 66

    res = run_nsga2_optimization(
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        n_replications=N_REPLICATIONS,
        base_seed=BASE_SEED,
        verbose=True,
        n_cores=N_CORES
    )

    export_history_to_csv(res, filename="moo_simulation_results.csv")