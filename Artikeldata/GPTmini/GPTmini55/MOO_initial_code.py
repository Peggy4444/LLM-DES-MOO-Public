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

# Constants expected to be provided by the simulation module
# If running standalone, ensure these are defined in the simulation code:
# RANDOM_SEED, WARMUP_SECONDS, MEASURE_UNTIL
try:
    RANDOM_SEED
    WARMUP_SECONDS
    MEASURE_UNTIL
except NameError:
    # Fallback defaults if not defined in the simulation module; the simulation file
    # should normally provide these constants.
    RANDOM_SEED = 55
    WARMUP_SECONDS = 86400
    MEASURE_UNTIL = 691200

def evaluate_single_individual(args):
    """
    Evaluate one individual (one set of buffer capacities) over multiple replications.
    Returns a tuple (F_list, G_value) where:
      - F_list = [avg_wip, -avg_throughput] (pymoo minimizes)
      - G_value <= 0 means feasible, > 0 means infeasible (constraint violation)
    Feasibility rule: a configuration is considered infeasible if the simulation
    produces zero parts across replications or if any replication raises an exception.
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x]
    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    try:
        for r in range(n_replications):
            seed = base_seed + r + local_rng.randint(0, 1_000_000)

            # Try to call a simulation entrypoint that accepts buffer capacities.
            # The simulation code provided alongside this MOO script must expose a
            # run_simulation function that accepts (seed, caps=..., warmup=..., measure_until=...).
            # If such a signature is not present, a TypeError will be raised and propagated
            # to the outer except which will mark the individual as infeasible.
            res = run_simulation(seed, caps=caps, warmup=warmup, measure_until=measure_until)

            tp = res["overall"]["throughput"]
            wp = res["overall"]["wip"]
            produced = res["overall"].get("produced_parts", None)

            # Mark infeasible if produced parts are zero or missing
            if produced is None or produced == 0 or tp == 0.0:
                return ([1e6, 1e6], 1.0)

            throughputs.append(tp)
            wips.append(wp)

        avg_throughput = statistics.mean(throughputs)
        avg_wip = statistics.mean(wips)

        return ([avg_wip, -avg_throughput], -1.0)

    except Exception:
        # Any exception during simulation marks the individual infeasible
        return ([1e6, 1e6], 1.0)


class BufferCapacityProblem(Problem):
    """
    Multi-objective problem: minimize WIP and maximize throughput (as -throughput).

    Decision variables: capacities for all delay buffers in the provided simulation model.
    For the provided production model these are:
      post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
      pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    => 6 decision variables, each integer in [1, 10].
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
        self.n_cores = n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=int)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # Use exactly self.n_cores CPU processes
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        F = np.zeros((n_individuals, self.n_obj), dtype=float)
        G = np.zeros((n_individuals, self.n_constr), dtype=float)

        for i, (f_vals, g_val) in enumerate(results):
            F[i, :] = f_vals
            G[i, 0] = g_val

        out["F"] = F
        out["G"] = G


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export feasible solutions from each generation to CSV.
    Rows corresponding to infeasible individuals (G > 0) are omitted.
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
            # If constraint information is present, skip infeasible individuals
            if G is not None:
                try:
                    if G[ind_idx, 0] > 0:
                        continue
                except Exception:
                    pass

            caps = [int(v) for v in x]
            wip = float(F[ind_idx, 0])
            throughput = float(-F[ind_idx, 1])  # second objective stored as -throughput
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


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=3,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50
):
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


if __name__ == "__main__":
    # Ensure the multiprocessing start method is set (optional, but can help on some platforms)
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    # Run with exactly 50 cores, population 50, 50 generations
    result = run_nsga2_optimization(pop_size=50, n_gen=50, n_replications=3, verbose=True, n_cores=50)

    export_history_to_csv(result, filename="moo_simulation_results.csv")