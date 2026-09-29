import random
import statistics
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Assumes the simulation module is available in the same environment
from simulation import (
    RANDOM_SEED,
    WARMUP_SECONDS,
    MEASURE_UNTIL,
    run_simulation,
)


# Baseline process times from the given simulation model
BASE_PROCESS_TIMES = {
    "Loading robot": 12.0,
    "Conveyor belt": 6.0,
    "Washing machine": 14.0,
    "Hantering cell": 25.0,
    "Presses cell 1": 175.0,
    "Presses cell 2": 176.0,
    "Quality station cell": 41.0,
}

MACHINE_ORDER = [
    "Loading robot",
    "Conveyor belt",
    "Washing machine",
    "Hantering cell",
    "Presses cell 1",
    "Presses cell 2",
    "Quality station cell",
]


def run_simulation_with_process_flags(seed, flags, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Wrapper around the original run_simulation that applies process time
    reduction flags to the global simulation model.

    flags: array-like of length 7, each element in {0,1}
           0 -> keep baseline process time
           1 -> reduce process time by 10%
    Constraint: no process time may be reduced by more than 10%.
                Since flags are binary, this is always satisfied.
                If any flag is outside {0,1}, return None (invalid).
    """
    # Validate flags are 0 or 1
    for f in flags:
        if f not in (0, 1):
            return None

    # Compute new process times (baseline or 10% reduction)
    new_process_times = {}
    for name, flag in zip(MACHINE_ORDER, flags):
        base = BASE_PROCESS_TIMES[name]
        if flag == 1:
            val = base * 0.9
        else:
            val = base
        # Constraint: process time cannot be reduced by more than 10%
        # (already enforced by construction). If violated, return None.
        if val < base * 0.9 - 1e-9:
            return None
        new_process_times[name] = val

    # We need to call the original run_simulation but with modified process times.
    # To keep compatibility, we assume run_simulation can be monkey-patched
    # via a global configuration dictionary in the simulation module.
    # Here we set a global variable that the simulation code reads.
    from simulation import PROCESS_TIME_OVERRIDES

    # Backup old overrides
    old_overrides = PROCESS_TIME_OVERRIDES.copy()
    try:
        PROCESS_TIME_OVERRIDES.clear()
        PROCESS_TIME_OVERRIDES.update(new_process_times)
        res = run_simulation(seed, warmup=warmup, measure_until=measure_until)
    finally:
        PROCESS_TIME_OVERRIDES.clear()
        PROCESS_TIME_OVERRIDES.update(old_overrides)

    return res


def evaluate_single_replication(args):
    """
    Helper function for parallel execution of a single replication.
    args: (ind_idx, rep_idx, flags, base_seed)
    Returns: (ind_idx, rep_idx, throughput, wip, energy_per_part) or
             (ind_idx, rep_idx, None, None, None) if infeasible.
    """
    ind_idx, rep_idx, flags, base_seed = args
    seed = base_seed + rep_idx + random.randint(0, 1000000)

    res = run_simulation_with_process_flags(seed, flags, WARMUP_SECONDS, MEASURE_UNTIL)
    if res is None:
        return ind_idx, rep_idx, None, None, None

    throughput = res["overall"]["throughput"]
    wip = res["overall"]["wip"]
    produced_parts = res["overall"]["produced_parts"]

    total_energy_run = sum(mdata["total_energy"] for mdata in res["machine_energy"].values())
    energy_per_part = (total_energy_run / produced_parts) if produced_parts > 0 else float("inf")

    return ind_idx, rep_idx, throughput, wip, energy_per_part


class ProcessTimeReductionProblem(Problem):
    """
    Multi-objective optimization problem for machine process time reductions.

    Decision variables (binary flags, 0 or 1):
        x[0] = Loading robot process time reduction 10% flag
        x[1] = Conveyor belt process time reduction 10% flag
        x[2] = Washing machine process time reduction 10% flag
        x[3] = Hantering cell process time reduction 10% flag
        x[4] = Presses cell 1 process time reduction 10% flag
        x[5] = Presses cell 2 process time reduction 10% flag
        x[6] = Quality station cell process time reduction 10% flag

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = average energy per part (to be minimized)

    Constraint handling:
        If a point violates the constraint (process time reduced by more than 10%
        or invalid flags), it is not evaluated and is treated as infeasible.
        Such points are not exported to the CSV file.
    """

    def __init__(
        self,
        n_var=7,
        n_obj=2,
        n_constr=0,
        xl=None,
        xu=None,
        n_replications=10,
        base_seed=RANDOM_SEED,
    ):
        if xl is None:
            xl = np.array([0] * n_var, dtype=int)
        if xu is None:
            xu = np.array([1] * n_var, dtype=int)

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

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=int)
        n_individuals = X.shape[0]

        F = np.full((n_individuals, 2), np.nan, dtype=float)

        # Prepare tasks for parallel execution
        tasks = []
        for i in range(n_individuals):
            flags = X[i, :]
            # Quick feasibility check: flags must be 0 or 1
            if not np.all(np.logical_or(flags == 0, flags == 1)):
                continue
            for r in range(self.n_replications):
                tasks.append((i, r, flags.copy(), self.base_seed))

        # Parallel execution using exactly 50 processes
        results_by_ind = {i: [] for i in range(n_individuals)}
        with ProcessPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(evaluate_single_replication, t) for t in tasks]
            for fut in as_completed(futures):
                ind_idx, rep_idx, throughput, wip, energy_per_part = fut.result()
                if throughput is None or wip is None or energy_per_part is None:
                    # Mark this individual as infeasible by leaving NaNs
                    results_by_ind[ind_idx] = None
                else:
                    if results_by_ind[ind_idx] is not None:
                        results_by_ind[ind_idx].append((throughput, wip, energy_per_part))

        # Aggregate results
        for i in range(n_individuals):
            vals = results_by_ind[i]
            if vals is None or len(vals) == 0:
                # Infeasible or failed; keep NaNs
                continue
            throughputs = [v[0] for v in vals]
            wips = [v[1] for v in vals]
            energies = [v[2] for v in vals]

            avg_wip = statistics.mean(wips)
            avg_energy = statistics.mean(energies)

            F[i, 0] = avg_wip
            F[i, 1] = avg_energy

        out["F"] = F


def run_nsga2_optimization(
    pop_size=10,
    n_gen=30,
    n_replications=10,
    base_seed=RANDOM_SEED,
    verbose=True,
):
    problem = ProcessTimeReductionProblem(
        n_var=7,
        n_obj=2,
        xl=np.array([0] * 7),
        xu=np.array([1] * 7),
        n_replications=n_replications,
        base_seed=base_seed,
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
    Export all feasible solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Only feasible points (i.e., those with finite objective values) are exported.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "loading_robot_process_time_reduction10percent_flag",
        "conveyor_belt_process_time_reduction10percent_flag",
        "washing_machine_process_time_reduction10percent_flag",
        "hantering_cell_process_time_reduction10percent_flag",
        "presses_cell1_process_time_reduction10percent_flag",
        "presses_cell2_process_time_reduction10percent_flag",
        "quality_station_cell_process_time_reduction10percent_flag",
        "wip",
        "energy_per_part",
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")

        for ind_idx, (x, f) in enumerate(zip(X, F)):
            wip = float(f[0])
            energy_per_part = float(f[1])

            # Skip infeasible or non-evaluated points (NaN or inf)
            if not np.isfinite(wip) or not np.isfinite(energy_per_part):
                continue

            flags = [int(v) for v in x[:7]]
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "loading_robot_process_time_reduction10percent_flag": flags[0],
                "conveyor_belt_process_time_reduction10percent_flag": flags[1],
                "washing_machine_process_time_reduction10percent_flag": flags[2],
                "hantering_cell_process_time_reduction10percent_flag": flags[3],
                "presses_cell1_process_time_reduction10percent_flag": flags[4],
                "presses_cell2_process_time_reduction10percent_flag": flags[5],
                "quality_station_cell_process_time_reduction10percent_flag": flags[6],
                "wip": wip,
                "energy_per_part": energy_per_part,
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = run_nsga2_optimization(
        pop_size=10,
        n_gen=30,
        n_replications=10,
        verbose=True,
    )
    export_history_to_csv(result, filename="moo_simulation_results.csv")