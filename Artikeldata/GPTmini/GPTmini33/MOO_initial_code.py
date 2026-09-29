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

# NOTE:
# This MOO script is intended to be integrated with the simulation model code
# (the large simpy-based factory model you already have in another file or in
# the same script). The MOO functions below assume that the following symbols
# are available in the execution namespace:
#   - simpy, DelayBuffer, Machine, part_generator, reset_machine_stats
#   - splitter, merger, kwh_per_sec
#   - WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED
#
# If you place this MOO code in a separate file, import those names from the
# module that contains your simulation model. If you append this code to the
# simulation file, it will directly reuse the simulation classes and helpers.

# Decision variables: capacities for the following DelayBuffers (6 buffers):
#   post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
#   pre_press1_buffer, pre_press2_buffer, post_press12_buffer
#
# All capacities are integers in [1, 10].

# Number of CPU cores to use for parallel evaluation (exactly 50 as requested)
PARALLEL_CORES = 50


def is_feasible_caps(caps):
    """Feasibility check for a candidate capacity vector.
    Currently enforces integer membership and bounds [1,10] for each buffer.
    Extend this function if you have additional constraints.
    """
    if len(caps) != 6:
        return False
    for v in caps:
        try:
            iv = int(v)
        except Exception:
            return False
        if iv < 1 or iv > 10:
            return False
    return True


def run_simulation_with_caps(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Build and run the full factory simulation using the provided capacities.

    This function mirrors the topology and parameters of the provided
    simulation model but makes the 6 DelayBuffer capacities configurable.

    It assumes all helper classes/functions (DelayBuffer, Machine, part_generator,
    reset_machine_stats, splitter, merger, kwh_per_sec, etc.) are present.
    """
    # Ensure integer capacities
    caps = [int(v) for v in caps]
    (cap_post_loading,
     cap_post_conveyor,
     cap_post_washing,
     cap_pre_press1,
     cap_pre_press2,
     cap_post_press12) = caps

    # Use the same run_simulation structure as your model but with variable caps.
    random.seed(seed)
    env = simpy.Environment()

    # Raw input capacity
    raw_input = simpy.Store(env, capacity=1000)

    # Delay buffers with variable capacities and delays matching the original model
    post_loading_buffer = DelayBuffer(env, cap=cap_post_loading, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=cap_post_conveyor, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=cap_post_washing, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=cap_pre_press1, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=cap_pre_press2, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=cap_post_press12, delay=32)

    # Final sinks
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Helper stores (immediate buffers)
    pre_press1 = simpy.Store(env, capacity=3)
    pre_press2 = simpy.Store(env, capacity=3)
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    # Machines with the same parameters as the original model
    loading_robot = Machine(
        env, "Loading robot", input_buffer=raw_input, output_buffer=post_loading_buffer,
        process_time=12.0, availability=90.49, mttr=68.0,
        working_power=kwh_per_sec(0.72), waiting_power=kwh_per_sec(0.25),
    )

    conveyor_belt = Machine(
        env, "Conveyor belt", input_buffer=post_loading_buffer, output_buffer=post_conveyor_buffer,
        process_time=6.0, availability=100.0, mttr=1.0,
        working_power=kwh_per_sec(0.0), waiting_power=kwh_per_sec(0.0),
    )

    washing_machine = Machine(
        env, "Washing machine", input_buffer=post_conveyor_buffer, output_buffer=post_washing_buffer,
        process_time=14.0, availability=80.89, mttr=269.0,
        working_power=kwh_per_sec(35.24), waiting_power=kwh_per_sec(4.28),
    )

    hantering_cell = Machine(
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_press1,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

    # Split stream from pre_press1 into two parallel pre-press buffers evenly
    env.process(splitter(env, pre_press1, pre_press1_buffer, pre_press2_buffer))

    presses_cell1 = Machine(
        env, "Presses cell 1", input_buffer=pre_press1_buffer, output_buffer=press1_out,
        process_time=175.0, availability=87.79, mttr=73.0,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    presses_cell2 = Machine(
        env, "Presses cell 2", input_buffer=pre_press2_buffer, output_buffer=press2_out,
        process_time=176.0, availability=87.69, mttr=74.0,
        working_power=kwh_per_sec(1.27), waiting_power=kwh_per_sec(1.25),
    )

    # Merge outputs of both presses into the post-press delay buffer
    merger(env, press1_out, press2_out, post_press12_buffer)

    quality_station = Machine(
        env, "Quality station cell", input_buffer=post_press12_buffer, output_buffer=sink,
        process_time=41.0, availability=85.87, mttr=66.0,
        working_power=kwh_per_sec(0.84), waiting_power=kwh_per_sec(0.58),
        defect_rate=0.089, defect_sink=defects,
    )

    machines_list = [
        loading_robot, conveyor_belt, washing_machine,
        hantering_cell, presses_cell1, presses_cell2, quality_station,
    ]

    # Feed raw input
    env.process(part_generator(env, raw_input))

    # Warmup run
    env.run(until=warmup)

    # Reset machine stats after warmup
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    delay_buffers = [
        post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
        pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    ]

    def sample_wip(env):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            # Sample every 10 minutes (600s).
            yield env.timeout(600)

    env.process(sample_wip(env))
    env.run(until=measure_until)

    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0

    result = {"overall": {
        "throughput": throughput,
        "wip": avg_wip,
        "produced_parts": total_produced},
        "machine_energy": {}}

    for m in machines_list:
        waiting_energy = m.waiting_energy_consumption()
        working_energy = m.working_energy_consumption()
        total_energy = waiting_energy + working_energy
        result["machine_energy"][m.name] = {
            "working_time": m.working_time,
            "waiting_time": m.failed_time_total + m.blocked_time,
            "working_energy": working_energy,
            "waiting_energy": waiting_energy,
            "total_energy": total_energy}

    return result


def evaluate_single_individual(args):
    """
    Evaluate one individual (vector of 6 capacities) over multiple replications.
    args: (x, n_replications, base_seed, warmup, measure_until)
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x]

    # Feasibility check: if infeasible, do not run simulations
    if not is_feasible_caps(caps):
        # Return large penalty objectives (pymoo minimizes). We avoid running sims.
        return [1e9, 1e9]

    throughputs = []
    wips = []

    # Use local RNG to produce distinct seeds per replication
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 10**6)
        res = run_simulation_with_caps(seed, caps, warmup=warmup, measure_until=measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Objectives: minimize wip, minimize -throughput (i.e., maximize throughput)
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective problem with 6 integer decision variables (buffer capacities).
    """

    def __init__(self, n_var=6, n_obj=2, n_constr=0,
                 xl=None, xu=None,
                 n_replications=5,
                 base_seed=RANDOM_SEED,
                 n_cores=PARALLEL_CORES):

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

        # Prepare tasks for only feasible individuals. For infeasible individuals
        # we will fill the objective with large penalty values and skip simulation.
        tasks = []
        task_indices = []
        for i in range(n_individuals):
            caps = X[i].tolist()
            if is_feasible_caps(caps):
                tasks.append((X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL))
                task_indices.append(i)

        # Initialize result container with penalties for infeasible individuals
        F = np.full((n_individuals, self.n_obj), fill_value=1e9, dtype=float)

        if tasks:
            # Use a multiprocessing pool with the specified number of cores
            with multiprocessing.Pool(processes=self.n_cores) as pool:
                results = pool.map(evaluate_single_individual, tasks)

            # Map results back to F
            for idx, res in zip(task_indices, results):
                F[idx, :] = res

        out["F"] = F


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=PARALLEL_CORES
):
    """
    Execute NSGA-II optimization for buffer capacities.
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
    Export feasible solutions from every generation (including initial population)
    along with their KPIs and decision variables to a CSV file. Infeasible
    solutions are skipped.
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
            caps = [int(v) for v in x]
            # Skip infeasible individuals (either detected by caps bounds or penalty)
            if not is_feasible_caps(caps):
                continue
            # Also skip individuals that carry the large-penalty objective (not evaluated)
            if np.any(np.isclose(f, 1e9)):
                continue

            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
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
    # Run NSGA-II with requested configuration: pop size 50, 50 generations,
    # using exactly PARALLEL_CORES for parallel evaluation.
    res = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        base_seed=RANDOM_SEED,
        verbose=True,
        n_cores=PARALLEL_CORES
    )

    export_history_to_csv(res, filename="moo_simulation_results.csv")