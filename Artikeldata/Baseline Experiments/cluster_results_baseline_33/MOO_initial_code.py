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


def run_simulation_with_caps(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Decision variables: capacities of all DelayBuffers (1-10)
    caps = list(map(int, caps))
    if len(caps) != 6:
        raise ValueError("Expected 6 capacity values")

    (
        cap_post_loading,
        cap_post_conveyor,
        cap_post_washing,
        cap_pre_press1,
        cap_pre_press2,
        cap_post_press12,
    ) = caps

    # Raw input buffer with defined capacity (fixed, not a decision variable)
    raw_input = simpy.Store(env, capacity=1000)

    # Delay buffers with decision-variable capacities and fixed delays
    post_loading_buffer = DelayBuffer(env, cap=cap_post_loading, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=cap_post_conveyor, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=cap_post_washing, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=cap_pre_press1, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=cap_pre_press2, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=cap_post_press12, delay=32)

    # Sinks with large capacity (treated as sinks, so unbounded effectively)
    sink = simpy.Store(env, capacity=float("inf"))
    defects = simpy.Store(env, capacity=float("inf"))

    # Helper stores between parallel presses and merger (fixed capacities)
    pre_press1 = simpy.Store(env, capacity=3)
    pre_press2 = simpy.Store(env, capacity=3)
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    loading_robot = Machine(
        env,
        "Loading robot",
        input_buffer=raw_input,
        output_buffer=post_loading_buffer,
        process_time=12.0,
        availability=90.49,
        mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    conveyor_belt = Machine(
        env,
        "Conveyor belt",
        input_buffer=post_loading_buffer,
        output_buffer=post_conveyor_buffer,
        process_time=6.0,
        availability=100.0,
        mttr=1.0,
        working_power=kwh_per_sec(0.0),
        waiting_power=kwh_per_sec(0.0),
    )

    washing_machine = Machine(
        env,
        "Washing machine",
        input_buffer=post_conveyor_buffer,
        output_buffer=post_washing_buffer,
        process_time=14.0,
        availability=80.89,
        mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    hantering_cell = Machine(
        env,
        "Hantering cell",
        input_buffer=post_washing_buffer,
        output_buffer=pre_press1,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    # Split evenly into two parallel press buffers, honoring capacities
    env.process(splitter(env, pre_press1, pre_press1_buffer, pre_press2_buffer))

    presses_cell1 = Machine(
        env,
        "Presses cell 1",
        input_buffer=pre_press1_buffer,
        output_buffer=press1_out,
        process_time=175.0,
        availability=87.79,
        mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    presses_cell2 = Machine(
        env,
        "Presses cell 2",
        input_buffer=pre_press2_buffer,
        output_buffer=press2_out,
        process_time=176.0,
        availability=87.69,
        mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    # Merge parallel press outputs into common buffer with defined capacity
    merger(env, press1_out, press2_out, post_press12_buffer)

    quality_station = Machine(
        env,
        "Quality station cell",
        input_buffer=post_press12_buffer,
        output_buffer=sink,
        process_time=41.0,
        availability=85.87,
        mttr=66.0,
        working_power=kwh_per_sec(0.84),
        waiting_power=kwh_per_sec(0.58),
        defect_rate=0.089,
        defect_sink=defects,
    )

    machines_list = [
        loading_robot,
        conveyor_belt,
        washing_machine,
        hantering_cell,
        presses_cell1,
        presses_cell2,
        quality_station,
    ]

    # Generator respects raw_input capacity
    env.process(part_generator(env, raw_input))

    # Warm-up
    env.run(until=warmup)

    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    delay_buffers = [
        post_loading_buffer,
        post_conveyor_buffer,
        post_washing_buffer,
        pre_press1_buffer,
        pre_press2_buffer,
        post_press12_buffer,
    ]

    # WIP definition: items in delay buffers + items in process
    # IMPORTANT: Exclude helper stores and raw input
    def sample_wip(env_local):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env_local.timeout(600)

    env.process(sample_wip(env))
    env.run(until=measure_until)

    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0

    result = {
        "overall": {
            "throughput": throughput,
            "wip": avg_wip,
            "produced_parts": total_produced,
        },
        "machine_energy": {},
    }

    for m in machines_list:
        waiting_energy = m.waiting_energy_consumption()
        working_energy = m.working_energy_consumption()
        total_energy = waiting_energy + working_energy
        result["machine_energy"][m.name] = {
            "working_time": m.working_time,
            "waiting_time": m.failed_time_total + m.blocked_time,
            "working_energy": working_energy,
            "waiting_energy": waiting_energy,
            "total_energy": total_energy,
        }

    return result


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x[:6]]

    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        res = run_simulation_with_caps(seed, caps, warmup, measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Objectives: f1 = WIP (min), f2 = -throughput (max throughput)
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables (all integer in [1,10]):
        x[0] = post_loading_buffer cap
        x[1] = post_conveyor_buffer cap
        x[2] = post_washing_buffer cap
        x[3] = pre_press1_buffer cap
        x[4] = pre_press2_buffer cap
        x[5] = post_press12_buffer cap

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
        n_replications=5,
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
        self.n_cores = 50  # enforce exactly 50 cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=RANDOM_SEED,
    verbose=True,
):
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1] * 6),
        xu=np.array([10] * 6),
        n_replications=n_replications,
        base_seed=base_seed,
        n_cores=50,
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
    Only feasible points (within bounds) are exported.
    """

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "cap_post_loading",
        "cap_post_conveyor",
        "cap_post_washing",
        "cap_pre_press1",
        "cap_pre_press2",
        "cap_post_press12",
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

            # Constraint: all capacities must be within [1,10]
            if any((c < 1 or c > 10) for c in caps):
                continue

            wip = float(f[0])
            throughput = float(-f[1])

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "cap_post_loading": caps[0],
                "cap_post_conveyor": caps[1],
                "cap_post_washing": caps[2],
                "cap_pre_press1": caps[3],
                "cap_pre_press2": caps[4],
                "cap_post_press12": caps[5],
                "wip": wip,
                "throughput": throughput,
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        verbose=True,
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")