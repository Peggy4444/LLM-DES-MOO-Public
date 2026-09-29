import random
import statistics
from copy import deepcopy

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from multiprocessing import Pool

# Assumes run_simulation, RANDOM_SEED, REPLICATIONS, WARMUP_SECONDS, MEASURE_UNTIL
# are imported from the existing simulation module.


class BufferOptimizationProblem(Problem):
    def __init__(self):
        # 6 delay buffers: post_loading, post_conveyor, post_washing,
        # pre_press1, pre_press2, post_press12
        n_var = 6
        xl = np.array([1] * n_var, dtype=int)
        xu = np.array([10] * n_var, dtype=int)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        # x: [cap_post_loading, cap_post_conveyor, cap_post_washing,
        #     cap_pre_press1, cap_pre_press2, cap_post_press12]
        caps = [int(v) for v in x]

        # Constraint: all capacities between 1 and 10 inclusive (already enforced by bounds)
        # If any capacity is outside [1,10], mark as infeasible and skip evaluation.
        if any(c < 1 or c > 10 for c in caps):
            out["F"] = np.array([np.inf, np.inf])
            return

        # Run multiple replications and average KPIs
        runs = REPLICATIONS
        overall_results = []

        for i in range(runs):
            seed = RANDOM_SEED + i
            res = run_simulation_with_caps(seed, caps)
            overall_results.append(res["overall"])

        mean_throughput = statistics.mean(o["throughput"] for o in overall_results)
        mean_wip = statistics.mean(o["wip"] for o in overall_results)

        # Objectives: minimize wip, maximize throughput -> minimize -throughput
        f1 = mean_wip
        f2 = -mean_throughput

        out["F"] = np.array([f1, f2])


def run_simulation_with_caps(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Wrapper around the original run_simulation that overrides the six delay buffer capacities.
    caps: [cap_post_loading, cap_post_conveyor, cap_post_washing,
           cap_pre_press1, cap_pre_press2, cap_post_press12]
    """
    import simpy

    random.seed(seed)
    env = simpy.Environment()

    # Raw input buffer with explicit capacity
    raw_input = simpy.Store(env, capacity=1000)

    # Unpack capacities
    cap_post_loading, cap_post_conveyor, cap_post_washing, \
        cap_pre_press1, cap_pre_press2, cap_post_press12 = caps

    # Delay buffers (raw/normal) with decision-variable capacities & fixed delays
    post_loading_buffer = DelayBuffer(env, cap=cap_post_loading, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=cap_post_conveyor, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=cap_post_washing, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=cap_pre_press1, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=cap_pre_press2, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=cap_post_press12, delay=32)

    # Helper buffers with explicit capacities (sinks)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Buffer between hantering cell and splitter (must have defined capacity)
    pre_split_press_buffer = simpy.Store(env, capacity=6)

    # Buffers at press outputs before merger (defined capacities)
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

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
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_split_press_buffer,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

    env.process(splitter(env, pre_split_press_buffer, pre_press1_buffer, pre_press2_buffer))

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

    env.process(part_generator(env, raw_input))

    env.run(until=warmup)

    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    delay_buffers = [
        post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
        pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    ]

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


def evaluate_individual(x):
    problem = BufferOptimizationProblem()
    out = {}
    problem._evaluate(x, out)
    return out["F"]


def run_moo_optimization():
    problem = BufferOptimizationProblem()

    algorithm = NSGA2(
        pop_size=50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 50)

    with Pool(50) as pool:
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=RANDOM_SEED,
            save_history=False,
            verbose=True,
            pf=False,
            evaluator=None,
            callback=None,
            return_least_infeasible=False,
            n_jobs=50,
            func_eval=pool.map
        )

    # Save only feasible solutions (those not having inf objectives)
    import csv

    X = res.X
    F = res.F

    rows = []
    for x, f in zip(X, F):
        if np.isinf(f).any():
            continue
        row = {
            "cap_post_loading": int(x[0]),
            "cap_post_conveyor": int(x[1]),
            "cap_post_washing": int(x[2]),
            "cap_pre_press1": int(x[3]),
            "cap_pre_press2": int(x[4]),
            "cap_post_press12": int(x[5]),
            "mean_wip": float(f[0]),
            "mean_throughput": float(-f[1])
        }
        rows.append(row)

    with open("moo_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "cap_post_loading",
            "cap_post_conveyor",
            "cap_post_washing",
            "cap_pre_press1",
            "cap_pre_press2",
            "cap_post_press12",
            "mean_wip",
            "mean_throughput"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return res


if __name__ == "__main__":
    run_moo_optimization()