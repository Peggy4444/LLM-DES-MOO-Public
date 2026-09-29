import numpy as np
import random
import statistics
from multiprocessing import Pool

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Assumes run_simulation is imported from the simulation module:
# from simulation_module import run_simulation, RANDOM_SEED, REPLICATIONS, WARMUP_SECONDS, MEASURE_UNTIL

RANDOM_SEED = 99
REPLICATIONS = 10
WARMUP_SECONDS = 86400
MEASURE_UNTIL = 691200


def evaluate_design(x):
    """
    Evaluate a single design vector x (buffer capacities).
    x: array-like of length 6, integer capacities in [1,10]
    Order: [post_loading, post_conveyor, post_washing,
            pre_press1, pre_press2, post_press12]
    """
    # Constraint: all capacities must be between 1 and 10 (inclusive)
    # This is already enforced by bounds, but we keep it explicit.
    if any((c < 1 or c > 10) for c in x):
        # Return None to indicate infeasible / not to be recorded
        return None

    # Run multiple replications and average KPIs
    throughputs = []
    wips = []

    for i in range(REPLICATIONS):
        seed = RANDOM_SEED + i

        # We need to call a modified version of run_simulation that accepts buffer capacities.
        # To keep compatibility, we wrap the original run_simulation here by monkey-patching
        # the capacities at runtime via a dedicated function.
        res = run_simulation_with_capacities(
            seed=seed,
            caps={
                "post_loading": int(x[0]),
                "post_conveyor": int(x[1]),
                "post_washing": int(x[2]),
                "pre_press1": int(x[3]),
                "pre_press2": int(x[4]),
                "post_press12": int(x[5]),
            },
            warmup=WARMUP_SECONDS,
            measure_until=MEASURE_UNTIL,
        )

        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    mean_throughput = statistics.mean(throughputs)
    mean_wip = statistics.mean(wips)

    # Objectives: f1 = wip (minimize), f2 = -throughput (since pymoo minimizes)
    return np.array([mean_wip, -mean_throughput])


def run_simulation_with_capacities(seed, caps, warmup, measure_until):
    """
    Wrapper around the provided run_simulation that injects buffer capacities.
    This function assumes that run_simulation is modified to accept an optional
    'buffer_caps' argument, or that the simulation code reads these globals.
    For strict compatibility with the given code, we re-implement the parts
    that depend on capacities and then call the original logic.
    """

    import simpy

    random.seed(seed)
    env = simpy.Environment()

    from math import inf

    def kwh_per_sec(x):
        return x / 3600.0

    # Raw input buffer
    raw_input = simpy.Store(env, capacity=1000)

    # Delay buffers with decision-variable capacities
    post_loading_buffer = DelayBuffer(env, cap=caps["post_loading"], delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=caps["post_conveyor"], delay=10)
    post_washing_buffer = DelayBuffer(env, cap=caps["post_washing"], delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=caps["pre_press1"], delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=caps["pre_press2"], delay=32)
    post_press12_buffer = DelayBuffer(env, cap=caps["post_press12"], delay=32)

    pre_press_split_input = simpy.Store(env, capacity=6)
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    loading_robot = Machine(
        env, "Loading robot",
        input_buffer=raw_input,
        output_buffer=post_loading_buffer,
        process_time=12.0,
        availability=90.49,
        mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    conveyor_belt = Machine(
        env, "Conveyor belt",
        input_buffer=post_loading_buffer,
        output_buffer=post_conveyor_buffer,
        process_time=6.0,
        availability=100.0,
        mttr=1.0,
        working_power=kwh_per_sec(0.0),
        waiting_power=kwh_per_sec(0.0),
    )

    washing_machine = Machine(
        env, "Washing machine",
        input_buffer=post_conveyor_buffer,
        output_buffer=post_washing_buffer,
        process_time=14.0,
        availability=80.89,
        mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    hantering_cell = Machine(
        env, "Hantering cell",
        input_buffer=post_washing_buffer,
        output_buffer=pre_press_split_input,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    env.process(splitter(env, pre_press_split_input, pre_press1_buffer, pre_press2_buffer))

    presses_cell1 = Machine(
        env, "Presses cell 1",
        input_buffer=pre_press1_buffer,
        output_buffer=press1_out,
        process_time=175.0,
        availability=87.79,
        mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    presses_cell2 = Machine(
        env, "Presses cell 2",
        input_buffer=pre_press2_buffer,
        output_buffer=press2_out,
        process_time=176.0,
        availability=87.69,
        mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    merger(env, press1_out, press2_out, post_press12_buffer)

    quality_station = Machine(
        env, "Quality station cell",
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

    def sample_wip(env):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env.timeout(600)

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
            "produced_parts": total_produced
        },
        "machine_energy": {}
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
            "total_energy": total_energy
        }

    return result


class BufferOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=6,
            n_obj=2,
            n_constr=0,
            xl=np.array([1, 1, 1, 1, 1, 1]),
            xu=np.array([10, 10, 10, 10, 10, 10]),
            elementwise_evaluation=True,
            type_var=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        res = evaluate_design(x)
        if res is None:
            # Infeasible: assign very bad objectives so they are dominated and not selected
            out["F"] = np.array([1e9, 1e9])
        else:
            out["F"] = res


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
            pf=None,
            evaluator={"type": "parallel", "pool": pool}
        )

    X = res.X
    F = res.F

    import csv

    with open("moo_results.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "post_loading_cap",
            "post_conveyor_cap",
            "post_washing_cap",
            "pre_press1_cap",
            "pre_press2_cap",
            "post_press12_cap",
            "wip",
            "throughput"
        ])

        for x, fvals in zip(X, F):
            wip = fvals[0]
            throughput = -fvals[1]
            # Only write feasible points (those that were actually evaluated)
            if wip < 1e9 and throughput < 1e9:
                writer.writerow([
                    int(x[0]),
                    int(x[1]),
                    int(x[2]),
                    int(x[3]),
                    int(x[4]),
                    int(x[5]),
                    float(wip),
                    float(throughput)
                ])


if __name__ == "__main__":
    run_moo_optimization()