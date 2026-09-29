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

    # Unpack capacities: all are in [1, 10]
    (
        cap_post_loading,
        cap_post_conveyor,
        cap_post_washing,
        cap_pre_press1,
        cap_pre_press2,
        cap_post_press12,
    ) = caps

    # Raw input buffer (fixed)
    raw_input = simpy.Store(env, capacity=1000)

    # Delay buffers with decision-variable capacities
    post_loading_buffer = DelayBuffer(env, cap=cap_post_loading, delay=10)    # PostLoadingBuffer
    post_conveyor_buffer = DelayBuffer(env, cap=cap_post_conveyor, delay=10)  # PostConveyorBuffer
    post_washing_buffer = DelayBuffer(env, cap=cap_post_washing, delay=10)    # PostWashingBuffer
    pre_press1_buffer = DelayBuffer(env, cap=cap_pre_press1, delay=32)        # PrePress1Buffer
    pre_press2_buffer = DelayBuffer(env, cap=cap_pre_press2, delay=32)        # PrePress2Buffer
    post_press12_buffer = DelayBuffer(env, cap=cap_post_press12, delay=32)    # PostPress1&Press2Buffer

    # Helper buffers (fixed capacities)
    hantering_to_split = simpy.Store(env, capacity=3)
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

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
        output_buffer=hantering_to_split,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    env.process(splitter(env, hantering_to_split, pre_press1_buffer, pre_press2_buffer))

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

    env.process(part_generator(env, raw_input))

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

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Decision variables (all integer in [1,10]):
        x[0] = cap_post_loading
        x[1] = cap_post_conveyor
        x[2] = cap_post_washing
        x[3] = cap_pre_press1
        x[4] = cap_pre_press2
        x[5] = cap_post_press12

    Objectives:
        f1 = average WIP (min)
        f2 = -average throughput (min, since throughput is maximized)
    """

    def __init__(
        self,
        n_var=6,
        n_obj=2,
        n_constr=0,
        xl=None,
        xu=None,
        n_replications=REPLICATIONS,
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
        self.n_cores = 50  # force exactly 50 cores

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
    n_replications=REPLICATIONS,
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
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPLICATIONS,
        verbose=True,
    )
    export_history_to_csv(result, filename="moo_simulation_results.csv")