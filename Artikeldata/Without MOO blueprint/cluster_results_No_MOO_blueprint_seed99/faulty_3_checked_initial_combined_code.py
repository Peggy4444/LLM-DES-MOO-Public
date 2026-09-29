import simpy
import random
import statistics
import numpy as np
import os
import csv
from multiprocessing import Pool

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator

RANDOM_SEED = 99
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME
REPLICATIONS = 10


def production_wait_time(now: float) -> float:
    """
    Compute how long (in seconds) the machine must wait until production is allowed.
    Stop windows: Friday 17:00 -> Saturday 07:00 & Saturday 17:00 -> Sunday 07:00
    """
    SEC_PER_DAY = 86400
    day = int((now // SEC_PER_DAY) % 7)  # 0=Mon ... 6=Sun
    time_of_day = now % SEC_PER_DAY
    t_07 = 7 * 3600
    t_17 = 17 * 3600

    if day == 4:  # Friday
        if time_of_day >= t_17:
            return (SEC_PER_DAY - time_of_day) + t_07
    elif day == 5:  # Saturday
        if time_of_day < t_07:
            return t_07 - time_of_day
        if time_of_day >= t_17:
            return (SEC_PER_DAY - time_of_day) + t_07
    elif day == 6:  # Sunday
        if time_of_day < t_07:
            return t_07 - time_of_day

    return 0.0


def _has_free_capacity(buf):
    # Works for both simpy.Store and DelayBuffer
    if hasattr(buf, "free_capacity"):
        return buf.free_capacity() > 0
    return len(buf.items) < buf.capacity


def splitter(env, input_store, out1, out2):
    """
    Split items from input_store evenly across out1 and out2, honoring capacities.
    """
    toggle = 0
    while True:
        part = yield input_store.get()
        first, second = (out1, out2) if toggle == 0 else (out2, out1)

        # Try preferred first; if full, use second
        if _has_free_capacity(first):
            yield first.put(part)
            toggle ^= 1
        else:
            yield second.put(part)


def forwarder(env, src, dst):
    while True:
        part = yield src.get()
        yield dst.put(part)


def merger(env, a, b, out):
    """
    Merge two stores into one output store.
    """
    env.process(forwarder(env, a, out))
    env.process(forwarder(env, b, out))


def reset_machine_stats(m):
    m.working_time = 0
    m.failed_time_total = 0
    m.wait_input_time = 0
    m.blocked_time = 0
    m.processed_count = 0
    m.window_wait_time = 0
    m.last_reset = m.env.now


class DelayBuffer:
    """Single store with a global capacity cap that includes in-transit + ready."""
    def __init__(self, env, cap, delay):
        self.env = env
        self.delay = delay
        self.cap = cap
        self.store = simpy.Store(env, capacity=cap)
        self.tokens = simpy.Container(env, init=cap, capacity=cap)
        self._in_transit = 0

    def put(self, part):
        return self.env.process(self._delayed_put(part))

    def get(self):
        return self.env.process(self._get_and_release())

    @property
    def items(self):
        return self.store.items

    @property
    def capacity(self):
        return self.store.capacity

    def in_transit_count(self):
        return self._in_transit

    def free_capacity(self):
        return int(self.tokens.level)

    def _delayed_put(self, part):
        # Reserve capacity (includes transit + stored)
        yield self.tokens.get(1)
        self._in_transit += 1
        try:
            # Transport/processing delay
            yield self.env.timeout(self.delay)
            yield self.store.put(part)
        finally:
            self._in_transit -= 1

    def _get_and_release(self):
        part = yield self.store.get()
        # Release capacity token once part leaves buffer
        yield self.tokens.put(1)
        return part


class Machine:
    """Event-driven machine with 100% accurate queue physics and zero busy-waiting."""
    def __init__(self, env, name, input_buffer, output_buffer, process_time,
                 availability, mttr, working_power, waiting_power,
                 defect_rate=None, defect_sink=None, capacity=1):

        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.process_time = process_time
        self.availability = availability
        self.mttr = mttr
        self.defect_rate = defect_rate
        self.defect_sink = defect_sink
        self.working_power = working_power
        self.waiting_power = waiting_power
        self.capacity = capacity

        self.resource = simpy.Resource(env, capacity=capacity)
        self.worker_processes = []

        self.working_time = 0
        self.failed_time_total = 0
        self.wait_input_time = 0
        self.blocked_time = 0
        self.active_count = 0
        self.processed_count = 0
        self.window_wait_time = 0
        self.last_reset = 0.0
        self.is_up = True

        # Event trigger for waking up sleeping workers without polling
        self.repair_event = env.event()

        if availability < 100:
            avail_frac = availability / 100.0
            self.mtbf = mttr * (avail_frac / (1 - avail_frac))
            env.process(self._breakdown_cycle())
        else:
            self.mtbf = float('inf')

        for _ in range(capacity):
            p = env.process(self.run())
            self.worker_processes.append(p)

    def _breakdown_cycle(self):
        while True:
            t_up = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(t_up)

            self.is_up = False
            for p in self.worker_processes:
                try:
                    p.interrupt("BREAKDOWN")
                except RuntimeError:
                    pass

            t_repair = random.expovariate(1.0 / self.mttr)
            start_repair = self.env.now
            yield self.env.timeout(t_repair)

            if self.env.now >= self.last_reset:
                repair_start_effective = max(start_repair, self.last_reset)
                self.failed_time_total += max(0.0, self.env.now - repair_start_effective)

            self.is_up = True

            # Instantly wake up all workers waiting for the machine to be fixed
            self.repair_event.succeed()
            self.repair_event = self.env.event()

    def run(self):
        while True:
            with self.resource.request() as res_req:
                yield res_req

                # 1. Starvation Phase (Event-Driven, Zero Polling)
                part = None
                start_starve = self.env.now
                start_fails = self.failed_time_total
                req = None  # Lazily initialize the request

                while part is None:
                    if not self.is_up:
                        try:
                            # Wait until the already-scheduled repair is completed.
                            yield self.repair_event
                        except simpy.Interrupt:
                            pass
                        continue

                    # Machine is confirmed UP. Safe to request a part.
                    if req is None:
                        req = self.input_buffer.get()

                    try:
                        part = yield req
                        if self.env.now >= self.last_reset:
                            gross_wait = self.env.now - max(start_starve, self.last_reset)
                            fails_during_wait = self.failed_time_total - start_fails
                            self.wait_input_time += max(0.0, gross_wait - fails_during_wait)
                    except simpy.Interrupt:
                        # Interrupted by breakdown while actively waiting for a part!
                        # DO NOT cancel req. Let the loop cycle back, sleep on repair_event,
                        # and resume waiting on the existing req once repaired so we don't drop parts.
                        pass

                self.processed_count += 1
                self.active_count += 1

                # 2. Shift schedule wait
                w = production_wait_time(self.env.now)
                if w:
                    if self.env.now >= self.last_reset:
                        self.window_wait_time += w
                    req_w = self.env.timeout(w)
                    while True:
                        try:
                            yield req_w
                            break
                        except simpy.Interrupt:
                            pass

                # 3. Processing Phase (Event-Driven)
                pt = self.process_time() if callable(self.process_time) else self.process_time
                remaining = pt

                while remaining > 0:
                    if not self.is_up:
                        try:
                            yield self.repair_event
                        except simpy.Interrupt:
                            pass
                        continue

                    start_work = self.env.now
                    try:
                        yield self.env.timeout(remaining)
                        self.working_time += (self.env.now - max(start_work, self.last_reset))
                        remaining = 0
                    except simpy.Interrupt:
                        self.working_time += (self.env.now - max(start_work, self.last_reset))
                        remaining -= (self.env.now - start_work)

                # 4. Routing Phase
                start_block = self.env.now
                start_fails = self.failed_time_total

                if self.defect_rate is not None and self.defect_sink is not None and random.random() < self.defect_rate:
                    part["defect"] = 1
                    req_out = self.defect_sink.put(part)
                else:
                    part["defect"] = 0
                    req_out = self.output_buffer.put(part)

                while True:
                    try:
                        yield req_out
                        break
                    except simpy.Interrupt:
                        pass

                gross_block = self.env.now - max(start_block, self.last_reset)
                fails_during_block = self.failed_time_total - start_fails
                self.blocked_time += max(0.0, gross_block - fails_during_block)
                self.active_count -= 1

    def waiting_energy_consumption(self):
        return self.waiting_power * (self.wait_input_time +
                                     self.failed_time_total +
                                     self.blocked_time +
                                     self.window_wait_time)

    def working_energy_consumption(self):
        return self.working_power * self.working_time


def part_generator(env, output_buffer):
    part_id = 0
    while True:
        part = {"id": part_id}
        yield output_buffer.put(part)
        part_id += 1
        yield env.timeout(1)


def kwh_per_sec(x):
    return x / 3600.0


def run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Raw input buffer with defined capacity (matches description: raw & normal buffers have capacity)
    raw_input = simpy.Store(env, capacity=1000)

    # Delay buffers (normal buffers) with specified capacities and process times
    post_loading_buffer = DelayBuffer(env, cap=2, delay=10)    # PostLoadingBuffer
    post_conveyor_buffer = DelayBuffer(env, cap=2, delay=10)   # PostConveyorBuffer
    post_washing_buffer = DelayBuffer(env, cap=2, delay=10)    # PostWashingBuffer
    pre_press1_buffer = DelayBuffer(env, cap=3, delay=32)      # PrePress1Buffer
    pre_press2_buffer = DelayBuffer(env, cap=3, delay=32)      # PrePress2Buffer
    post_press12_buffer = DelayBuffer(env, cap=3, delay=32)    # PostPress1&Press2Buffer

    # Helper buffers for press routing / merging with explicit capacities
    pre_press_split_input = simpy.Store(env, capacity=6)  # Between Hantering cell and splitter
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    # Sinks with defined capacities
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Machines according to station table and direct-follow relationships
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
        output_buffer=pre_press_split_input,  # feeds into splitter input
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    # Split stream from Hantering cell evenly into PrePress1Buffer and PrePress2Buffer
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

    # Merge parallel press outputs into PostPress1&Press2Buffer
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

    # Part generation into raw input (with its capacity)
    env.process(part_generator(env, raw_input))

    # Warmup
    env.run(until=warmup)

    # Reset statistics after warmup
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    delay_buffers = [
        post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
        pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    ]

    # WIP definition: items in delay buffers + items in process
    # IMPORTANT: Exclude helper stores and raw input

    def sample_wip(env):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            # sample every 10 minutes
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


def run_simulation_with_capacities(seed, caps, warmup, measure_until):
    """
    Simulation wrapper that uses decision-variable buffer capacities.
    """
    random.seed(seed)
    env = simpy.Environment()

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


def evaluate_design(x):
    """
    Evaluate a single design vector x (buffer capacities).
    x: array-like of length 6, integer capacities in [1,10]
    Order: [post_loading, post_conveyor, post_washing,
            pre_press1, pre_press2, post_press12]
    """
    # Constraint: all capacities must be between 1 and 10 (inclusive)
    if any((c < 1 or c > 10) for c in x):
        return None

    throughputs = []
    wips = []

    for i in range(REPLICATIONS):
        seed = RANDOM_SEED + i

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

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "moo_simulation_results.csv")

    # Run optimization using exactly 50 cores
    with Pool(50) as pool:
        evaluator = Evaluator(evaluate=None, n_processes=None, func_eval=None, func_batch=None, pool=pool)
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=RANDOM_SEED,
            save_history=True,
            verbose=True,
            pf=None,
            evaluator=evaluator
        )

    # Collect all feasible solutions from all generations
    rows = []
    # Initial population
    if hasattr(res, "history") and res.history is not None:
        for algo in res.history:
            pop = algo.pop
            X_gen = pop.get("X")
            F_gen = pop.get("F")
            for x, fvals in zip(X_gen, F_gen):
                wip = float(fvals[0])
                throughput = -float(fvals[1])
                # Filter out infeasible / penalized solutions
                if wip >= 1e9 or throughput >= 1e9:
                    continue
                rows.append([
                    int(x[0]),
                    int(x[1]),
                    int(x[2]),
                    int(x[3]),
                    int(x[4]),
                    int(x[5]),
                    wip,
                    throughput
                ])
    else:
        # Fallback: only final population
        X = res.X
        F = res.F
        for x, fvals in zip(X, F):
            wip = float(fvals[0])
            throughput = -float(fvals[1])
            if wip >= 1e9 or throughput >= 1e9:
                continue
            rows.append([
                int(x[0]),
                int(x[1]),
                int(x[2]),
                int(x[3]),
                int(x[4]),
                int(x[5]),
                wip,
                throughput
            ])

    # Remove duplicates (same decision vector and KPIs)
    unique = []
    seen = set()
    for r in rows:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Write CSV with self-explanatory column names
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "post_loading_buffer_capacity",
            "post_conveyor_buffer_capacity",
            "post_washing_buffer_capacity",
            "pre_press1_buffer_capacity",
            "pre_press2_buffer_capacity",
            "post_press12_buffer_capacity",
            "wip",
            "throughput"
        ])
        writer.writerows(unique)


if __name__ == "__main__":
    run_moo_optimization()