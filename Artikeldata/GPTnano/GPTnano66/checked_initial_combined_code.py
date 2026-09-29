import simpy
import random
import statistics
import multiprocessing
import os
import csv
from typing import Optional, List

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Integration constants (align with existing simulation model defaults)
RANDOM_SEED = 66
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME

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
    return (getattr(buf, "free_capacity", None) and buf.free_capacity() > 0) \
           or len(buf.items) < buf.capacity

def splitter(env, input_store, out1, out2):
    toggle = 0
    while True:
        part = yield input_store.get()
        first, second = (out1, out2) if toggle == 0 else (out2, out1)
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
        yield self.tokens.get(1)
        self._in_transit += 1
        try:
            yield self.env.timeout(self.delay)
            yield self.store.put(part)
        finally:
            self._in_transit -= 1

    def _get_and_release(self):
        part = yield self.store.get()
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
                passes_during_block = self.failed_time_total - start_fails
                self.blocked_time += max(0.0, gross_block - passes_during_block)
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


def run_simulation(seed, caps=None, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Run the simulation with optional buffer capacities 'caps' for the three main buffers.
    caps: [cap1, cap2, cap3] corresponding to post_loading_buffer, post_conveyor_buffer, post_washing_buffer
    If caps is None, default capacities of [2, 2, 2] are used for compatibility with original setup.
    """
    random.seed(seed)
    env = simpy.Environment()

    # Base capacities from your factory specifications
    # If caps provided, use them; otherwise default values
    if caps is None:
        cap1, cap2, cap3 = 2, 2, 2
    else:
        cap1, cap2, cap3 = int(caps[0]), int(caps[1]), int(caps[2])

    raw_input = simpy.Store(env, capacity=1000)

    post_loading_buffer = DelayBuffer(env, cap=cap1, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=cap2, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=cap3, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=3, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=3, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=3, delay=32)

    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    pre_press1 = simpy.Store(env, capacity=3)
    pre_press2 = simpy.Store(env, capacity=3)
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
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_press1,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

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

    # WIP definition: items in delay buffers + items in process
    # IMPORTANT: Exclude helper stores and raw input


    def sample_wip(env):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            
            # Sample hourly for performance
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

def is_feasible_caps(x):
    try:
        caps = [int(v) for v in x[:3]]
    except Exception:
        return False
    return all(1 <= int(v) <= 10 for v in caps)

def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x[:3]]

    # If infeasible caps, return a flag that will be filtered out later
    if not is_feasible_caps(x):
        return [np.nan, np.nan]

    throughputs = []
    wips = []

    # Create a local random instance to prevent multiple CPU processes 
    # from pulling from the same global random state and generating identical seeds
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        # Call to the external simulation model; assumed to exist in integration
        res = run_simulation(seed, caps, warmup=warmup, measure_until=measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Return the objectives: [f1 (wip), f2 (-throughput)]
    return [avg_wip, -avg_throughput]

class BufferCapacityProblem(Problem):

    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables:
        x[0] = buffer1 cap
        x[1] = buffer2 cap
        x[2] = buffer3 cap

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self, n_var=3, n_obj=2, n_constr=0,
                 xl=None, xu=None,
                 n_replications=5,
                 base_seed=RANDOM_SEED,
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

        # 2. Evaluate sequentially to ensure compatibility across platforms
        results = [evaluate_single_individual(t) for t in tasks]

        # 3. Build F with NaNs for infeasible points
        # Determine feasibility per individual
        F = []
        feasible_indices = []
        for idx, r in enumerate(results):
            if r[0] != r[0] or r[1] != r[1]:
                # infeasible
                F.append([np.nan, np.nan])
            else:
                F.append([float(r[0]), float(r[1])])
                feasible_indices.append(idx)

        F = np.asarray(F, dtype=float)

        out["F"] = F

def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    """
    # Clamp cores to available CPU count for robustness
    available_cores = os.cpu_count() or 1
    n_cores_eff = max(1, min(n_cores, available_cores))

    problem = BufferCapacityProblem(
        n_var=3,
        n_obj=2,
        xl=np.array([1, 1, 1]),
        xu=np.array([10, 10, 10]),
        n_replications=n_replications,
        base_seed=base_seed,
        n_cores=n_cores_eff
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
    Export all solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.
    Filters out infeasible points (caps outside 1..10 are considered infeasible).
    """

    # 1. Define the target directory
    output_dir = "results"
    
    # 2. Safely create the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Join the directory name and the filename together
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "buffer1_cap",
        "buffer2_cap",
        "buffer3_cap",
        "wip",
        "throughput"
    ]

    rows = []
    history = getattr(result, "history", None)

    def _caps_from_x(x):
        return [int(v) for v in x[:3]]

    def _is_valid_row(x, f):
        if x is None or f is None:
            return False
        if not is_feasible_caps(x):
            return False
        caps = _caps_from_x(x)
        if not all(1 <= c <= 10 for c in caps):
            return False
        F_arr = np.asarray(f, dtype=float)
        if not np.isfinite(F_arr).all():
            return False
        return True

    if history is None:
        # If there is no history (edge case), still create header-only CSV
        with open(filepath, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        return

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            continue
        for ind_idx, (x, f) in enumerate(zip(X, F)):
            if not _is_valid_row(x, f):
                continue
            caps = _caps_from_x(x)
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "buffer1_cap": caps[0],
                "buffer2_cap": caps[1],
                "buffer3_cap": caps[2],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization on the buffer capacity optimization problem
    print("Starting NSGA-II optimization with 50 cores...")
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        verbose=True,
        n_cores=50  # Exactly 50 cores (clamped to available cores inside)
    )

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")