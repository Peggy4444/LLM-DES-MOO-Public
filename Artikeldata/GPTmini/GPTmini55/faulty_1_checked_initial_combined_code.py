import simpy
import random
import statistics
import multiprocessing
import os
import numpy as np
import csv
from typing import List

# pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Simulation constants
RANDOM_SEED = 55
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
    """
    Round-robin splitter that tries to split items evenly between out1 and out2.
    If the preferred output is full, it will try the other. The toggle is flipped
    after each successful put to keep the split as even as possible.
    """
    toggle = 0
    while True:
        part = yield input_store.get()
        first, second = (out1, out2) if toggle == 0 else (out2, out1)

        # Prefer first, otherwise second. Flip toggle after successful put so we alternate.
        if _has_free_capacity(first):
            yield first.put(part)
        elif _has_free_capacity(second):
            yield second.put(part)
        else:
            # If neither has immediate free capacity, block on the preferred one.
            # This ensures we don't drop the part; toggle will still flip afterwards to
            # continue round-robin behavior.
            yield first.put(part)

        toggle ^= 1

def forwarder(env, src, dst):
    while True:
        part = yield src.get()
        yield dst.put(part)

def merger(env, a, b, out):
    # create two forwarders that push from a->out and b->out
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
                    p.interrupt()
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

def run_simulation(seed, caps: List[int] = None, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Run a single simulation.

    caps: Optional list/iterable of 6 integers representing capacities for:
      [post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
       pre_press1_buffer, pre_press2_buffer, post_press12_buffer]

    Returns the result dictionary with keys 'overall' and 'machine_energy'.
    """
    random.seed(seed)
    env = simpy.Environment()

    # Default capacity values if caps not provided
    # Defaults from original simulation
    default_caps = [2, 2, 2, 3, 3, 3]
    if caps is None:
        caps = default_caps
    else:
        # Ensure exactly 6 integers
        if len(caps) != 6:
            raise ValueError("caps must be an iterable of 6 integers.")
        caps = [int(v) for v in caps]

    # Base capacities from your factory specifications
    raw_input = simpy.Store(env, capacity=1000)

    # Map caps to corresponding buffers
    post_loading_buffer = DelayBuffer(env, cap=caps[0], delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=caps[1], delay=10)
    post_washing_buffer = DelayBuffer(env, cap=caps[2], delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=caps[3], delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=caps[4], delay=32)
    post_press12_buffer = DelayBuffer(env, cap=caps[5], delay=32)

    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Helper stores (explicit capacities defined)
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

    # Split the stream from hantering_cell -> pre_press1 & pre_press2 (even split)
    # Note: The original code used splitter(env, pre_press1, pre_press1_buffer, pre_press2_buffer)
    # which assumed hantering_cell output went to pre_press1 store. We need to connect hantering_cell output
    # to a store that then gets split into the two pre_press buffers. We use pre_press1 as the source store.
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

    # Merge press outputs into post_press12_buffer
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

    # Warm-up
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
            
            # Sample every 10 minutes for performance (adjust as needed)
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

# --- MOO code integration starts here ---

# Provide fallback constants if not present (they are present above)
try:
    RANDOM_SEED
    WARMUP_SECONDS
    MEASURE_UNTIL
except NameError:
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

            # Call simulation entrypoint with capacities
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
    output_dir = "results"
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
    return filepath

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
    # Ensure the multiprocessing start method is set (try fork, fallback to spawn)
    try:
        multiprocessing.set_start_method("fork")
    except Exception:
        try:
            multiprocessing.set_start_method("spawn")
        except Exception:
            pass

    # Run NSGA2 with exactly 50 cores as requested
    POP_SIZE = 50
    N_GEN = 50
    N_REPLICATIONS = 3
    N_CORES = 50

    print(f"Starting NSGA2 optimization with pop_size={POP_SIZE}, n_gen={N_GEN}, n_replications={N_REPLICATIONS}, n_cores={N_CORES}")
    result = run_nsga2_optimization(pop_size=POP_SIZE, n_gen=N_GEN, n_replications=N_REPLICATIONS, verbose=True, n_cores=N_CORES)

    csv_path = export_history_to_csv(result, filename="moo_simulation_results.csv")
    print(f"Exported feasible solutions history to: {csv_path}")

    # Print a self-explanatory table of found solutions (feasible ones)
    print("\nFeasible solutions (from export):")
    try:
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                print("No feasible solutions found (CSV is empty).")
            else:
                # Print header
                headers = reader.fieldnames
                # Format a simple table
                col_widths = {h: max(len(h), 12) for h in headers}
                for row in rows:
                    for h in headers:
                        col_widths[h] = max(col_widths[h], len(str(row[h])))
                # Print header row
                header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
                print(header_line)
                print("-" * len(header_line))
                # Print up to all rows
                for row in rows:
                    line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
                    print(line)
    except Exception as e:
        print(f"Failed to read/export CSV results for display: {e}")