import simpy
import random
import statistics
import multiprocessing as mp
import csv
import os
from functools import partial

RANDOM_SEED = 66
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME

POP_SIZE = 50
N_GEN = 100
N_CORES = 50

# Decision variables: capacities of all buffers (1-10, integers)
# Order: [raw_input, post_loading, post_conveyor, post_washing,
#         pre_press1_delay, pre_press2_delay, post_press12_delay,
#         pre_press1_store, pre_press2_store, press1_out_store, press2_out_store]
VAR_BOUNDS = [(1, 10)] * 11


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
    """Alternates parts between out1 and out2, but respects buffer capacities."""
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

                # 1. Starvation Phase
                part = None
                start_starve = self.env.now
                start_fails = self.failed_time_total
                req = None

                while part is None:
                    if not self.is_up:
                        try:
                            yield self.repair_event
                        except simpy.Interrupt:
                            pass
                        continue

                    if req is None:
                        req = self.input_buffer.get()

                    try:
                        part = yield req
                        if self.env.now >= self.last_reset:
                            gross_wait = self.env.now - max(start_starve, self.last_reset)
                            fails_during_wait = self.failed_time_total - start_fails
                            self.wait_input_time += max(0.0, gross_wait - fails_during_wait)
                    except simpy.Interrupt:
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

                # 3. Processing Phase
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
        # 1 second between arrivals
        yield env.timeout(1)


def kwh_per_sec(x):
    return x / 3600.0


def run_simulation(seed, capacities=None, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Default capacities
    default_caps = {
        "raw_input": 1000,
        "post_loading": 2,
        "post_conveyor": 2,
        "post_washing": 2,
        "pre_press1_delay": 3,
        "pre_press2_delay": 3,
        "post_press12_delay": 3,
        "pre_press1_store": 3,
        "pre_press2_store": 3,
        "press1_out_store": 3,
        "press2_out_store": 3,
    }

    if capacities is None:
        capacities = default_caps
    else:
        # Merge user capacities with defaults to ensure all keys exist
        merged = default_caps.copy()
        merged.update(capacities)
        capacities = merged

    # Raw input buffer capacity as specified / optimized
    raw_input = simpy.Store(env, capacity=capacities["raw_input"])

    # Delay buffers with explicit capacities and process times (optimized)
    post_loading_buffer = DelayBuffer(env, cap=capacities["post_loading"], delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=capacities["post_conveyor"], delay=10)
    post_washing_buffer = DelayBuffer(env, cap=capacities["post_washing"], delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=capacities["pre_press1_delay"], delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=capacities["pre_press2_delay"], delay=32)
    post_press12_buffer = DelayBuffer(env, cap=capacities["post_press12_delay"], delay=32)

    # Final sinks (defining capacities explicitly)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Helper (normal) buffers in the press area, capacities defined explicitly / optimized
    pre_press1 = simpy.Store(env, capacity=capacities["pre_press1_store"])
    pre_press2 = simpy.Store(env, capacity=capacities["pre_press2_store"])
    press1_out = simpy.Store(env, capacity=capacities["press1_out_store"])
    press2_out = simpy.Store(env, capacity=capacities["press2_out_store"])

    # Production line machines / cells
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
        output_buffer=pre_press1,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    # Split from handling cell to two press delay buffers (parallel presses)
    env.process(splitter(env, pre_press1, pre_press1_buffer, pre_press2_buffer))

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

    # Merge press outputs into common post-press delay buffer
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

    # Part generator feeding the raw input buffer, respecting its capacity
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

    # WIP definition: items in delay buffers + items in machines
    # Exclude helper stores and raw input

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


def run_simulation_with_capacities(ind, seed):
    """
    Wrapper around run_simulation that applies the individual's buffer capacities.
    """
    capacities = {
        "raw_input": ind[0],
        "post_loading": ind[1],
        "post_conveyor": ind[2],
        "post_washing": ind[3],
        "pre_press1_delay": ind[4],
        "pre_press2_delay": ind[5],
        "post_press12_delay": ind[6],
        "pre_press1_store": ind[7],
        "pre_press2_store": ind[8],
        "press1_out_store": ind[9],
        "press2_out_store": ind[10],
    }
    return run_simulation(seed, capacities=capacities)


def evaluate_individual(ind, seed_offset=0, runs=3):
    """
    Evaluate an individual by running the simulation multiple times and averaging.
    Objectives:
      - f1: WIP (to minimize)
      - f2: -Throughput (to minimize, since we want to maximize throughput)
    Constraint:
      - If any simulation run produces zero parts, treat as infeasible.
    """
    wip_vals = []
    thr_vals = []

    for r in range(runs):
        seed = RANDOM_SEED + seed_offset + r
        res = run_simulation_with_capacities(ind, seed)
        produced = res["overall"]["produced_parts"]
        if produced <= 0:
            # Infeasible
            return None
        wip_vals.append(res["overall"]["wip"])
        thr_vals.append(res["overall"]["throughput"])

    avg_wip = sum(wip_vals) / len(wip_vals)
    avg_thr = sum(thr_vals) / len(thr_vals)

    return (avg_wip, -avg_thr)


def dominates(a, b):
    """
    Return True if objective vector a dominates b (strict Pareto dominance).
    Both a and b are tuples (f1, f2) to be minimized.
    """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(pop_objs):
    """
    Perform fast non-dominated sorting.
    pop_objs: list of objective tuples.
    Returns: list of fronts, each front is a list of indices.
    """
    S = [[] for _ in range(len(pop_objs))]
    n = [0] * len(pop_objs)
    rank = [0] * len(pop_objs)
    fronts = [[]]

    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if p == q:
                continue
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, rank


def crowding_distance(front, pop_objs):
    """
    Compute crowding distance for a front.
    front: list of indices
    pop_objs: list of objective tuples
    Returns: dict index -> distance
    """
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance

    num_obj = len(pop_objs[0])
    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda i: pop_objs[i][m])
        f_min = pop_objs[front_sorted[0]][m]
        f_max = pop_objs[front_sorted[-1]][m]
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        if f_max == f_min:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_f = pop_objs[front_sorted[k - 1]][m]
            next_f = pop_objs[front_sorted[k + 1]][m]
            distance[front_sorted[k]] += (next_f - prev_f) / (f_max - f_min)
    return distance


def tournament_selection(pop, pop_objs, k=2):
    """
    Binary tournament selection based on rank and crowding distance.
    pop: list of individuals
    pop_objs: list of objective tuples
    Returns: selected individual (deep copy not necessary for ints).
    """
    i, j = random.sample(range(len(pop)), 2)
    a = pop[i]
    b = pop[j]
    if a["rank"] < b["rank"]:
        return a["ind"]
    elif a["rank"] > b["rank"]:
        return b["ind"]
    else:
        if a["crowding"] > b["crowding"]:
            return a["ind"]
        else:
            return b["ind"]


def crossover(parent1, parent2, pc=0.9):
    """
    Single-point crossover for integer vectors.
    """
    if random.random() > pc:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    """
    Uniform mutation for integer variables within bounds.
    """
    for i, (low, up) in enumerate(VAR_BOUNDS):
        if random.random() < pm:
            ind[i] = random.randint(low, up)
    return ind


def init_population():
    pop = []
    for _ in range(POP_SIZE):
        ind = [random.randint(low, up) for (low, up) in VAR_BOUNDS]
        pop.append({"ind": ind, "objs": None, "rank": None, "crowding": None})
    return pop


def evaluate_population(pop, gen_seed_offset):
    """
    Evaluate all individuals in the population in parallel.
    Only feasible individuals (non-None objectives) are kept.
    """
    with mp.Pool(processes=N_CORES) as pool:
        func = partial(evaluate_individual, seed_offset=gen_seed_offset)
        inds = [p["ind"] for p in pop]
        results = pool.map(func, inds)

    new_pop = []
    for p, objs in zip(pop, results):
        if objs is not None:
            p["objs"] = objs
            new_pop.append(p)
    return new_pop


def assign_rank_and_crowding(pop):
    """
    Assign rank and crowding distance to population.
    """
    pop_objs = [p["objs"] for p in pop]
    fronts, rank = fast_non_dominated_sort(pop_objs)
    for i, p in enumerate(pop):
        p["rank"] = rank[i]
        p["crowding"] = 0.0

    for front in fronts:
        dist = crowding_distance(front, pop_objs)
        for i in front:
            pop[i]["crowding"] = dist[i]


def make_offspring(pop):
    """
    Create offspring population using selection, crossover, and mutation.
    """
    offspring = []
    while len(offspring) < POP_SIZE:
        parent1 = tournament_selection(pop, [p["objs"] for p in pop])
        parent2 = tournament_selection(pop, [p["objs"] for p in pop])
        c1, c2 = crossover(parent1, parent2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        offspring.append({"ind": c1, "objs": None, "rank": None, "crowding": None})
        if len(offspring) < POP_SIZE:
            offspring.append({"ind": c2, "objs": None, "rank": None, "crowding": None})
    return offspring


def environmental_selection(pop):
    """
    NSGA-II environmental selection to form next generation.
    """
    pop_objs = [p["objs"] for p in pop]
    fronts, rank = fast_non_dominated_sort(pop_objs)
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) <= POP_SIZE:
            for i in front:
                new_pop.append(pop[i])
        else:
            dist = crowding_distance(front, pop_objs)
            sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
            for i in sorted_front:
                if len(new_pop) < POP_SIZE:
                    new_pop.append(pop[i])
                else:
                    break
            break
    return new_pop


def run_moo(output_csv_path):
    """
    Run NSGA-II for the given number of generations.
    Only feasible individuals are evaluated and stored.
    """
    random.seed(RANDOM_SEED)

    # Initialize population
    pop = init_population()

    # Evaluate initial population
    pop = evaluate_population(pop, gen_seed_offset=0)
    if not pop:
        return

    assign_rank_and_crowding(pop)

    # Prepare CSV
    header = [
        "generation",
        "individual_id",
        "raw_input_capacity",
        "post_loading_buffer_capacity",
        "post_conveyor_buffer_capacity",
        "post_washing_buffer_capacity",
        "pre_press1_delay_buffer_capacity",
        "pre_press2_delay_buffer_capacity",
        "post_press12_delay_buffer_capacity",
        "pre_press1_store_capacity",
        "pre_press2_store_capacity",
        "press1_out_store_capacity",
        "press2_out_store_capacity",
        "wip",
        "throughput"
    ]
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Log initial population
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        for idx, p in enumerate(pop):
            ind = p["ind"]
            wip, neg_thr = p["objs"]
            row = [
                0,
                idx
            ] + ind + [
                wip,
                -neg_thr
            ]
            writer.writerow(row)

    # Generational loop
    for gen in range(1, N_GEN + 1):
        offspring = make_offspring(pop)
        offspring = evaluate_population(offspring, gen_seed_offset=gen * 10000)
        if not offspring:
            continue

        # Combine and select
        combined = pop + offspring
        assign_rank_and_crowding(combined)
        pop = environmental_selection(combined)

        # Log current population
        with open(output_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for idx, p in enumerate(pop):
                ind = p["ind"]
                wip, neg_thr = p["objs"]
                row = [
                    gen,
                    idx
                ] + ind + [
                    wip,
                    -neg_thr
                ]
                writer.writerow(row)


if __name__ == "__main__":
    results_dir = "results"
    output_csv = os.path.join(results_dir, "moo_simulation_results.csv")
    run_moo(output_csv)