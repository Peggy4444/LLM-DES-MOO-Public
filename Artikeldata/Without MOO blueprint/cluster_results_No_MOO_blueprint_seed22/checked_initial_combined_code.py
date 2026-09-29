import simpy
import random
import statistics
import multiprocessing as mp
import csv
import os
from functools import partial

RANDOM_SEED = 22
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME

POP_SIZE = 50
N_GEN = 100
N_CORES = 50

BUFFER_NAMES = [
    "post_loading_buffer",
    "post_conveyor_buffer",
    "post_washing_buffer",
    "pre_press1_buffer",
    "pre_press2_buffer",
    "post_press12_buffer",
]

VAR_MIN = 1
VAR_MAX = 10
N_VARS = len(BUFFER_NAMES)


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
    Split stream from input_store evenly into out1 and out2.
    If the preferred buffer is full, send to the other buffer.
    All buffers involved must have finite capacity.
    """
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
    """
    Merge two parallel streams a and b into out.
    All buffers must have defined capacities.
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

            self.repair_event.succeed()
            self.repair_event = self.env.event()

    def run(self):
        while True:
            with self.resource.request() as res_req:
                yield res_req

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
        yield output_buffer.put({"id": part_id})
        part_id += 1
        yield env.timeout(1)


def kwh_per_sec(x):
    return x / 3600.0


def run_simulation(seed, caps_dict=None, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    raw_input = simpy.Store(env, capacity=1000)

    def get_cap(name, default):
        if caps_dict is None:
            return default
        return int(caps_dict.get(name, default))

    post_loading_buffer = DelayBuffer(env, cap=get_cap("post_loading_buffer", 2), delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=get_cap("post_conveyor_buffer", 2), delay=10)
    post_washing_buffer = DelayBuffer(env, cap=get_cap("post_washing_buffer", 2), delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=get_cap("pre_press1_buffer", 3), delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=get_cap("pre_press2_buffer", 3), delay=32)
    post_press12_buffer = DelayBuffer(env, cap=get_cap("post_press12_buffer", 3), delay=32)

    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    pre_press_split = simpy.Store(env, capacity=6)
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
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_press_split,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

    env.process(splitter(env, pre_press_split, pre_press1_buffer, pre_press2_buffer))

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


def evaluate_individual(args, seed_offset=0):
    """
    Evaluate one individual.
    ind: list of capacities for the 6 delay buffers in the same order as BUFFER_NAMES.
    Returns (throughput, wip) or None if constraint violated.
    """
    ind, seed_offset = args

    if any((c < VAR_MIN or c > VAR_MAX) for c in ind):
        return None

    caps_dict = {name: cap for name, cap in zip(BUFFER_NAMES, ind)}
    seed = RANDOM_SEED + seed_offset

    res = run_simulation_with_caps(seed, caps_dict)
    throughput = res["overall"]["throughput"]
    wip = res["overall"]["wip"]

    if throughput <= 0:
        return None

    return throughput, wip


def dominates(a_obj, b_obj):
    """
    Return True if a dominates b.
    a_obj, b_obj: (throughput, wip)
    Maximize throughput, minimize wip.
    """
    a_t, a_w = a_obj
    b_t, b_w = b_obj

    not_worse = (a_t >= b_t) and (a_w <= b_w)
    strictly_better = (a_t > b_t) or (a_w < b_w)
    return not_worse and strictly_better


def fast_non_dominated_sort(pop_objs):
    """
    NSGA-II fast non-dominated sort.
    pop_objs: list of objective tuples or None for infeasible.
    Returns: list of fronts, each front is list of indices.
    """
    S = [[] for _ in pop_objs]
    n = [0 for _ in pop_objs]
    fronts = [[]]

    for p in range(len(pop_objs)):
        if pop_objs[p] is None:
            continue
        for q in range(len(pop_objs)):
            if pop_objs[q] is None or p == q:
                continue
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0 and pop_objs[p] is not None:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(front, pop_objs):
    """
    Compute crowding distance for a front.
    front: list of indices
    pop_objs: list of (throughput, wip)
    Returns: dict index -> distance
    """
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance

    for m in range(2):
        front_sorted = sorted(front, key=lambda i: pop_objs[i][m])
        f_min = pop_objs[front_sorted[0]][m]
        f_max = pop_objs[front_sorted[-1]][m]
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        if f_max == f_min:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_val = pop_objs[front_sorted[k - 1]][m]
            next_val = pop_objs[front_sorted[k + 1]][m]
            distance[front_sorted[k]] += (next_val - prev_val) / (f_max - f_min)
    return distance


def tournament_selection(pop, pop_objs, k=2):
    """
    Binary tournament selection based on dominance.
    pop: list of individuals
    pop_objs: list of objective tuples or None
    Returns: selected individual.
    """
    i, j = random.sample(range(len(pop)), 2)
    obj_i = pop_objs[i]
    obj_j = pop_objs[j]
    if obj_i is None and obj_j is not None:
        return pop[j]
    if obj_j is None and obj_i is not None:
        return pop[i]
    if obj_i is None and obj_j is None:
        return pop[i]
    if dominates(obj_i, obj_j):
        return pop[i]
    if dominates(obj_j, obj_i):
        return pop[j]
    return pop[i] if random.random() < 0.5 else pop[j]


def crossover(parent1, parent2, pc=0.9):
    """
    One-point crossover for integer vectors.
    """
    if random.random() > pc or len(parent1) < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    """
    Uniform mutation for integer variables in [VAR_MIN, VAR_MAX].
    """
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def create_initial_population():
    pop = []
    for _ in range(POP_SIZE):
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
        pop.append(ind)
    return pop


def run_simulation_with_caps(seed, caps_dict):
    """
    Wrapper around run_simulation that applies custom capacities.
    """
    return run_simulation(seed, caps_dict=caps_dict)


def evaluate_population(pop, pool, gen):
    """
    Evaluate all individuals in population using multiprocessing pool.
    Returns list of objective tuples or None for infeasible.
    """
    tasks = [(ind, gen * POP_SIZE + i) for i, ind in enumerate(pop)]
    results = pool.map(evaluate_individual, tasks)
    return results


def nsga2():
    random.seed(RANDOM_SEED)

    if N_CORES != 50:
        raise RuntimeError("N_CORES must be exactly 50.")
    if N_CORES > mp.cpu_count():
        raise RuntimeError(f"Requested {N_CORES} cores, but only {mp.cpu_count()} available.")

    pop = create_initial_population()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "moo_simulation_results.csv")
    write_header = not os.path.exists(csv_file)
    csv_f = open(csv_file, mode="w", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        header = (
            ["generation", "individual_index"]
            + [f"buffer_capacity_{name}" for name in BUFFER_NAMES]
            + ["throughput", "wip"]
        )
        writer.writerow(header)

    with mp.Pool(processes=N_CORES) as pool:
        for gen in range(N_GEN):
            pop_objs = evaluate_population(pop, pool, gen)

            for idx, (ind, obj) in enumerate(zip(pop, pop_objs)):
                if obj is None:
                    continue
                throughput, wip = obj
                row = [gen, idx] + ind + [throughput, wip]
                writer.writerow(row)

            csv_f.flush()

            fronts = fast_non_dominated_sort(pop_objs)
            new_pop = []
            for front in fronts:
                if len(front) == 0:
                    continue
                if len(new_pop) + len(front) > POP_SIZE:
                    distances = crowding_distance(front, pop_objs)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    remaining = POP_SIZE - len(new_pop)
                    new_pop.extend([pop[i] for i in sorted_front[:remaining]])
                    break
                else:
                    new_pop.extend([pop[i] for i in front])

            while len(new_pop) < POP_SIZE:
                ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
                new_pop.append(ind)

            offspring = []
            while len(offspring) < POP_SIZE:
                p1 = tournament_selection(new_pop, pop_objs)
                p2 = tournament_selection(new_pop, pop_objs)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                if len(offspring) < POP_SIZE:
                    offspring.append(c2)

            pop = offspring

    csv_f.close()


if __name__ == "__main__":
    nsga2()