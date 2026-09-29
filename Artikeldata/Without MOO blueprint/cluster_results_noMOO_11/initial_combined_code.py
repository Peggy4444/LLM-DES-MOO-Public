import simpy
import random
import statistics
import multiprocessing as mp
import csv
import os
from functools import partial

RANDOM_SEED = 11
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME

# MOO parameters
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
N_VARS = len(BUFFER_NAMES)
VAR_MIN = 1
VAR_MAX = 10


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


def run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL, buffer_caps=None):
    random.seed(seed)
    env = simpy.Environment()

    # Raw input buffer with explicit capacity
    raw_input = simpy.Store(env, capacity=1000)

    # Determine buffer capacities (either from buffer_caps or defaults)
    if buffer_caps is None:
        # Default capacities as in original code
        cap_post_loading = 2
        cap_post_conveyor = 2
        cap_post_washing = 2
        cap_pre_press1 = 3
        cap_pre_press2 = 3
        cap_post_press12 = 3
    else:
        # buffer_caps is a list of 6 integers in [1,10] in the order of BUFFER_NAMES
        (cap_post_loading,
         cap_post_conveyor,
         cap_post_washing,
         cap_pre_press1,
         cap_pre_press2,
         cap_post_press12) = buffer_caps

    # Process buffers with defined capacities and delays
    post_loading_buffer = DelayBuffer(env, cap=cap_post_loading, delay=10)     # PostLoadingBuffer
    post_conveyor_buffer = DelayBuffer(env, cap=cap_post_conveyor, delay=10)    # PostConveyorBuffer
    post_washing_buffer = DelayBuffer(env, cap=cap_post_washing, delay=10)     # PostWashingBuffer
    pre_press1_buffer = DelayBuffer(env, cap=cap_pre_press1, delay=32)       # PrePress1Buffer
    pre_press2_buffer = DelayBuffer(env, cap=cap_pre_press2, delay=32)       # PrePress2Buffer
    post_press12_buffer = DelayBuffer(env, cap=cap_post_press12, delay=32)     # PostPress1&Press2Buffer

    # Final and defect sinks with explicit capacities (sinks may be large/unbounded)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Loading robot -> Conveyor belt
    loading_robot = Machine(
        env, "Loading robot", input_buffer=raw_input, output_buffer=post_loading_buffer,
        process_time=12.0, availability=90.49, mttr=68.0,
        working_power=kwh_per_sec(0.72), waiting_power=kwh_per_sec(0.25),
    )

    # Conveyor belt -> Washing machine
    conveyor_belt = Machine(
        env, "Conveyor belt", input_buffer=post_loading_buffer, output_buffer=post_conveyor_buffer,
        process_time=6.0, availability=100.0, mttr=1.0,
        working_power=kwh_per_sec(0.0), waiting_power=kwh_per_sec(0.0),
    )

    # Washing machine -> Hantering cell
    washing_machine = Machine(
        env, "Washing machine", input_buffer=post_conveyor_buffer, output_buffer=post_washing_buffer,
        process_time=14.0, availability=80.89, mttr=269.0,
        working_power=kwh_per_sec(35.24), waiting_power=kwh_per_sec(4.28),
    )

    # Hantering cell -> PrePress buffers (split to presses)
    # First go into an intermediate store with explicit capacity, then split evenly
    pre_press_split_input = simpy.Store(env, capacity=6)

    hantering_cell = Machine(
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_press_split_input,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

    # Split evenly into the two pre-press delay buffers, respecting capacities
    env.process(splitter(env, pre_press_split_input, pre_press1_buffer, pre_press2_buffer))

    # Parallel relationships: Presses cell 1 || Presses cell 2
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

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

    # Merge outputs of the parallel presses into the post-press buffer
    merger(env, press1_out, press2_out, post_press12_buffer)

    # Presses -> Quality station cell
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

    # Constant arrival of raw parts into raw_input with defined capacity
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
            # Sample every 10 minutes
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


def run_simulation_with_buffers(seed, buffer_caps):
    # buffer_caps is a list of 6 integers in [1,10]
    return run_simulation(seed, buffer_caps=buffer_caps)


def evaluate_individual(ind, base_seed):
    # Constraint: sum of all buffer capacities <= 30
    if sum(ind) > 30:
        return None

    runs = 3
    throughputs = []
    wips = []
    for i in range(runs):
        seed = base_seed + i
        res = run_simulation_with_buffers(seed, ind)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])
    avg_throughput = sum(throughputs) / runs
    avg_wip = sum(wips) / runs
    return avg_wip, -avg_throughput  # minimize wip, maximize throughput -> minimize -throughput


def dominates(a, b):
    # a and b are tuples (f1, f2), lower is better
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def fast_non_dominated_sort(pop_objs):
    S = [[] for _ in range(len(pop_objs))]
    n = [0 for _ in range(len(pop_objs))]
    rank = [0 for _ in range(len(pop_objs))]
    fronts = [[]]

    for p in range(len(pop_objs)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop_objs)):
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
    distance = {i: 0.0 for i in front}
    if len(front) == 0:
        return distance
    n_obj = len(pop_objs[0])
    for m in range(n_obj):
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
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None:
            best = i
        else:
            # Compare rank and crowding later; here we just use objective sum as a simple proxy
            if sum(pop_objs[i]) < sum(pop_objs[best]):
                best = i
    return pop[best]


def crossover(p1, p2, pc=0.9):
    if random.random() > pc:
        return p1[:], p2[:]
    point = random.randint(1, N_VARS - 1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    for i in range(N_VARS):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def nsga2_optimize():
    random.seed(RANDOM_SEED)
    base_seed = RANDOM_SEED + 10000

    # Initialize population
    population = []
    while len(population) < POP_SIZE:
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
        if sum(ind) <= 30:
            population.append(ind)

    # Ensure exactly 50 cores are used
    pool = mp.Pool(processes=N_CORES)
    eval_func = partial(evaluate_individual, base_seed=base_seed)

    # Prepare results directory and file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "moo_simulation_results.csv")
    if os.path.exists(results_file):
        os.remove(results_file)
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["post_loading_buffer_capacity",
             "post_conveyor_buffer_capacity",
             "post_washing_buffer_capacity",
             "pre_press1_buffer_capacity",
             "pre_press2_buffer_capacity",
             "post_press12_buffer_capacity"] +
            ["wip", "throughput"]
        )
        writer.writerow(header)

    for gen in range(N_GEN):
        objs = pool.map(eval_func, population)
        valid_indices = [i for i, o in enumerate(objs) if o is not None]
        population = [population[i] for i in valid_indices]
        objs = [objs[i] for i in valid_indices]

        # Write valid individuals to CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            for ind, (wip, neg_throughput) in zip(population, objs):
                writer.writerow(ind + [wip, -neg_throughput])

        if not population:
            # If all individuals are infeasible, reinitialize population
            while len(population) < POP_SIZE:
                ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
                if sum(ind) <= 30:
                    population.append(ind)
            continue

        # NSGA-II selection
        fronts, rank = fast_non_dominated_sort(objs)
        new_population = []
        while len(new_population) < POP_SIZE:
            for front in fronts:
                if len(new_population) + len(front) > POP_SIZE:
                    dist = crowding_distance(front, objs)
                    sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                    for i in sorted_front:
                        if len(new_population) < POP_SIZE:
                            new_population.append(population[i])
                        else:
                            break
                    break
                else:
                    for i in front:
                        new_population.append(population[i])
            if len(new_population) >= POP_SIZE:
                break

        population = new_population

        # Create offspring
        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(population, objs)
            p2 = tournament_selection(population, objs)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            if sum(c1) <= 30:
                offspring.append(c1)
            if len(offspring) < POP_SIZE and sum(c2) <= 30:
                offspring.append(c2)

        population = offspring

    pool.close()
    pool.join()


if __name__ == "__main__":
    nsga2_optimize()