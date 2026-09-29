import simpy
import random
import statistics
import multiprocessing as mp
import csv
import os

RANDOM_SEED = 33
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME
REPLICATIONS = 10

POP_SIZE = 50
N_GEN = 100
N_CORES = 4
RANDOM_SEED_MOO = 1234

# Decision variables: capacities of all buffers (1-10, integers)
# Order:
# 0: post_loading_buffer.cap
# 1: post_conveyor_buffer.cap
# 2: post_washing_buffer.cap
# 3: pre_press1_buffer.cap
# 4: pre_press2_buffer.cap
# 5: post_press12_buffer.cap
# 6: pre_press_split.capacity
# 7: press1_out.capacity
# 8: press2_out.capacity
# 9: raw_input.capacity
N_VAR = 10
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
    """
    Split parts roughly evenly between out1 and out2.
    If the preferred buffer is full, send to the other one.
    Assumes all buffers (including DelayBuffer) have a defined capacity.
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
    Merge from two input buffers into a single output buffer.
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
    """
    Simple source: produces one part per second into the raw input buffer.
    The raw buffer has a fixed capacity, so upstream flow is limited.
    """
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

    if buffer_caps is None:
        buffer_caps = {}

    # Raw input buffer with explicit capacity (source-limited)
    raw_input_cap = buffer_caps.get("raw_input", 1000)
    raw_input = simpy.Store(env, capacity=raw_input_cap)

    # Delay buffers with given capacities and process times
    post_loading_cap = buffer_caps.get("post_loading_buffer", 2)
    post_conveyor_cap = buffer_caps.get("post_conveyor_buffer", 2)
    post_washing_cap = buffer_caps.get("post_washing_buffer", 2)
    pre_press1_cap = buffer_caps.get("pre_press1_buffer", 3)
    pre_press2_cap = buffer_caps.get("pre_press2_buffer", 3)
    post_press12_cap = buffer_caps.get("post_press12_buffer", 3)

    post_loading_buffer = DelayBuffer(env, cap=post_loading_cap, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=post_conveyor_cap, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=post_washing_cap, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=pre_press1_cap, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=pre_press2_cap, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=post_press12_cap, delay=32)

    # Helper / normal (non-delay) buffers with explicit capacities
    pre_press_split_cap = buffer_caps.get("pre_press_split", 3)
    press1_out_cap = buffer_caps.get("press1_out", 3)
    press2_out_cap = buffer_caps.get("press2_out", 3)

    pre_press_split = simpy.Store(env, capacity=pre_press_split_cap)   # between Hantering cell and splitter
    press1_out = simpy.Store(env, capacity=press1_out_cap)
    press2_out = simpy.Store(env, capacity=press2_out_cap)

    # Sinks with large capacity (treated as sinks)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Machines as per station table
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

    # Splitter: splits the stream from Hantering cell to the two pre-press delay buffers
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

    # Merger: merge outputs of the two presses into the post-press delay buffer
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

    # Part generator feeding the raw input buffer (source)
    env.process(part_generator(env, raw_input))

    # Warmup
    env.run(until=warmup)

    # Reset machine stats after warmup
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    # Only the named delay buffers are part of WIP definition
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


def is_feasible(x):
    # Constraint: sum of all capacities <= 60
    return sum(x) <= 60


def evaluate_individual(x):
    if not is_feasible(x):
        # Return None to indicate infeasible; caller must skip
        return None

    buffer_caps = {
        "post_loading_buffer": x[0],
        "post_conveyor_buffer": x[1],
        "post_washing_buffer": x[2],
        "pre_press1_buffer": x[3],
        "pre_press2_buffer": x[4],
        "post_press12_buffer": x[5],
        "pre_press_split": x[6],
        "press1_out": x[7],
        "press2_out": x[8],
        "raw_input": x[9],
    }

    throughputs = []
    wips = []

    for i in range(REPLICATIONS):
        seed = RANDOM_SEED + i
        res = run_simulation(seed, buffer_caps=buffer_caps)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = sum(throughputs) / len(throughputs)
    avg_wip = sum(wips) / len(wips)

    # Objectives: f1 = WIP (minimize), f2 = -throughput (minimize => throughput maximize)
    return (avg_wip, -avg_throughput)


def init_population():
    pop = []
    for _ in range(POP_SIZE):
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VAR)]
        pop.append(ind)
    return pop


def tournament_selection(pop, fitnesses, k=2):
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None or dominates(fitnesses[i], fitnesses[best]):
            best = i
    return pop[best]


def dominates(f1, f2):
    # Minimization: f1 dominates f2 if f1 is no worse in all and better in at least one
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1[0] < f2[0] or f1[1] < f2[1])


def crossover(parent1, parent2, pc=0.9):
    if random.random() > pc:
        return parent1[:], parent2[:]
    point = random.randint(1, N_VAR - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    for i in range(N_VAR):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def non_dominated_sort(pop, fitnesses):
    S = [[] for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    rank = [0 for _ in range(len(pop))]
    fronts = [[]]

    for p in range(len(pop)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop)):
            if dominates(fitnesses[p], fitnesses[q]):
                S[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]):
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


def crowding_distance(front, fitnesses):
    distance = [0.0 for _ in front]
    if len(front) == 0:
        return distance
    num_obj = len(fitnesses[0])

    for m in range(num_obj):
        front_sorted = sorted(range(len(front)), key=lambda i: fitnesses[front[i]][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        f_min = fitnesses[front[front_sorted[0]]][m]
        f_max = fitnesses[front[front_sorted[-1]]][m]
        if f_max == f_min:
            continue
        for i in range(1, len(front) - 1):
            prev_f = fitnesses[front[front_sorted[i - 1]]][m]
            next_f = fitnesses[front[front_sorted[i + 1]]][m]
            distance[front_sorted[i]] += (next_f - prev_f) / (f_max - f_min)
    return distance


def select_next_generation(pop, fitnesses):
    fronts, _ = non_dominated_sort(pop, fitnesses)
    new_pop = []
    new_fit = []

    for front in fronts:
        if len(new_pop) + len(front) > POP_SIZE:
            dist = crowding_distance(front, fitnesses)
            sorted_front = sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
            for idx in sorted_front:
                if len(new_pop) >= POP_SIZE:
                    break
                new_pop.append(pop[front[idx]])
                new_fit.append(fitnesses[front[idx]])
            break
        else:
            for idx in front:
                new_pop.append(pop[idx])
                new_fit.append(fitnesses[idx])

    return new_pop, new_fit


def evaluate_population(pop, pool):
    results = pool.map(evaluate_individual, pop)
    new_pop = []
    new_fit = []
    for ind, fit in zip(pop, results):
        if fit is not None:
            new_pop.append(ind)
            new_fit.append(fit)
    return new_pop, new_fit


def main():
    random.seed(RANDOM_SEED_MOO)

    if N_CORES > mp.cpu_count():
        raise RuntimeError(f"Requested {N_CORES} cores, but only {mp.cpu_count()} available.")

    all_solutions = []

    with mp.Pool(processes=N_CORES) as pool:
        pop = init_population()
        pop, fitnesses = evaluate_population(pop, pool)

        # If all individuals are infeasible, reinitialize until we get some feasible ones
        while len(pop) == 0:
            pop = init_population()
            pop, fitnesses = evaluate_population(pop, pool)

        # Store initial feasible population
        for ind, fit in zip(pop, fitnesses):
            wip = fit[0]
            throughput = -fit[1]
            all_solutions.append(ind + [wip, throughput])

        for gen in range(N_GEN):
            offspring = []
            while len(offspring) < POP_SIZE:
                p1 = tournament_selection(pop, fitnesses)
                p2 = tournament_selection(pop, fitnesses)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                if len(offspring) < POP_SIZE:
                    offspring.append(c2)

            offspring, off_fit = evaluate_population(offspring, pool)

            # If all offspring infeasible, keep current population
            if len(offspring) == 0:
                continue

            # Store all feasible offspring solutions
            for ind, fit in zip(offspring, off_fit):
                wip = fit[0]
                throughput = -fit[1]
                all_solutions.append(ind + [wip, throughput])

            combined_pop = pop + offspring
            combined_fit = fitnesses + off_fit

            pop, fitnesses = select_next_generation(combined_pop, combined_fit)

    # Prepare results directory and CSV output
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "moo_simulation_results.csv")

    header = [
        "post_loading_buffer_capacity",
        "post_conveyor_buffer_capacity",
        "post_washing_buffer_capacity",
        "pre_press1_buffer_capacity",
        "pre_press2_buffer_capacity",
        "post_press12_buffer_capacity",
        "pre_press_split_buffer_capacity",
        "press1_out_buffer_capacity",
        "press2_out_buffer_capacity",
        "raw_input_buffer_capacity",
        "wip",
        "throughput"
    ]

    with open(out_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in all_solutions:
            writer.writerow(row)


if __name__ == "__main__":
    main()