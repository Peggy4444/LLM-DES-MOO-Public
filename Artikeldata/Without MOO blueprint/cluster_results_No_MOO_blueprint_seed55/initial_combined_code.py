import simpy
import random
import statistics
import multiprocessing as mp
import csv
import os
from functools import partial

RANDOM_SEED = 55
SIM_TIME = 691200          # 8 days in seconds
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME
REPLICATIONS = 10

# MOO parameters
POP_SIZE = 50
N_GEN = 100
N_CORES = 50
RANDOM_SEED_MOO = 1234

# Decision variables: capacities of all buffers (1-10, integer)
# Order:
# 0: raw_input
# 1: post_loading_buffer
# 2: post_conveyor_buffer
# 3: post_washing_buffer
# 4: pre_press1_buffer
# 5: pre_press2_buffer
# 6: post_press12_buffer
# 7: pre_press_split_buffer
# 8: press1_out
# 9: press2_out

VAR_MIN = 1
VAR_MAX = 10
N_VARS = 10

# Constraint: sum of capacities <= 60
def is_feasible(ind):
    return sum(ind) <= 60


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
    """Split stream approximately evenly; respect finite capacities."""
    toggle = 0
    while True:
        part = yield input_store.get()
        first, second = (out1, out2) if toggle == 0 else (out2, out1)

        # Try preferred output if it has free capacity, otherwise use the other.
        if _has_free_capacity(first):
            yield first.put(part)
            toggle ^= 1
        else:
            # if second is also full, this will block until it has capacity
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
        # respect raw buffer capacity
        yield output_buffer.put(part)
        part_id += 1
        yield env.timeout(1)


def kwh_per_sec(x):
    return x / 3600.0


def run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL, buffer_caps=None):
    random.seed(seed)
    env = simpy.Environment()

    # Default capacities if none provided
    if buffer_caps is None:
        buffer_caps = [1000, 2, 2, 2, 3, 3, 3, 6, 3, 3]
    # Map decision variables to capacities
    raw_input_cap = buffer_caps[0]
    post_loading_cap = buffer_caps[1]
    post_conveyor_cap = buffer_caps[2]
    post_washing_cap = buffer_caps[3]
    pre_press1_cap = buffer_caps[4]
    pre_press2_cap = buffer_caps[5]
    post_press12_cap = buffer_caps[6]
    pre_press_split_cap = buffer_caps[7]
    press1_out_cap = buffer_caps[8]
    press2_out_cap = buffer_caps[9]

    # Raw input buffer – finite, defined capacity
    raw_input = simpy.Store(env, capacity=raw_input_cap)

    # Delay buffers with given capacities and process times
    post_loading_buffer = DelayBuffer(env, cap=post_loading_cap, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=post_conveyor_cap, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=post_washing_cap, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=pre_press1_cap, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=pre_press2_cap, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=post_press12_cap, delay=32)

    # Helper / normal buffers between parallel machines with defined capacity
    pre_press_split_buffer = simpy.Store(env, capacity=pre_press_split_cap)  # output of Hantering, input to splitter
    press1_out = simpy.Store(env, capacity=press1_out_cap)              # output of Press cell 1
    press2_out = simpy.Store(env, capacity=press2_out_cap)              # output of Press cell 2

    # Sinks with finite but large capacity
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Machines
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
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_press_split_buffer,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

    # Split from common pre-press buffer into two pre-press delay buffers (parallel route)
    env.process(splitter(env, pre_press_split_buffer, pre_press1_buffer, pre_press2_buffer))

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

    # Merge the two parallel press outputs into the common post-press delay buffer
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

    # Warmup
    env.run(until=warmup)

    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    delay_buffers = [
        post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
        pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    ]

    # WIP definition: items in delay buffers + in transit + in machines
    # Exclude helper stores and raw input

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


def decode_and_run(ind, base_seed):
    throughputs = []
    wips = []

    for r in range(REPLICATIONS):
        seed = base_seed + r
        res = run_simulation(seed, buffer_caps=ind)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = sum(throughputs) / len(throughputs)
    avg_wip = sum(wips) / len(wips)
    return avg_wip, -avg_throughput


def init_individual():
    return [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]


def tournament_selection(pop, k=2):
    best = None
    for _ in range(k):
        ind = random.choice(pop)
        if best is None:
            best = ind
        else:
            if dominates(ind, best):
                best = ind
    return best


def dominates(a, b):
    # a dominates b if a is no worse in all objectives and better in at least one
    not_worse = all(x <= y for x, y in zip(a["objs"], b["objs"]))
    strictly_better = any(x < y for x, y in zip(a["objs"], b["objs"]))
    return not_worse and strictly_better


def fast_non_dominated_sort(pop):
    fronts = []
    S = {}
    n = {}
    rank = {}

    for p in range(len(pop)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop)):
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
    front = [i for i in range(len(pop)) if n[i] == 0]
    fronts.append(front)

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
    return fronts


def crowding_distance_assignment(front, pop):
    l = len(front)
    if l == 0:
        return {}
    distances = {i: 0.0 for i in front}
    n_obj = len(pop[0]["objs"])

    for m in range(n_obj):
        front_sorted = sorted(front, key=lambda i: pop[i]["objs"][m])
        distances[front_sorted[0]] = float("inf")
        distances[front_sorted[-1]] = float("inf")
        obj_min = pop[front_sorted[0]]["objs"][m]
        obj_max = pop[front_sorted[-1]]["objs"][m]
        if obj_max == obj_min:
            continue
        for k in range(1, l - 1):
            prev_obj = pop[front_sorted[k - 1]]["objs"][m]
            next_obj = pop[front_sorted[k + 1]]["objs"][m]
            distances[front_sorted[k]] += (next_obj - prev_obj) / (obj_max - obj_min)
    return distances


def crossover(parent1, parent2, pc=0.9):
    if random.random() > pc:
        return parent1[:], parent2[:]
    point = random.randint(1, N_VARS - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    for i in range(N_VARS):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def evaluate_population(pop, pool, base_seed, all_solutions):
    # Evaluate only individuals that are feasible and not yet evaluated
    tasks = []
    idx_map = []
    for i, ind in enumerate(pop):
        if ind["objs"] is None and ind["feasible"]:
            tasks.append(ind["vars"])
            idx_map.append(i)

    if tasks:
        func = partial(decode_and_run, base_seed=base_seed)
        results = pool.map(func, tasks)
        for idx, objs in zip(idx_map, results):
            pop[idx]["objs"] = objs
            # Store all evaluated feasible solutions
            vars_ = pop[idx]["vars"]
            wip = objs[0]
            throughput = -objs[1]
            all_solutions.append({
                "raw_input_capacity": vars_[0],
                "post_loading_buffer_capacity": vars_[1],
                "post_conveyor_buffer_capacity": vars_[2],
                "post_washing_buffer_capacity": vars_[3],
                "pre_press1_buffer_capacity": vars_[4],
                "pre_press2_buffer_capacity": vars_[5],
                "post_press12_buffer_capacity": vars_[6],
                "pre_press_split_buffer_capacity": vars_[7],
                "press1_out_buffer_capacity": vars_[8],
                "press2_out_buffer_capacity": vars_[9],
                "wip": wip,
                "throughput": throughput
            })


def create_initial_population():
    pop = []
    while len(pop) < POP_SIZE:
        vars_ = init_individual()
        feasible = is_feasible(vars_)
        if not feasible:
            continue
        pop.append({"vars": vars_, "objs": None, "feasible": True})
    return pop


def make_offspring(pop):
    offspring = []
    while len(offspring) < POP_SIZE:
        p1 = tournament_selection(pop)
        p2 = tournament_selection(pop)
        c1_vars, c2_vars = crossover(p1["vars"], p2["vars"])
        c1_vars = mutate(c1_vars)
        c2_vars = mutate(c2_vars)

        for child_vars in (c1_vars, c2_vars):
            if len(offspring) >= POP_SIZE:
                break
            if not is_feasible(child_vars):
                continue
            offspring.append({"vars": child_vars, "objs": None, "feasible": True})
    return offspring


def nsga2():
    random.seed(RANDOM_SEED_MOO)
    all_solutions = []

    with mp.Pool(processes=N_CORES) as pool:
        pop = create_initial_population()
        evaluate_population(pop, pool, base_seed=10000, all_solutions=all_solutions)

        for gen in range(N_GEN):
            offspring = make_offspring(pop)
            evaluate_population(offspring, pool, base_seed=10000 + (gen + 1) * 1000, all_solutions=all_solutions)

            combined = pop + offspring
            fronts = fast_non_dominated_sort(combined)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) > POP_SIZE:
                    distances = crowding_distance_assignment(front, combined)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    needed = POP_SIZE - len(new_pop)
                    new_pop.extend([combined[i] for i in sorted_front[:needed]])
                    break
                else:
                    new_pop.extend([combined[i] for i in front])
            pop = new_pop

        # Final non-dominated front (not strictly needed for CSV, but kept for completeness)
        fronts = fast_non_dominated_sort(pop)
        best_front = fronts[0]
        pareto_set = [pop[i] for i in best_front]
        return pareto_set, all_solutions


def save_results(all_solutions, filename="results/moo_simulation_results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fieldnames = [
        "raw_input_capacity",
        "post_loading_buffer_capacity",
        "post_conveyor_buffer_capacity",
        "post_washing_buffer_capacity",
        "pre_press1_buffer_capacity",
        "pre_press2_buffer_capacity",
        "post_press12_buffer_capacity",
        "pre_press_split_buffer_capacity",
        "press1_out_buffer_capacity",
        "press2_out_buffer_capacity",
        "wip",
        "throughput"
    ]
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_solutions:
            writer.writerow(row)


if __name__ == "__main__":
    pareto, all_solutions = nsga2()
    save_results(all_solutions, filename="results/moo_simulation_results.csv")