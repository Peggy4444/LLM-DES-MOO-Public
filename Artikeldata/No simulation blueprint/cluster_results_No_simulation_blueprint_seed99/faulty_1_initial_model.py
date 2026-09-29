import simpy
import random
import math
import statistics

RANDOM_SEED = 99

SIM_TIME = 691200
WARM_UP = 86400
REPLICATIONS = 10

# Interarrival time (assumed)
MEAN_INTERARRIVAL = 60.0  # seconds, adjust as needed

DEFECT_RATE = 0.089

# Machine parameters: (mean_proc_time, availability, MTTR, idle_energy, work_energy)
MACHINES = {
    "Conveyor":        (6.0,   1.0000,   1.0,   0.00,  0.00),
    "Handling":        (25.0,  0.9779,  74.0,   0.50,  0.74),
    "Loading":         (12.0,  0.9049,  68.0,   0.25,  0.72),
    "Press1":          (175.0, 0.8779,  73.0,   1.25,  1.28),
    "Press2":          (176.0, 0.8769,  74.0,   1.25,  1.27),
    "Quality":         (41.0,  0.8587,  66.0,   0.58,  0.84),
    "Washing":         (14.0,  0.8089, 269.0,   4.28, 35.24),
}

# Buffer specs: (capacity, process_time)
BUFFERS = {
    "PostLoadingBuffer":      (2, 10.0),
    "PostConveyorBuffer":     (2, 10.0),
    "PostWashingBuffer":      (2, 10.0),
    "PrePress1Buffer":        (3, 32.0),
    "PrePress2Buffer":        (3, 32.0),
    "PostPress1_2Buffer":     (3, 32.0),
}

# -------------------------------------------------------------------
# Calendar: factory closed Fri 17:00–Sat 07:00 and Sat 17:00–Sun 07:00
# A week = 7*24*3600 = 604800 s, days: 0=Mon ... 4=Fri ... 6=Sun
# Return True if plant open at absolute time t
# -------------------------------------------------------------------
DAY = 24 * 3600

def is_open(t):
    t_week = t % (7 * DAY)
    day = int(t_week // DAY)
    sec = t_week % DAY
    # Closed Fri 17:00–Sat 07:00
    if day == 4 and sec >= 17 * 3600:
        return False
    if day == 5 and sec < 7 * 3600:
        return False
    # Closed Sat 17:00–Sun 07:00
    if day == 5 and sec >= 17 * 3600:
        return False
    if day == 6 and sec < 7 * 3600:
        return False
    return True

def wait_until_open(env):
    while not is_open(env.now):
        # jump to next second until open; to speed up, jump in chunks
        # but make sure not to skip an opening moment
        yield env.timeout(60)

# -------------------------------------------------------------------
# Machine with availability (up/down) and energy tracking
# -------------------------------------------------------------------
class Machine:
    def __init__(self, env, name, mean_proc, availability, mttr,
                 idle_energy, work_energy, stats):
        self.env = env
        self.name = name
        self.mean_proc = mean_proc
        self.mttr = mttr
        self.idle_energy = idle_energy      # kW
        self.work_energy = work_energy      # kW
        self.stats = stats

        self.resource = simpy.Resource(env, capacity=1)

        # failure parameters
        if availability <= 0:
            self.mttf = float("inf")
        else:
            self.mttf = mttr * availability / (1.0 - availability)

        self.up = True
        self.proc_interrupt = env.event()  # dummy, replaced per job

        # background failure process
        env.process(self.failure_process())

    def failure_process(self):
        while True:
            if not self.up:
                yield self.env.timeout(1)
                continue
            ttf = random.expovariate(1.0 / self.mttf) if self.mttf < float("inf") else float("inf")
            yield self.env.timeout(ttf)
            if not is_open(self.env.now):
                continue
            self.up = False
            # repair time
            ttr = random.expovariate(1.0 / self.mttr)
            while ttr > 0:
                if is_open(self.env.now):
                    dt = min(60, ttr)
                else:
                    dt = 60
                yield self.env.timeout(dt)
                if is_open(self.env.now):
                    ttr -= dt
            self.up = True

    def process(self, part):
        with self.resource.request() as req:
            yield req
            yield self.env.process(self._run_job(part))

    def _run_job(self, part):
        # Wait for calendar opening
        yield from wait_until_open(self.env)

        start = self.env.now
        remaining = random.expovariate(1.0 / self.mean_proc)

        # energy tracking in kWh: kW * h
        if self.env.now >= WARM_UP:
            self.stats['energy'] += self.idle_energy * 0  # no idle while in process

        while remaining > 0:
            if not is_open(self.env.now) or not self.up:
                yield self.env.timeout(60)
                continue
            dt = min(1.0, remaining)
            yield self.env.timeout(dt)
            remaining -= dt
            if self.env.now >= WARM_UP:
                self.stats['energy'] += self.work_energy * (dt / 3600.0)

        end = self.env.now
        if self.env.now >= WARM_UP:
            self.stats['machine_busy_time'][self.name] += (end - start)

# -------------------------------------------------------------------
# Buffer with capacity and processing delay
# -------------------------------------------------------------------
class Buffer:
    def __init__(self, env, name, capacity, proc_time):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.proc_time = proc_time
        self.store = simpy.Store(env, capacity=capacity)

    def put(self, item):
        return self.store.put(item)

    def get_and_process(self):
        item = yield self.store.get()
        # calendar-aware processing
        remaining = random.expovariate(1.0 / self.proc_time)
        while remaining > 0:
            if not is_open(self.env.now):
                yield self.env.timeout(60)
                continue
            dt = min(1.0, remaining)
            yield self.env.timeout(dt)
            remaining -= dt
        return item

# -------------------------------------------------------------------
# Part generator
# -------------------------------------------------------------------
def source(env, loading, post_loading_buffer, stats):
    i = 0
    while True:
        if is_open(env.now):
            inter = random.expovariate(1.0 / MEAN_INTERARRIVAL)
        else:
            inter = 60.0
        yield env.timeout(inter)
        if not is_open(env.now):
            continue
        i += 1
        part = {'id': i, 'birth': env.now}
        env.process(part_flow(env, part, loading, post_loading_buffer, stats))

# -------------------------------------------------------------------
# Part flow through the system
# Loading -> PostLoadingBuffer -> Conveyor -> PostConveyorBuffer
# -> Washing -> PostWashingBuffer -> Handling -> (split to Press1/2)
# -> Press1/2 -> PostPress1_2Buffer -> Quality -> Sink/Defect
# -------------------------------------------------------------------
def part_flow(env, part, loading, post_loading_buffer, stats):
    # Loading robot
    yield env.process(loading.process(part))
    yield post_loading_buffer.put(part)

    # PostLoadingBuffer -> Conveyor
    part = yield env.process(post_loading_buffer.get_and_process())
    yield env.process(stats['machines']['Conveyor'].process(part))

    # PostConveyorBuffer
    pcb = stats['buffers']['PostConveyorBuffer']
    yield pcb.put(part)
    part = yield env.process(pcb.get_and_process())

    # Conveyor -> Washing
    yield env.process(stats['machines']['Washing'].process(part))

    # PostWashingBuffer
    pwb = stats['buffers']['PostWashingBuffer']
    yield pwb.put(part)
    part = yield env.process(pwb.get_and_process())

    # Washing -> Handling
    yield env.process(stats['machines']['Handling'].process(part))

    # Split to Press1 or Press2 evenly (based on global counter)
    stats['press_counter'] += 1
    if stats['press_counter'] % 2 == 1:
        target_pre = stats['buffers']['PrePress1Buffer']
        press_machine = stats['machines']['Press1']
    else:
        target_pre = stats['buffers']['PrePress2Buffer']
        press_machine = stats['machines']['Press2']

    # PrePress buffer
    yield target_pre.put(part)
    part = yield env.process(target_pre.get_and_process())

    # Press
    yield env.process(press_machine.process(part))

    # PostPress1&2Buffer
    postp = stats['buffers']['PostPress1_2Buffer']
    yield postp.put(part)
    part = yield env.process(postp.get_and_process())

    # Quality station
    yield env.process(stats['machines']['Quality'].process(part))

    # Defect check
    if random.random() < DEFECT_RATE:
        # defect sink
        if env.now >= WARM_UP:
            stats['defects'] += 1
        return

    # Good part to sink
    if env.now >= WARM_UP:
        stats['throughput'] += 1
        stats['flow_times'].append(env.now - part['birth'])

# -------------------------------------------------------------------
# Build model and run one replication
# -------------------------------------------------------------------
def run_replication(rep):
    random.seed(RANDOM_SEED + rep)
    env = simpy.Environment()

    stats = {
        'throughput': 0,
        'defects': 0,
        'flow_times': [],
        'energy': 0.0,
        'machine_busy_time': {m: 0.0 for m in MACHINES.keys()},
        'press_counter': 0,
        'buffers': {},
        'machines': {}
    }

    # Instantiate buffers
    for name, (cap, pt) in BUFFERS.items():
        stats['buffers'][name] = Buffer(env, name, cap, pt)

    # Instantiate machines
    for name, (pt, avail, mttr, idle_e, work_e) in MACHINES.items():
        stats['machines'][name] = Machine(env, name, pt, avail, mttr,
                                          idle_e, work_e, stats)

    loading = stats['machines']['Loading']
    post_loading_buffer = stats['buffers']['PostLoadingBuffer']

    env.process(source(env, loading, post_loading_buffer, stats))

    env.run(until=SIM_TIME)

    run_time_hours = max(1.0, (SIM_TIME - WARM_UP) / 3600.0)
    throughput_rate = stats['throughput'] / run_time_hours
    wip = 0
    for buf in stats['buffers'].values():
        wip += len(buf.store.items)
    mean_energy_per_part = (stats['energy'] / stats['throughput']) if stats['throughput'] > 0 else 0.0

    return {
        'throughput_rate': throughput_rate,
        'wip': wip,
        'mean_energy_per_part': mean_energy_per_part
    }

# -------------------------------------------------------------------
# Run all replications and aggregate
# -------------------------------------------------------------------
all_throughput = []
all_wip = []
all_energy = []

for r in range(REPLICATIONS):
    res = run_replication(r)
    all_throughput.append(res['throughput_rate'])
    all_wip.append(res['wip'])
    all_energy.append(res['mean_energy_per_part'])

mean_throughput = statistics.mean(all_throughput) if all_throughput else 0.0
mean_wip = statistics.mean(all_wip) if all_wip else 0.0
mean_energy = statistics.mean(all_energy) if all_energy else 0.0

print("=== Mean Overall KPIs over 10 runs ===")
print(f"Throughput = {mean_throughput:.3f} parts/hour")
print(f"WIP = {mean_wip:.3f} parts")
print(f"Mean Energy Consumption per Part = {mean_energy:.6f} kWh/part")