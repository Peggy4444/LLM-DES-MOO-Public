import simpy
import random
import math
import statistics

RANDOM_SEED = 11

SIM_TIME = 691200          # total simulation time (s)
WARMUP = 86400             # warm-up time (s)
N_REPS = 10

INTERARRIVAL = 20.0        # assumed, not specified

DEFECT_RATE = 0.089

# Stations parameters (s, %, MTTR_s, kW_idle, kW_work)
STATIONS = {
    "Conveyor":      {"pt": 6.0,   "avail": 100.00, "mttr": 1.0,  "idle": 0.00, "work": 0.00},
    "Handling":      {"pt": 25.0,  "avail": 97.79,  "mttr": 74.0, "idle": 0.50, "work": 0.74},
    "Loading":       {"pt": 12.0,  "avail": 90.49,  "mttr": 68.0, "idle": 0.25, "work": 0.72},
    "Press1":        {"pt": 175.0, "avail": 87.79,  "mttr": 73.0, "idle": 1.25, "work": 1.28},
    "Press2":        {"pt": 176.0, "avail": 87.69,  "mttr": 74.0, "idle": 1.25, "work": 1.27},
    "Quality":       {"pt": 41.0,  "avail": 85.87,  "mttr": 66.0, "idle": 0.58, "work": 0.84},
    "Washing":       {"pt": 14.0,  "avail": 80.89,  "mttr": 269.0,"idle": 4.28, "work": 35.24},
}

BUFFERS = {
    "PostLoading":       {"cap": 2, "pt": 10.0},
    "PostConveyor":      {"cap": 2, "pt": 10.0},
    "PostWashing":       {"cap": 2, "pt": 10.0},
    "PrePress1":         {"cap": 3, "pt": 32.0},
    "PrePress2":         {"cap": 3, "pt": 32.0},
    "PostPress12":       {"cap": 3, "pt": 32.0},
}

# Derived MTBF from availability: A = MTBF / (MTBF + MTTR)
def mtbf_from_avail(avail, mttr):
    a = avail / 100.0
    if a >= 0.999999:
        return 1e20
    return mttr * a / (1 - a)

def is_weekend_stop(t):
    # Friday 17:00 -> Saturday 07:00
    # Saturday 17:00 -> Sunday 07:00
    day = (t // 86400) % 7  # 0=Monday
    tod = t % 86400
    h = tod / 3600.0
    if day == 4:  # Friday
        return h >= 17.0
    if day == 5:  # Saturday
        return True
    if day == 6:  # Sunday
        return h < 7.0
    return False

def time_to_next_run(t):
    # advance time in 1-minute increments until in working time
    while is_weekend_stop(t):
        t += 60
    return t

class MonitoredResource:
    def __init__(self, env, name, capacity, idle_power=0.0, work_power=0.0,
                 avail=100.0, mttr=0.0):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.res = simpy.Resource(env, capacity=capacity)
        self.idle_power = idle_power
        self.work_power = work_power
        self.mttr = mttr
        self.mtbf = mtbf_from_avail(avail, mttr)
        self.broken = False
        self.total_work_energy = 0.0
        self.total_idle_energy = 0.0
        self.last_time = env.now
        self.working_units = 0
        self.idle_units = capacity
        if self.mtbf < 1e19:
            env.process(self.breakdown_process())

    def _update_energy(self):
        now = self.env.now
        dt = now - self.last_time
        self.total_work_energy += self.working_units * self.work_power * dt / 3600.0
        self.total_idle_energy += self.idle_units * self.idle_power * dt / 3600.0
        self.last_time = now

    def request(self):
        return self._Request(self)

    class _Request:
        def __init__(self, parent):
            self.parent = parent
            self.req = parent.res.request()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.parent._update_energy()
            self.parent.working_units -= 1
            self.parent.idle_units += 1
            self.parent.res.release(self.req)

        def __await__(self):
            parent = self.parent
            req = self.req
            while True:
                if parent.broken:
                    yield parent.env.timeout(1)
                    continue
                result = yield req | parent.env.process(parent.wait_until_repaired())
                if req in result:
                    parent._update_energy()
                    parent.working_units += 1
                    parent.idle_units -= 1
                    return
                yield parent.env.timeout(0)

    def wait_until_repaired(self):
        while self.broken:
            yield self.env.timeout(1)

    def breakdown_process(self):
        while True:
            mtbf_sample = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(mtbf_sample)
            self.broken = True
            repair_time = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(repair_time)
            self.broken = False

class Buffer:
    def __init__(self, env, name, capacity, proc_time):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.proc_time = proc_time

    def put(self, item):
        return self.store.put(item)

    def get(self):
        return self.store.get()

    def process(self, item):
        yield self.env.timeout(self.proc_time)

class SystemModel:
    def __init__(self, env):
        self.env = env
        self.loading = MonitoredResource(env, "Loading",
                                         capacity=1,
                                         idle_power=STATIONS["Loading"]["idle"],
                                         work_power=STATIONS["Loading"]["work"],
                                         avail=STATIONS["Loading"]["avail"],
                                         mttr=STATIONS["Loading"]["mttr"])
        self.conveyor = MonitoredResource(env, "Conveyor",
                                          capacity=1,
                                          idle_power=STATIONS["Conveyor"]["idle"],
                                          work_power=STATIONS["Conveyor"]["work"],
                                          avail=STATIONS["Conveyor"]["avail"],
                                          mttr=STATIONS["Conveyor"]["mttr"])
        self.washing = MonitoredResource(env, "Washing",
                                         capacity=1,
                                         idle_power=STATIONS["Washing"]["idle"],
                                         work_power=STATIONS["Washing"]["work"],
                                         avail=STATIONS["Washing"]["avail"],
                                         mttr=STATIONS["Washing"]["mttr"])
        self.handling = MonitoredResource(env, "Handling",
                                          capacity=1,
                                          idle_power=STATIONS["Handling"]["idle"],
                                          work_power=STATIONS["Handling"]["work"],
                                          avail=STATIONS["Handling"]["avail"],
                                          mttr=STATIONS["Handling"]["mttr"])
        self.press1 = MonitoredResource(env, "Press1",
                                        capacity=1,
                                        idle_power=STATIONS["Press1"]["idle"],
                                        work_power=STATIONS["Press1"]["work"],
                                        avail=STATIONS["Press1"]["avail"],
                                        mttr=STATIONS["Press1"]["mttr"])
        self.press2 = MonitoredResource(env, "Press2",
                                        capacity=1,
                                        idle_power=STATIONS["Press2"]["idle"],
                                        work_power=STATIONS["Press2"]["work"],
                                        avail=STATIONS["Press2"]["avail"],
                                        mttr=STATIONS["Press2"]["mttr"])
        self.quality = MonitoredResource(env, "Quality",
                                         capacity=1,
                                         idle_power=STATIONS["Quality"]["idle"],
                                         work_power=STATIONS["Quality"]["work"],
                                         avail=STATIONS["Quality"]["avail"],
                                         mttr=STATIONS["Quality"]["mttr"])

        self.post_loading = Buffer(env, "PostLoading",
                                   BUFFERS["PostLoading"]["cap"],
                                   BUFFERS["PostLoading"]["pt"])
        self.post_conveyor = Buffer(env, "PostConveyor",
                                    BUFFERS["PostConveyor"]["cap"],
                                    BUFFERS["PostConveyor"]["pt"])
        self.post_washing = Buffer(env, "PostWashing",
                                   BUFFERS["PostWashing"]["cap"],
                                   BUFFERS["PostWashing"]["pt"])
        self.pre_press1 = Buffer(env, "PrePress1",
                                 BUFFERS["PrePress1"]["cap"],
                                 BUFFERS["PrePress1"]["pt"])
        self.pre_press2 = Buffer(env, "PrePress2",
                                 BUFFERS["PrePress2"]["cap"],
                                 BUFFERS["PrePress2"]["pt"])
        self.post_press12 = Buffer(env, "PostPress12",
                                   BUFFERS["PostPress12"]["cap"],
                                   BUFFERS["PostPress12"]["pt"])

        self.arrivals = 0
        self.good_parts = 0
        self.defects = 0
        self.wip_area = 0.0
        self.last_wip_time = env.now

        self.total_energy = 0.0

    def update_wip(self):
        now = self.env.now
        dt = now - self.last_wip_time
        wip = (len(self.post_loading.store.items) +
               len(self.post_conveyor.store.items) +
               len(self.post_washing.store.items) +
               len(self.pre_press1.store.items) +
               len(self.pre_press2.store.items) +
               len(self.post_press12.store.items))
        self.wip_area += wip * dt
        self.last_wip_time = now

    def aggregate_energy(self):
        self.total_energy = (
            self.loading.total_work_energy + self.loading.total_idle_energy +
            self.conveyor.total_work_energy + self.conveyor.total_idle_energy +
            self.washing.total_work_energy + self.washing.total_idle_energy +
            self.handling.total_work_energy + self.handling.total_idle_energy +
            self.press1.total_work_energy + self.press1.total_idle_energy +
            self.press2.total_work_energy + self.press2.total_idle_energy +
            self.quality.total_work_energy + self.quality.total_idle_energy
        )

def weekend_wait(env):
    while is_weekend_stop(env.now):
        t_next = time_to_next_run(env.now)
        yield env.timeout(t_next - env.now)
    return

def part_flow(env, system, name):
    system.arrivals += 1
    yield env.process(weekend_wait(env))
    req = system.loading.request()
    yield env.process(req)
    try:
        yield env.timeout(STATIONS["Loading"]["pt"])
    finally:
        req.__exit__(None, None, None)
    yield env.process(system.post_loading.process(name))
    yield system.post_loading.put(name)

    item = yield system.post_loading.get()
    yield env.process(weekend_wait(env))
    req = system.conveyor.request()
    yield env.process(req)
    try:
        yield env.timeout(STATIONS["Conveyor"]["pt"])
    finally:
        req.__exit__(None, None, None)
    yield env.process(system.post_conveyor.process(item))
    yield system.post_conveyor.put(item)

    item = yield system.post_conveyor.get()
    yield env.process(weekend_wait(env))
    req = system.washing.request()
    yield env.process(req)
    try:
        yield env.timeout(STATIONS["Washing"]["pt"])
    finally:
        req.__exit__(None, None, None)
    yield env.process(system.post_washing.process(item))
    yield system.post_washing.put(item)

    item = yield system.post_washing.get()
    yield env.process(weekend_wait(env))
    req = system.handling.request()
    yield env.process(req)
    try:
        yield env.timeout(STATIONS["Handling"]["pt"])
    finally:
        req.__exit__(None, None, None)

    # split evenly to Press1 / Press2
    if system.arrivals % 2 == 0:
        target_buf = system.pre_press1
    else:
        target_buf = system.pre_press2

    yield env.process(target_buf.process(item))
    yield target_buf.put(item)

    if target_buf is system.pre_press1:
        item = yield system.pre_press1.get()
        yield env.process(weekend_wait(env))
        req = system.press1.request()
        yield env.process(req)
        try:
            yield env.timeout(STATIONS["Press1"]["pt"])
        finally:
            req.__exit__(None, None, None)
    else:
        item = yield system.pre_press2.get()
        yield env.process(weekend_wait(env))
        req = system.press2.request()
        yield env.process(req)
        try:
            yield env.timeout(STATIONS["Press2"]["pt"])
        finally:
            req.__exit__(None, None, None)

    yield env.process(system.post_press12.process(item))
    yield system.post_press12.put(item)

    item = yield system.post_press12.get()
    yield env.process(weekend_wait(env))
    req = system.quality.request()
    yield env.process(req)
    try:
        yield env.timeout(STATIONS["Quality"]["pt"])
    finally:
        req.__exit__(None, None, None)

    if random.random() < DEFECT_RATE:
        system.defects += 1
        return
    system.good_parts += 1

def source(env, system):
    i = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / INTERARRIVAL))
        env.process(part_flow(env, system, f"Part_{i}"))
        i += 1

def run_replication(rep_id):
    random.seed(RANDOM_SEED + rep_id)
    env = simpy.Environment()
    system = SystemModel(env)
    env.process(source(env, system))

    def wip_tracker():
        while True:
            yield env.timeout(60)
            system.update_wip()
    env.process(wip_tracker())

    env.run(until=SIM_TIME)
    system.update_wip()
    system.aggregate_energy()

    effective_time = SIM_TIME - WARMUP
    good = system.good_parts
    throughput_per_hour = good / (effective_time / 3600.0) if effective_time > 0 else 0.0
    mean_wip = system.wip_area / SIM_TIME if SIM_TIME > 0 else 0.0
    energy_per_part = system.total_energy / good if good > 0 else 0.0
    return throughput_per_hour, mean_wip, energy_per_part

throughputs = []
wips = []
energies = []

for r in range(N_REPS):
    tp, w, e = run_replication(r)
    throughputs.append(tp)
    wips.append(w)
    energies.append(e)

mean_tp = statistics.mean(throughputs) if throughputs else 0.0
mean_wip = statistics.mean(wips) if wips else 0.0
mean_energy = statistics.mean(energies) if energies else 0.0

print("=== Mean Overall KPIs over 10 runs ===")
print("Throughput = {:.3f} parts/hour".format(mean_tp))
print("WIP = {:.3f} parts".format(mean_wip))
print("Mean Energy Consumption per Part = {:.6f} kWh/part".format(mean_energy))