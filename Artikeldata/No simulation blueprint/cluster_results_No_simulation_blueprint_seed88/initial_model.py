import simpy
import random
import statistics
import math

RANDOM_SEED = 88

SIM_TIME = 691200      # total simulation time [s]
WARMUP = 86400         # warmup period [s]
REPLICATIONS = 10

# Inter-arrival time of raw parts (chosen so that system is stably utilized)
INTER_ARRIVAL = 60.0   # seconds between arrivals

# Defect parameters
DEFECT_RATE = 0.089

# Machine data: avg processing time [s], availability [%], MTTR [s], idle kW, working kW
MACHINES = {
    "Conveyor":        dict(pt=6.0,   avail=100.00, mttr=1.0,  idle=0.00, work=0.00),
    "Handling":        dict(pt=25.0,  avail=97.79,  mttr=74.0, idle=0.50, work=0.74),
    "Loader":          dict(pt=12.0,  avail=90.49,  mttr=68.0, idle=0.25, work=0.72),
    "Press1":          dict(pt=175.0, avail=87.79,  mttr=73.0, idle=1.25, work=1.28),
    "Press2":          dict(pt=176.0, avail=87.69,  mttr=74.0, idle=1.25, work=1.27),
    "Quality":         dict(pt=41.0,  avail=85.87,  mttr=66.0, idle=0.58, work=0.84),
    "Washer":          dict(pt=14.0,  avail=80.89,  mttr=269.0,idle=4.28, work=35.24),
}

# Buffer data: capacity, process time [s]
BUFFERS = {
    "PostLoading":    dict(cap=2, pt=10),
    "PostConveyor":   dict(cap=2, pt=10),
    "PostWashing":    dict(cap=2, pt=10),
    "PrePress1":      dict(cap=3, pt=32),
    "PrePress2":      dict(cap=3, pt=32),
    "PostPresses":    dict(cap=3, pt=32),
}

# Util calendar: production stops from Fri 17:00–Sat 07:00 and Sat 17:00–Sun 07:00 every week
DAY = 24 * 3600
WEEK = 7 * DAY

def is_work_time(t):
    """Return True if time t (seconds) is in working period."""
    t_week = t % WEEK
    day = int(t_week // DAY)    # 0=Mon ... 4=Fri,5=Sat,6=Sun
    time_in_day = t_week % DAY

    if day < 4:  # Mon–Thu: always working
        return True
    if day == 4:  # Fri
        return time_in_day < 17 * 3600
    if day == 5:  # Sat
        return 7 * 3600 <= time_in_day < 17 * 3600
    if day == 6:  # Sun
        return time_in_day >= 7 * 3600
    return True

def time_to_next_work(env):
    """If in non-work period, return time until next work start, else 0."""
    t = env.now
    if is_work_time(t):
        return 0.0
    t_week = t % WEEK
    day = int(t_week // DAY)
    time_in_day = t_week % DAY

    if day == 4:  # Fri after 17:00 -> Sat 07:00
        target = 5 * DAY + 7 * 3600
    elif day == 5:  # Sat after 17:00 -> Sun 07:00
        target = 6 * DAY + 7 * 3600
    else:  # Sun before 07:00 -> Sun 07:00
        target = 6 * DAY + 7 * 3600

    delta = target - t_week
    if delta < 0:
        delta += WEEK
    return delta

class Machine:
    def __init__(self, env, name, data, stats):
        self.env = env
        self.name = name
        self.pt = data["pt"]
        self.avail = data["avail"] / 100.0
        self.mttr = data["mttr"]
        self.idle_power = data["idle"]
        self.work_power = data["work"]
        self.resource = simpy.Resource(env, capacity=1)
        self.stats = stats
        self.broken = False
        self.proc = env.process(self.breakdown_process())

    def breakdown_process(self):
        # Simple availability model: mean uptime derived from availability and MTTR
        if self.avail <= 0.0 or self.avail >= 1.0:
            return
        mean_uptime = self.mttr * self.avail / (1 - self.avail)
        while True:
            up = random.expovariate(1.0 / mean_uptime)
            yield self.env.timeout(up)
            self.broken = True
            down = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(down)
            self.broken = False

    def energy_log(self, power, duration):
        # kW * h -> kWh
        self.stats["energy"] += power * (duration / 3600.0)

    def process_part(self, part):
        # Wait for working time
        delta = time_to_next_work(self.env)
        if delta > 0:
            self.energy_log(self.idle_power, delta)
            yield self.env.timeout(delta)

        with self.resource.request() as req:
            yield req
            # Again ensure we are in work time
            delta = time_to_next_work(self.env)
            if delta > 0:
                self.energy_log(self.idle_power, delta)
                yield self.env.timeout(delta)

            remaining = self.pt
            while remaining > 0:
                if self.broken:
                    # wait until repaired
                    self.energy_log(self.idle_power, 1)
                    yield self.env.timeout(1)
                else:
                    step = min(1.0, remaining)
                    self.energy_log(self.work_power, step)
                    yield self.env.timeout(step)
                    remaining -= step

class Buffer:
    def __init__(self, env, name, data, stats):
        self.env = env
        self.name = name
        self.cap = data["cap"]
        self.pt = data["pt"]
        self.store = simpy.Store(env, capacity=self.cap)
        self.stats = stats

    def put(self, item):
        return self.store.put(item)

    def get(self):
        return self.store.get()

    def process_item(self, item):
        self.stats["energy"] += 0.0  # buffers assumed no energy consumption
        yield self.env.timeout(self.pt)

def part_generator(env, stats, buffers, machines):
    i = 0
    press_toggle = 0  # for even splitting between Press1 and Press2
    while True:
        i += 1
        part = {"id": i, "birth": env.now}
        stats["created"] += 1

        env.process(part_flow(env, part, stats, buffers, machines, press_toggle))
        press_toggle = 1 - press_toggle
        inter = random.expovariate(1.0 / INTER_ARRIVAL)
        yield env.timeout(inter)

def part_flow(env, part, stats, buffers, machines, press_toggle):
    # Loading robot
    yield env.process(machines["Loader"].process_part(part))
    # PostLoadingBuffer
    yield buffers["PostLoading"].put(part)
    part = yield buffers["PostLoading"].get()
    yield env.process(buffers["PostLoading"].process_item(part))

    # Conveyor
    yield env.process(machines["Conveyor"].process_part(part))
    # PostConveyorBuffer
    yield buffers["PostConveyor"].put(part)
    part = yield buffers["PostConveyor"].get()
    yield env.process(buffers["PostConveyor"].process_item(part))

    # Washing machine
    yield env.process(machines["Washer"].process_part(part))
    # PostWashingBuffer
    yield buffers["PostWashing"].put(part)
    part = yield buffers["PostWashing"].get()
    yield env.process(buffers["PostWashing"].process_item(part))

    # Handling cell
    yield env.process(machines["Handling"].process_part(part))

    # Split to Press1 or Press2 using buffers PrePress1 / PrePress2
    if press_toggle == 0:
        target_press = "Press1"
        buf_name = "PrePress1"
    else:
        target_press = "Press2"
        buf_name = "PrePress2"

    yield buffers[buf_name].put(part)
    part = yield buffers[buf_name].get()
    yield env.process(buffers[buf_name].process_item(part))

    # Press cell
    yield env.process(machines[target_press].process_part(part))

    # Merge after presses
    yield buffers["PostPresses"].put(part)
    part = yield buffers["PostPresses"].get()
    yield env.process(buffers["PostPresses"].process_item(part))

    # Quality station
    yield env.process(machines["Quality"].process_part(part))

    # Defect check
    if random.random() < DEFECT_RATE:
        stats["defects"] += 1
        # part goes to defect sink
        return

    # Good part finished
    stats["finished"] += 1
    if env.now > WARMUP:
        stats["finished_after_warmup"] += 1
        stats["sojourn_times"].append(env.now - part["birth"])

def run_replication(rep_id):
    random.seed(RANDOM_SEED + rep_id)
    env = simpy.Environment()

    # statistics
    stats = {
        "created": 0,
        "finished": 0,
        "finished_after_warmup": 0,
        "defects": 0,
        "sojourn_times": [],
        "energy": 0.0
    }

    # create machines
    machines = {name: Machine(env, name, data, stats) for name, data in MACHINES.items()}
    # create buffers
    buffers = {name: Buffer(env, name, data, stats) for name, data in BUFFERS.items()}

    env.process(part_generator(env, stats, buffers, machines))

    env.run(until=SIM_TIME)

    sim_hours_effective = (SIM_TIME - WARMUP) / 3600.0

    throughput_per_hour = stats["finished_after_warmup"] / sim_hours_effective if sim_hours_effective > 0 else 0.0
    mean_wip = 0.0
    if stats["sojourn_times"]:
        mean_throughput_rate = stats["finished_after_warmup"] / (SIM_TIME - WARMUP)
        mean_wip = mean_throughput_rate * statistics.mean(stats["sojourn_times"])
    energy_per_part = stats["energy"] / stats["finished_after_warmup"] if stats["finished_after_warmup"] > 0 else 0.0

    return throughput_per_hour, mean_wip, energy_per_part

throughputs = []
wips = []
energies = []

for r in range(REPLICATIONS):
    th, wip, en = run_replication(r)
    throughputs.append(th)
    wips.append(wip)
    energies.append(en)

mean_throughput = statistics.mean(throughputs) if throughputs else 0.0
mean_wip = statistics.mean(wips) if wips else 0.0
mean_energy = statistics.mean(energies) if energies else 0.0

print("=== Mean Overall KPIs over 10 runs ===")
print(f"Throughput = {mean_throughput:.3f} parts/hour")
print(f"WIP = {mean_wip:.3f} parts")
print(f"Mean Energy Consumption per Part = {mean_energy:.5f} kWh/part")