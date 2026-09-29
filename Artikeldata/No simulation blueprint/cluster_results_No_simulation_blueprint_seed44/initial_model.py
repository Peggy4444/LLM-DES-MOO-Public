import simpy
import random
import statistics
import math

RANDOM_SEED = 44

SIM_TIME = 691200        # total simulation time (s)
WARMUP = 86400           # warm-up period (s)
REPLICATIONS = 10

# Production calendar: stops from Fri 17:00- Sat 07:00 and Sat 17:00- Sun 07:00 each week
WEEK = 7 * 24 * 3600
DAY = 24 * 3600
HOUR = 3600

# Defect parameters (at Quality station)
DEFECT_RATE = 0.089

# Station data (mean process time in seconds)
STATIONS = {
    'Conveyor':      {'pt': 6.0},
    'Handling':      {'pt': 25.0},
    'Loading':       {'pt': 12.0},  # given but not in flow, included for completeness
    'Press1':        {'pt': 175.0},
    'Press2':        {'pt': 176.0},
    'Quality':       {'pt': 41.0},
    'Washing':       {'pt': 14.0},
}

# Buffer data (capacity and extra process time in seconds)
BUFFERS = {
    'PostLoading':        {'cap': 2, 'pt': 10},
    'PostConveyor':       {'cap': 2, 'pt': 10},
    'PostWashing':        {'cap': 2, 'pt': 10},
    'PrePress1':          {'cap': 3, 'pt': 32},
    'PrePress2':          {'cap': 3, 'pt': 32},
    'PostPress1and2':     {'cap': 3, 'pt': 32},
}

# Energy parameters (kW)
ENERGY = {
    'Handling': {'idle': 0.50, 'work': 0.74},
    'Loading':  {'idle': 0.25, 'work': 0.72},
    'Press1':   {'idle': 1.25, 'work': 1.28},
    'Press2':   {'idle': 1.25, 'work': 1.27},
    'Quality':  {'idle': 0.58, 'work': 0.84},
    'Washing':  {'idle': 4.28, 'work': 35.24},
    'Conveyor': {'idle': 0.0,  'work': 0.0},
}

# Downtime parameters derived from Availability and MTTR (not used in detail, but structure kept)
AVAILABILITY = {
    'Conveyor':  1.00,
    'Handling':  0.9779,
    'Loading':   0.9049,
    'Press1':    0.8779,
    'Press2':    0.8769,
    'Quality':   0.8587,
    'Washing':   0.8089
}
MTTR = {
    'Conveyor': 1.0,
    'Handling': 74.0,
    'Loading':  68.0,
    'Press1':   73.0,
    'Press2':   74.0,
    'Quality':  66.0,
    'Washing':  269.0
}

# Interarrival time of raw parts (not specified). Choose a rate to keep system busy.
INTERARRIVAL = 60.0  # seconds between raw arrivals (can be adjusted)


def is_working_time(t):
    """Return True if time t (seconds from 0) is within allowed production time."""
    w = t % WEEK
    day = int(w // DAY)
    t_day = w % DAY

    # Monday(0) - Thursday(3): always working
    if day in [0, 1, 2, 3]:
        return True

    # Friday (4): work until 17:00
    if day == 4:
        return t_day < 17 * HOUR

    # Saturday (5): work only after 07:00 until 17:00
    if day == 5:
        return 7 * HOUR <= t_day < 17 * HOUR

    # Sunday (6): work only after 07:00 until 24:00
    if day == 6:
        return t_day >= 7 * HOUR

    return True


def wait_for_working_time(env):
    """If in non-working time, yield until next working period."""
    while not is_working_time(env.now):
        # jump to next hour boundary to speed up
        next_check = math.floor(env.now / HOUR + 1) * HOUR
        yield env.timeout(max(1.0, next_check - env.now))


class EnergyMeter:
    """Tracks energy consumption per station (kWh)."""
    def __init__(self, env):
        self.env = env
        self.energy = {name: 0.0 for name in ENERGY}
        self.last_ts = {name: 0.0 for name in ENERGY}
        self.busy = {name: False for name in ENERGY}

    def set_state(self, station, busy):
        """Change station state and accumulate energy to now."""
        now = self.env.now
        last = self.last_ts[station]
        dt = (now - last) / 3600.0  # hours
        if dt > 0:
            power = ENERGY[station]['work'] if self.busy[station] else ENERGY[station]['idle']
            self.energy[station] += power * dt
        self.last_ts[station] = now
        self.busy[station] = busy

    def finalize(self):
        """Call at end to flush final energy accounting."""
        for st in ENERGY:
            self.set_state(st, self.busy[st])


class ProductionSystem:
    def __init__(self, env):
        self.env = env

        # Resources for machines
        self.conveyor = simpy.Resource(env, capacity=1)
        self.handling = simpy.Resource(env, capacity=1)
        self.loading = simpy.Resource(env, capacity=1)
        self.press1 = simpy.Resource(env, capacity=1)
        self.press2 = simpy.Resource(env, capacity=1)
        self.quality = simpy.Resource(env, capacity=1)
        self.washing = simpy.Resource(env, capacity=1)

        # Buffers as Stores with capacities
        self.post_loading = simpy.Store(env, capacity=BUFFERS['PostLoading']['cap'])
        self.post_conveyor = simpy.Store(env, capacity=BUFFERS['PostConveyor']['cap'])
        self.post_washing = simpy.Store(env, capacity=BUFFERS['PostWashing']['cap'])
        self.pre_press1 = simpy.Store(env, capacity=BUFFERS['PrePress1']['cap'])
        self.pre_press2 = simpy.Store(env, capacity=BUFFERS['PrePress2']['cap'])
        self.post_press12 = simpy.Store(env, capacity=BUFFERS['PostPress1and2']['cap'])

        # Splitter control for even distribution
        self.press_toggle = 0  # 0 -> Press1, 1 -> Press2

        # KPIs
        self.total_completed = 0
        self.total_defects = 0
        self.total_started = 0
        self.time_in_system = []
        self.energy_meter = EnergyMeter(env)

        # WIP tracking
        self.wip = 0
        self.area_wip = 0.0
        self.last_wip_ts = env.now

    def inc_wip(self, delta):
        now = self.env.now
        dt = now - self.last_wip_ts
        if dt > 0:
            self.area_wip += self.wip * dt
        self.wip += delta
        self.last_wip_ts = now

    def part_process(self, name):
        """Main part flow from raw input to output/defect sink."""
        self.total_started += 1
        self.inc_wip(+1)
        start_time = self.env.now

        # Loading robot -> PostLoadingBuffer
        with self.loading.request() as req:
            yield req
            yield from wait_for_working_time(self.env)
            self.energy_meter.set_state('Loading', True)
            yield self.env.timeout(STATIONS['Loading']['pt'])
            self.energy_meter.set_state('Loading', False)
        # PostLoading buffer time
        yield self.post_loading.put(name)
        yield from wait_for_working_time(self.env)
        yield self.env.timeout(BUFFERS['PostLoading']['pt'])
        _ = yield self.post_loading.get()

        # Conveyor belt -> PostConveyorBuffer
        with self.conveyor.request() as req:
            yield req
            yield from wait_for_working_time(self.env)
            self.energy_meter.set_state('Conveyor', True)
            yield self.env.timeout(STATIONS['Conveyor']['pt'])
            self.energy_meter.set_state('Conveyor', False)
        yield self.post_conveyor.put(name)
        yield from wait_for_working_time(self.env)
        yield self.env.timeout(BUFFERS['PostConveyor']['pt'])
        _ = yield self.post_conveyor.get()

        # Washing machine -> PostWashingBuffer
        with self.washing.request() as req:
            yield req
            yield from wait_for_working_time(self.env)
            self.energy_meter.set_state('Washing', True)
            yield self.env.timeout(STATIONS['Washing']['pt'])
            self.energy_meter.set_state('Washing', False)
        yield self.post_washing.put(name)
        yield from wait_for_working_time(self.env)
        yield self.env.timeout(BUFFERS['PostWashing']['pt'])
        _ = yield self.post_washing.get()

        # Handling cell -> split to PrePress1 / PrePress2
        with self.handling.request() as req:
            yield req
            yield from wait_for_working_time(self.env)
            self.energy_meter.set_state('Handling', True)
            yield self.env.timeout(STATIONS['Handling']['pt'])
            self.energy_meter.set_state('Handling', False)

        # Split evenly
        if self.press_toggle == 0:
            target_buffer = self.pre_press1
            self.press_toggle = 1
        else:
            target_buffer = self.pre_press2
            self.press_toggle = 0

        yield target_buffer.put(name)
        # PrePress buffer process time
        yield from wait_for_working_time(self.env)
        if target_buffer is self.pre_press1:
            yield self.env.timeout(BUFFERS['PrePress1']['pt'])
            _ = yield self.pre_press1.get()
            # Press1
            with self.press1.request() as req:
                yield req
                yield from wait_for_working_time(self.env)
                self.energy_meter.set_state('Press1', True)
                yield self.env.timeout(STATIONS['Press1']['pt'])
                self.energy_meter.set_state('Press1', False)
        else:
            yield self.env.timeout(BUFFERS['PrePress2']['pt'])
            _ = yield self.pre_press2.get()
            # Press2
            with self.press2.request() as req:
                yield req
                yield from wait_for_working_time(self.env)
                self.energy_meter.set_state('Press2', True)
                yield self.env.timeout(STATIONS['Press2']['pt'])
                self.energy_meter.set_state('Press2', False)

        # After presses -> PostPress1&2Buffer
        yield self.post_press12.put(name)
        yield from wait_for_working_time(self.env)
        yield self.env.timeout(BUFFERS['PostPress1and2']['pt'])
        _ = yield self.post_press12.get()

        # Quality station cell -> defect or good part
        with self.quality.request() as req:
            yield req
            yield from wait_for_working_time(self.env)
            self.energy_meter.set_state('Quality', True)
            yield self.env.timeout(STATIONS['Quality']['pt'])
            self.energy_meter.set_state('Quality', False)

        # Defect decision
        if random.random() < DEFECT_RATE:
            # defect sink
            self.total_defects += 1
        else:
            self.total_completed += 1
            if self.env.now > WARMUP:
                self.time_in_system.append(self.env.now - start_time)

        self.inc_wip(-1)

    def source(self, arrival_rate):
        i = 0
        while True:
            yield self.env.timeout(random.expovariate(1.0 / arrival_rate))
            if self.env.now > SIM_TIME:
                break
            i += 1
            self.env.process(self.part_process(f"Part_{i}"))


def run_replication(rep):
    random.seed(RANDOM_SEED + rep)
    env = simpy.Environment()
    system = ProductionSystem(env)
    env.process(system.source(INTERARRIVAL))
    env.run(until=SIM_TIME)
    system.energy_meter.finalize()

    sim_hours_effective = max(1e-9, (SIM_TIME - WARMUP) / 3600.0)

    throughput_per_hour = system.total_completed / sim_hours_effective
    mean_wip = system.area_wip / max(1e-9, (SIM_TIME - 0.0))
    total_energy = sum(system.energy_meter.energy.values())
    energy_per_part = total_energy / max(1, system.total_completed)

    return throughput_per_hour, mean_wip, energy_per_part


throughputs = []
wips = []
energies = []

for r in range(REPLICATIONS):
    th, w, e = run_replication(r)
    throughputs.append(th)
    wips.append(w)
    energies.append(e)

mean_throughput = statistics.mean(throughputs)
mean_wip = statistics.mean(wips)
mean_energy = statistics.mean(energies)

print("=== Mean Overall KPIs over 10 runs ===")
print(f"Throughput = {mean_throughput:.3f} parts/hour")
print(f"WIP = {mean_wip:.3f} parts")
print(f"Mean Energy Consumption per Part = {mean_energy:.5f} kWh/part")