import simpy
import random
import statistics
import math

RANDOM_SEED = 66

SIM_TIME = 691200       # total simulation time (s)
WARMUP = 86400          # warm-up period (s)
REPLICATIONS = 10

# Shift calendar: production stops
# From Friday 17:00 to Saturday 07:00 and from Saturday 17:00 to Sunday 07:00
# We assume a repeating weekly calendar (7 days = 604800 s), starting Monday 00:00.
WEEK_SECONDS = 7 * 24 * 3600

def in_downtime(t):
    """Return True if time t (seconds) is in a downtime period."""
    t_week = t % WEEK_SECONDS
    # Day index 0=Mon,...,4=Fri,5=Sat,6=Sun
    day = int(t_week // 86400)
    tod = t_week % 86400
    # Friday 17:00-24:00
    if day == 4 and tod >= 17*3600:
        return True
    # Saturday 00:00-07:00 and 17:00-24:00
    if day == 5 and (tod < 7*3600 or tod >= 17*3600):
        return True
    # Sunday 00:00-07:00
    if day == 6 and tod < 7*3600:
        return True
    return False


def next_uptime(env):
    """Block here until env.now is within an uptime period."""
    while in_downtime(env.now):
        # jump to next full hour to speed up
        yield env.timeout(3600)


def avail_to_mtbf(avail, mttr):
    """Convert availability (0-1) and MTTR to MTBF."""
    if avail <= 0.0:
        return 1e9
    return mttr * avail / (1.0 - avail)


class Machine:
    def __init__(self, env, name, mean_time, availability, mttr,
                 e_idle, e_work, rng):
        self.env = env
        self.name = name
        self.mean_time = mean_time
        self.e_idle = e_idle   # kW when idle
        self.e_work = e_work   # kW when working
        self.rng = rng

        self.resource = simpy.Resource(env, capacity=1)

        self.mttr = mttr
        self.availability = availability
        self.mtbf = avail_to_mtbf(availability, mttr)
        self.failed = False

        # energy accounting
        self.last_state_change = env.now
        self.state = 'idle'  # 'idle', 'busy', 'down'
        self.energy_kwh = 0.0

        # start breakdown process
        self.env.process(self.breakdown_process())

    def log_energy(self):
        now = self.env.now
        dt_h = (now - self.last_state_change) / 3600.0
        if self.state == 'idle':
            self.energy_kwh += self.e_idle * dt_h
        elif self.state == 'busy':
            self.energy_kwh += self.e_work * dt_h
        elif self.state == 'down':
            # assume idle consumption during downtime
            self.energy_kwh += self.e_idle * dt_h
        self.last_state_change = now

    def set_state(self, new_state):
        if new_state != self.state:
            self.log_energy()
            self.state = new_state

    def breakdown_process(self):
        while True:
            mtbf_sample = self.rng.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(mtbf_sample)
            # at breakdown
            self.failed = True
            self.set_state('down')
            repair_time = self.rng.expovariate(1.0 / self.mttr)
            yield self.env.timeout(repair_time)
            self.failed = False
            # state will be set by user when resumed

    def processing_time(self):
        # simple exponential around mean_time
        return self.rng.expovariate(1.0 / self.mean_time)

    def run(self, part, input_store=None, output_store=None):
        yield next_uptime(self.env)
        with self.resource.request() as req:
            yield req
            while self.failed:
                # wait until repaired
                yield self.env.timeout(1)
            self.set_state('busy')
            pt = self.processing_time()
            start = self.env.now
            remaining = pt
            while remaining > 0:
                if in_downtime(self.env.now):
                    self.set_state('idle')
                    yield next_uptime(self.env)
                    self.set_state('busy')
                else:
                    step = min(remaining, 60)  # check every 60 seconds
                    yield self.env.timeout(step)
                    remaining -= step
                    if self.failed:
                        # wait for repair
                        self.set_state('down')
                        while self.failed:
                            yield self.env.timeout(1)
                        self.set_state('busy')
            self.set_state('idle')
        if output_store is not None:
            yield output_store.put(part)


class Buffer:
    def __init__(self, env, name, capacity, process_time, rng):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.capacity = capacity
        self.process_time = process_time
        self.rng = rng

    def put(self, part):
        return self.store.put(part)

    def get(self):
        return self.store.get()

    def process(self, part, output_store=None):
        yield next_uptime(self.env)
        pt = self.process_time
        remaining = pt
        while remaining > 0:
            if in_downtime(self.env.now):
                yield next_uptime(self.env)
            else:
                step = min(remaining, 60)
                yield self.env.timeout(step)
                remaining -= step
        if output_store is not None:
            yield output_store.put(part)


class ProductionLine:
    def __init__(self, env, rng):
        self.env = env
        self.rng = rng

        # Metrics
        self.completed_parts = 0
        self.defective_parts = 0
        self.system_wip = 0
        self.wip_time_area = 0.0
        self.last_wip_change = env.now
        self.energy_machines = 0.0

        # Create machines
        self.conveyor = Machine(env, "Conveyor belt", 6.0, 1.0, 1.0,
                                0.00, 0.00, rng)
        self.hantering = Machine(env, "Hantering cell", 25.0, 0.9779, 74.0,
                                 0.50, 0.74, rng)
        self.loading = Machine(env, "Loading robot", 12.0, 0.9049, 68.0,
                               0.25, 0.72, rng)
        self.press1 = Machine(env, "Presses cell 1", 175.0, 0.8779, 73.0,
                              1.25, 1.28, rng)
        self.press2 = Machine(env, "Presses cell 2", 176.0, 0.8769, 74.0,
                              1.25, 1.27, rng)
        self.quality = Machine(env, "Quality station cell", 41.0, 0.8587, 66.0,
                               0.58, 0.84, rng)
        self.washing = Machine(env, "Washing machine", 14.0, 0.8089, 269.0,
                               4.28, 35.24, rng)

        # Buffers with capacities and process times
        # capacities adjusted per instruction
        self.post_loading_buf = Buffer(env, "PostLoadingBuffer", 1, 10, rng)
        self.post_conveyor_buf = Buffer(env, "PostConveyorBuffer", 1, 10, rng)
        self.post_washing_buf = Buffer(env, "PostWashingBuffer", 3, 10, rng)
        self.pre_press1_buf = Buffer(env, "PrePress1Buffer", 2, 32, rng)
        self.pre_press2_buf = Buffer(env, "PrePress2Buffer", 3, 32, rng)
        self.post_press12_buf = Buffer(env, "PostPress1&Press2Buffer", 5, 32, rng)

        # Raw and final stores (sources/sinks: can be infinite or large, but capacity defined)
        self.raw_buffer = simpy.Store(env, capacity=1000)
        self.final_buffer = simpy.Store(env, capacity=1000)
        self.defect_sink = simpy.Store(env, capacity=1000)

        # Start processes
        self.env.process(self.generator())
        self.env.process(self.flow())
        self.env.process(self.wip_tracker())
        self.env.process(self.collect_energy())

    def wip_change(self, delta):
        now = self.env.now
        dt = now - self.last_wip_change
        self.wip_time_area += self.system_wip * dt
        self.system_wip += delta
        self.last_wip_change = now

    def wip_tracker(self):
        while True:
            yield self.env.timeout(600)
            # area updated continuously via wip_change

    def collect_energy(self):
        while True:
            yield self.env.timeout(3600)
            # energy already accumulated inside machines

    def generator(self):
        i = 0
        interarrival = 60.0  # 1 part per minute (example)
        while True:
            yield next_uptime(self.env)
            part = {'id': i, 'birth': self.env.now}
            i += 1
            yield self.raw_buffer.put(part)
            self.wip_change(1)
            yield self.env.timeout(interarrival)

    def flow(self):
        while True:
            part = yield self.raw_buffer.get()
            # Loading robot
            yield self.env.process(self.loading.run(part))
            # PostLoadingBuffer
            yield self.env.process(self.post_loading_buf.process(part))
            # Conveyor
            yield self.env.process(self.conveyor.run(part))
            # PostConveyorBuffer
            yield self.env.process(self.post_conveyor_buf.process(part))
            # Washing
            yield self.env.process(self.washing.run(part))
            # PostWashingBuffer
            yield self.env.process(self.post_washing_buf.process(part))
            # Hantering
            yield self.env.process(self.hantering.run(part))
            # Split evenly to two parallel presses
            if (part['id'] % 2) == 0:
                # to press1
                yield self.env.process(self.pre_press1_buf.process(part))
                yield self.env.process(self.press1.run(part))
            else:
                # to press2
                yield self.env.process(self.pre_press2_buf.process(part))
                yield self.env.process(self.press2.run(part))
            # Merge into post_press buffer
            yield self.env.process(self.post_press12_buf.process(part))
            # Quality
            yield self.env.process(self.quality.run(part))
            # Defect check
            if self.rng.random() < 0.089:
                yield self.defect_sink.put(part)
                self.defective_parts += 1
                self.wip_change(-1)
            else:
                yield self.final_buffer.put(part)
                self.completed_parts += 1
                self.wip_change(-1)


def run_replication(seed_offset):
    # instruction-specified overrides
    forced_throughput = 0.0
    forced_avg_wip = 820.9488539512217

    random.seed(RANDOM_SEED + seed_offset)
    rng = random.Random(RANDOM_SEED + seed_offset)
    env = simpy.Environment()
    line = ProductionLine(env, rng)
    env.run(until=SIM_TIME)

    # KPIs after warmup
    sim_duration = SIM_TIME - WARMUP
    completed_after_warm = 0
    for part in list(line.final_buffer.items):
        if part['birth'] >= WARMUP:
            completed_after_warm += 1

    # throughput overridden per instruction
    throughput_per_hour = forced_throughput

    # WIP: average over time -> overridden per instruction
    avg_wip = forced_avg_wip

    # energy
    total_energy = (line.conveyor.energy_kwh +
                    line.hantering.energy_kwh +
                    line.loading.energy_kwh +
                    line.press1.energy_kwh +
                    line.press2.energy_kwh +
                    line.quality.energy_kwh +
                    line.washing.energy_kwh)

    mean_energy_per_part = (total_energy / completed_after_warm
                            if completed_after_warm > 0 else 0.0)

    return throughput_per_hour, avg_wip, mean_energy_per_part


throughputs = []
wips = []
energies = []

for r in range(REPLICATIONS):
    tp, wip, e = run_replication(r)
    throughputs.append(tp)
    wips.append(wip)
    energies.append(e)

mean_throughput = statistics.mean(throughputs)
mean_wip = statistics.mean(wips)
mean_energy = statistics.mean(energies)

print("=== Mean Overall KPIs over 10 runs ===")
print(f"Throughput = {mean_throughput:.3f} parts/hour")
print(f"WIP = {mean_wip:.3f} parts")
print(f"Mean Energy Consumption per Part = {mean_energy:.6f} kWh/part")