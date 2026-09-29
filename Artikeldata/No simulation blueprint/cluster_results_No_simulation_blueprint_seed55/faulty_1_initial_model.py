import simpy
import random
import statistics
import math

RANDOM_SEED = 55
SIM_TIME = 691200        # total simulation time (s)
WARMUP = 86400           # warm-up (s)
REPS = 10

# Shift calendar: production stops
# From Friday 17:00 to Saturday 07:00 and Saturday 17:00 to Sunday 07:00
SEC_PER_DAY = 24 * 3600

def is_working_time(t):
    """Return True if time t (seconds) is inside working period."""
    day = int(t // SEC_PER_DAY)  # 0=Mon,1=Tue,...,4=Fri,5=Sat,6=Sun,...
    sec_in_day = t % SEC_PER_DAY
    # No production on Sunday
    if day % 7 == 6:
        return False
    # Friday restrictions
    if day % 7 == 4:  # Friday
        if sec_in_day >= 17*3600:
            return False
    # Saturday restrictions
    if day % 7 == 5:  # Saturday
        if sec_in_day < 7*3600 or sec_in_day >= 17*3600:
            return False
    # Other days: full time
    return True

def wait_until_working(env):
    """Suspend process until next working time."""
    while not is_working_time(env.now):
        # advance in small steps until working time starts
        # (could be optimized by jumping to boundary times)
        yield env.timeout(300)  # 5 min steps


class Machine:
    def __init__(self, env, name, mean_proc, availability, mttr,
                 idle_energy, work_energy):
        self.env = env
        self.name = name
        self.mean_proc = mean_proc
        self.availability = availability / 100.0
        self.mttr = mttr
        self.idle_energy = idle_energy
        self.work_energy = work_energy
        self.resource = simpy.Resource(env, capacity=1)
        # failure parameters: availability = MTTF / (MTTF + MTTR) -> MTTF
        if self.availability < 1.0:
            self.mttf = (self.availability * self.mttr) / (1 - self.availability)
        else:
            self.mttf = None

    def process_part(self, part, stats):
        """Full processing with possible breakdowns and shift stops."""
        with self.resource.request() as req:
            yield req
            # waiting to be allowed to work by calendar
            if not is_working_time(self.env.now):
                idle_start = self.env.now
                yield from wait_until_working(self.env)
                idle_dur = self.env.now - idle_start
                stats['energy_idle'] += idle_dur * self.idle_energy

            remaining = random.expovariate(1.0 / self.mean_proc)

            while remaining > 0:
                start = self.env.now
                # time to next failure (if any)
                if self.mttf is not None:
                    ttf = random.expovariate(1.0 / self.mttf)
                else:
                    ttf = float('inf')

                # time until end of current working window
                # approximate by small step checking
                step = min(remaining, ttf, 300.0)
                # but we must be careful to obey calendar
                while step > 0:
                    if not is_working_time(self.env.now):
                        idle_start = self.env.now
                        yield from wait_until_working(self.env)
                        idle_dur = self.env.now - idle_start
                        stats['energy_idle'] += idle_dur * self.idle_energy
                    eff_step = min(step, remaining, ttf)
                    yield self.env.timeout(eff_step)
                    remaining -= eff_step
                    step -= eff_step
                    work_dur = eff_step
                    stats['energy_work'] += work_dur * self.work_energy
                    # failure?
                    if eff_step == ttf and self.mttf is not None:
                        # breakdown
                        repair_time = random.expovariate(1.0 / self.mttr)
                        idle_start = self.env.now
                        # during repair, ignore calendar (machine down anyway)
                        yield self.env.timeout(repair_time)
                        idle_dur = self.env.now - idle_start
                        stats['energy_idle'] += idle_dur * self.idle_energy
                        break  # restart remaining loop with new ttf
                # loop continues until remaining <= 0


class BufferWithDelay:
    """Finite buffer plus extra delay (process time) before leaving."""
    def __init__(self, env, name, capacity, delay):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.delay = delay

    def put(self, item):
        return self.store.put(item)

    def get(self):
        item = yield self.store.get()
        # delay is subject to calendar
        dur = self.delay
        while dur > 0:
            if not is_working_time(self.env.now):
                yield from wait_until_working(self.env)
            step = min(dur, 300.0)
            yield self.env.timeout(step)
            dur -= step
        return item


def part_process(env, name, machines, buffers, stats, defect_rate):
    """
    Single part routing:
    Loading -> PostLoadingBuffer -> Conveyor -> PostConveyorBuffer -> Washing
    -> PostWashingBuffer -> Hantering -> split to Press1 or Press2 (evenly)
    -> PrePressXBuffer -> PressX -> PostPress1&2Buffer -> Quality -> defect/success
    """
    stats['wip'] += 1

    # Load -> buffer
    yield env.process(machines['Loading robot'].process_part(name, stats))
    yield buffers['PostLoadingBuffer'].put(name)

    # Conveyor
    part = yield buffers['PostLoadingBuffer'].get()
    yield env.process(machines['Conveyor belt'].process_part(part, stats))
    yield buffers['PostConveyorBuffer'].put(part)

    # Washing
    part = yield buffers['PostConveyorBuffer'].get()
    yield env.process(machines['Washing machine'].process_part(part, stats))
    yield buffers['PostWashingBuffer'].put(part)

    # Hantering
    part = yield buffers['PostWashingBuffer'].get()
    yield env.process(machines['Hantering cell'].process_part(part, stats))

    # Decide press path by simple alternating (even split)
    if stats['press_counter'] % 2 == 0:
        press_name = 'Presses cell 1'
        prebuf = 'PrePress1Buffer'
    else:
        press_name = 'Presses cell 2'
        prebuf = 'PrePress2Buffer'
    stats['press_counter'] += 1

    # Pre-press buffer then press
    yield buffers[prebuf].put(part)
    part = yield buffers[prebuf].get()
    yield env.process(machines[press_name].process_part(part, stats))

    # Common buffer after presses
    yield buffers['PostPress1&Press2Buffer'].put(part)
    part = yield buffers['PostPress1&Press2Buffer'].get()

    # Quality
    yield env.process(machines['Quality station cell'].process_part(part, stats))

    # Defect or good
    if random.random() < defect_rate:
        stats['defects'] += 1
    else:
        stats['good'] += 1

    stats['wip'] -= 1


def source(env, machines, buffers, stats, defect_rate, interarrival):
    i = 0
    while True:
        if not is_working_time(env.now):
            yield from wait_until_working(env)
        i += 1
        env.process(part_process(env, f"Part_{i}", machines, buffers, stats, defect_rate))
        ia = random.expovariate(1.0 / interarrival)
        yield env.timeout(ia)


def run_replication(rep, results):
    random.seed(RANDOM_SEED + rep)
    env = simpy.Environment()

    stats = {
        'good': 0,
        'defects': 0,
        'wip': 0,
        'energy_idle': 0.0,
        'energy_work': 0.0,
        'press_counter': 0,
        'wip_time_area': 0.0,
        'last_wip_change': 0.0
    }

    def wip_monitor(env, stats):
        prev_time = env.now
        prev_wip = stats['wip']
        while True:
            yield env.timeout(60)  # sample every minute
            now = env.now
            stats['wip_time_area'] += prev_wip * (now - prev_time)
            prev_time = now
            prev_wip = stats['wip']

    # Stations
    machines = {
        'Conveyor belt': Machine(env, 'Conveyor belt', 6.0, 100.0, 1.0, 0.0, 0.0),
        'Hantering cell': Machine(env, 'Hantering cell', 25.0, 97.79, 74.0, 0.50, 0.74),
        'Loading robot': Machine(env, 'Loading robot', 12.0, 90.49, 68.0, 0.25, 0.72),
        'Presses cell 1': Machine(env, 'Presses cell 1', 175.0, 87.79, 73.0, 1.25, 1.28),
        'Presses cell 2': Machine(env, 'Presses cell 2', 176.0, 87.69, 74.0, 1.25, 1.27),
        'Quality station cell': Machine(env, 'Quality station cell', 41.0, 85.87, 66.0, 0.58, 0.84),
        'Washing machine': Machine(env, 'Washing machine', 14.0, 80.89, 269.0, 4.28, 35.24),
    }

    # Buffers (all with specified capacities)
    buffers = {
        'PostLoadingBuffer': BufferWithDelay(env, 'PostLoadingBuffer', capacity=2, delay=10),
        'PostConveyorBuffer': BufferWithDelay(env, 'PostConveyorBuffer', capacity=2, delay=10),
        'PostWashingBuffer': BufferWithDelay(env, 'PostWashingBuffer', capacity=2, delay=10),
        'PrePress1Buffer': BufferWithDelay(env, 'PrePress1Buffer', capacity=3, delay=32),
        'PrePress2Buffer': BufferWithDelay(env, 'PrePress2Buffer', capacity=3, delay=32),
        'PostPress1&Press2Buffer': BufferWithDelay(env, 'PostPress1&Press2Buffer', capacity=3, delay=32),
    }

    # Start processes
    env.process(source(env, machines, buffers, stats, defect_rate=0.089, interarrival=30.0))
    env.process(wip_monitor(env, stats))

    env.run(until=SIM_TIME)

    # KPIs post warm-up
    effective_time = SIM_TIME - WARMUP
    # Throughput: good parts after warm-up approximated (ignoring exact warm-up cut per part)
    throughput_per_hour = stats['good'] / (effective_time / 3600.0)
    avg_wip = stats['wip_time_area'] / SIM_TIME
    total_parts = max(1, stats['good'])  # avoid div0
    energy_per_part = (stats['energy_idle'] + stats['energy_work']) / total_parts / 3600.0

    results['throughput'].append(throughput_per_hour)
    results['wip'].append(avg_wip)
    results['energy_per_part'].append(energy_per_part)


if __name__ == "__main__":
    results = {
        'throughput': [],
        'wip': [],
        'energy_per_part': []
    }

    for r in range(REPS):
        run_replication(r, results)

    mean_throughput = statistics.mean(results['throughput'])
    mean_wip = statistics.mean(results['wip'])
    mean_energy = statistics.mean(results['energy_per_part'])

    print("=== Mean Overall KPIs over 10 runs ===")
    print(f"Throughput = {mean_throughput:.3f} parts/hour")
    print(f"WIP = {mean_wip:.3f} parts")
    print(f"Mean Energy Consumption per Part = {mean_energy:.6f} kWh/part")