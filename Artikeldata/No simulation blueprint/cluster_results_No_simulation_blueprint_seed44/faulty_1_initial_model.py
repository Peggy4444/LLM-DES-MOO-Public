import simpy
import random
import statistics
import math

RANDOM_SEED = 44
SIM_TIME = 691200          # total simulation time (s)
WARMUP = 86400             # warm-up period (s)
REPLICATIONS = 10

# ---------------------------------------------------------------------
# Calendar: production stops
# Friday 17:00 till Saturday 07:00 and Saturday 17:00 till Sunday 07:00
# We assume a weekly repeating pattern, starting at Monday 00:00 = t=0
# 0..604800 is one week
# Work is allowed except:
#   Fri 17:00 (day 4*86400+61200) to Sat 07:00 (day5*86400+25200)
#   Sat 17:00 (day5*86400+61200) to Sun 07:00 (day6*86400+25200)
# ---------------------------------------------------------------------

WEEK = 7 * 24 * 3600
DAY = 24 * 3600

FRI_17 = 4 * DAY + 17 * 3600
SAT_07 = 5 * DAY + 7 * 3600
SAT_17 = 5 * DAY + 17 * 3600
SUN_07 = 6 * DAY + 7 * 3600


def is_work_time(t):
    """Return True if time t is in allowed production time."""
    w = t % WEEK
    if (FRI_17 <= w < SAT_07) or (SAT_17 <= w < SUN_07):
        return False
    return True


def wait_until_work_time(env):
    """If now is in a stop interval, wait until next work start."""
    while not is_work_time(env.now):
        w = env.now % WEEK
        if FRI_17 <= w < SAT_07:
            delta = SAT_07 - w
        elif SAT_17 <= w < SUN_07:
            delta = SUN_07 - w
        else:
            delta = 0
        if delta > 0:
            yield env.timeout(delta)
        else:
            break


def work_time_timeout(env, duration):
    """
    Like env.timeout(duration) but skips over non-working intervals.
    Returns when a total of 'duration' work seconds have elapsed.
    """
    remaining = duration
    while remaining > 0:
        # If in a stop interval, skip until work time
        if not is_work_time(env.now):
            yield from wait_until_work_time(env)
            continue

        # At work time; find time until next stop (if any)
        w = env.now % WEEK
        if w < FRI_17:
            next_stop = FRI_17
        elif FRI_17 <= w < SAT_07:
            next_stop = w  # already in stop, but we would have caught above
        elif SAT_07 <= w < SAT_17:
            next_stop = SAT_17
        elif SAT_17 <= w < SUN_07:
            next_stop = w
        else:
            next_stop = WEEK + FRI_17  # next week

        time_to_stop = next_stop - w
        if remaining <= time_to_stop:
            yield env.timeout(remaining)
            remaining = 0
        else:
            yield env.timeout(time_to_stop)
            remaining -= time_to_stop
            # now we're at a stop, will loop and skip it


# ---------------------------------------------------------------------
# Machine / Buffer definitions
# ---------------------------------------------------------------------

class Machine:
    def __init__(self, env, name, avg_time, availability, mttr,
                 energy_idle, energy_work, energy_recorder):
        self.env = env
        self.name = name
        self.avg_time = avg_time
        self.mttr = mttr
        self.energy_idle = energy_idle
        self.energy_work = energy_work
        self.energy_recorder = energy_recorder

        # capacity 1 machine
        self.resource = simpy.Resource(env, capacity=1)

        # Availability -> exponential MTBF from availability and MTTR:
        # A = MTBF / (MTBF + MTTR) => MTBF = A*MTTR/(1-A)
        if availability < 1.0:
            self.mtbf = (availability * mttr) / (1.0 - availability)
        else:
            self.mtbf = None

        self.broken = False
        if self.mtbf:
            env.process(self.breakdown_process())

    def breakdown_process(self):
        while True:
            ttf = random.expovariate(1.0 / self.mtbf)
            yield work_time_timeout(self.env, ttf)
            self.broken = True
            # machine under repair: we just wait repair time; any
            # current process is assumed to be extended in real time
            ttr = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(ttr)
            self.broken = False

    def process_part(self, part, mean_time=None):
        """Process one part; mean_time overrides machine avg_time."""
        if mean_time is None:
            mean_time = self.avg_time

        # wait for resource
        with self.resource.request() as req:
            yield req
            # processing time
            p_time = random.expovariate(1.0 / mean_time)

            start = self.env.now
            # energy idle not counted while actively working,
            # we account only working power * work_time
            yield from work_time_timeout(self.env, p_time)
            end = self.env.now
            self.energy_recorder.add_work(self.energy_work * (end - start))


class Buffer:
    """FIFO buffer with capacity and optional processing time (like a mini station)."""
    def __init__(self, env, name, capacity, process_time, energy_recorder):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.process_time = process_time
        self.energy_recorder = energy_recorder

    def put(self, part):
        return self.store.put(part)

    def get(self):
        return self.store.get()

    def process_part(self, part):
        # buffer processing time (e.g., delay)
        p_time = random.expovariate(1.0 / self.process_time) if self.process_time > 0 else 0
        start = self.env.now
        yield from work_time_timeout(self.env, p_time)
        end = self.env.now
        # assume no special energy for buffers here
        _ = start, end


# ---------------------------------------------------------------------
# Energy recorder
# ---------------------------------------------------------------------

class EnergyRecorder:
    def __init__(self, env):
        self.env = env
        self.work_energy = 0.0  # kWh, if power in kW and time in hours

    def add_work(self, power_kw_times_seconds):
        # given power[kW] * seconds, convert to kWh: (kW * s) / 3600
        self.work_energy += power_kw_times_seconds / 3600.0


# ---------------------------------------------------------------------
# Main production line model
# ---------------------------------------------------------------------

class ProductionLine:
    def __init__(self, env):
        self.env = env
        self.energy = EnergyRecorder(env)

        # Machines (avg_time, availability, mttr, idlePower, workPower)
        self.conveyor = Machine(env, "Conveyor belt", 6.0, 1.00, 1.0,
                                0.00, 0.00, self.energy)
        self.hantering = Machine(env, "Hantering cell", 25.0, 0.9779, 74.0,
                                 0.50, 0.74, self.energy)
        self.loading = Machine(env, "Loading robot", 12.0, 0.9049, 68.0,
                               0.25, 0.72, self.energy)
        self.press1 = Machine(env, "Presses cell 1", 175.0, 0.8779, 73.0,
                              1.25, 1.28, self.energy)
        self.press2 = Machine(env, "Presses cell 2", 176.0, 0.8769, 74.0,
                              1.25, 1.27, self.energy)
        self.quality = Machine(env, "Quality station cell", 41.0, 0.8587, 66.0,
                               0.58, 0.84, self.energy)
        self.wash = Machine(env, "Washing machine", 14.0, 0.8089, 269.0,
                            4.28, 35.24, self.energy)

        # Buffers as specified
        self.post_loading_buf = Buffer(env, "PostLoadingBuffer", 2, 10, self.energy)
        self.post_conveyor_buf = Buffer(env, "PostConveyorBuffer", 2, 10, self.energy)
        self.post_washing_buf = Buffer(env, "PostWashingBuffer", 2, 10, self.energy)
        self.pre_press1_buf = Buffer(env, "PrePress1Buffer", 3, 32, self.energy)
        self.pre_press2_buf = Buffer(env, "PrePress2Buffer", 3, 32, self.energy)
        self.post_press_buf = Buffer(env, "PostPress1&Press2Buffer", 3, 32, self.energy)

        # Flow control between presses (even split by part count)
        self.num_to_press1 = 0
        self.num_to_press2 = 0

        # stats
        self.total_started = 0
        self.total_finished = 0
        self.total_defects = 0

        # WIP / inventory tracking
        self.num_in_system = 0
        self.area_num_in_system = 0.0
        self.last_wip_change = env.now
        env.process(self.track_wip())

        # start generator
        env.process(self.source())

    def track_wip(self):
        while True:
            now = self.env.now
            dt = now - self.last_wip_change
            if dt > 0:
                self.area_num_in_system += self.num_in_system * dt
                self.last_wip_change = now
            yield self.env.timeout(60.0)  # sample every minute

    def change_wip(self, delta):
        now = self.env.now
        dt = now - self.last_wip_change
        if dt > 0:
            self.area_num_in_system += self.num_in_system * dt
            self.last_wip_change = now
        self.num_in_system += delta

    def source(self):
        """Infinite source, one new raw part whenever system can accept it."""
        i = 0
        while True:
            yield from wait_until_work_time(self.env)
            i += 1
            part = {"id": i}
            self.total_started += 1
            self.change_wip(1)
            self.env.process(self.process_part(part))
            # 1-second interval between starts
            yield self.env.timeout(1.0)

    def process_part(self, part):
        # Loading robot
        yield from self.loading.process_part(part)
        yield self.post_loading_buf.put(part)

        # Post loading buffer
        part = yield self.post_loading_buf.get()
        yield from self.post_loading_buf.process_part(part)

        # Conveyor
        yield from self.conveyor.process_part(part)
        yield self.post_conveyor_buf.put(part)

        # Post conveyor buffer
        part = yield self.post_conveyor_buf.get()
        yield from self.post_conveyor_buf.process_part(part)

        # Washing machine
        yield from self.wash.process_part(part)
        yield self.post_washing_buf.put(part)

        # Post washing buffer
        part = yield self.post_washing_buf.get()
        yield from self.post_washing_buf.process_part(part)

        # Hantering cell
        yield from self.hantering.process_part(part)

        # Decide which press to send this part to (even split by counts)
        if self.num_to_press1 <= self.num_to_press2:
            target_press = 1
            self.num_to_press1 += 1
            yield self.pre_press1_buf.put(part)
        else:
            target_press = 2
            self.num_to_press2 += 1
            yield self.pre_press2_buf.put(part)

        # Process at chosen press (respecting the direct-follow relationships)
        if target_press == 1:
            # Hantering cell -> Presses cell 1
            p = yield self.pre_press1_buf.get()
            yield from self.pre_press1_buf.process_part(p)
            yield from self.press1.process_part(p)
        else:
            # Parallel path for Presses cell 2
            p = yield self.pre_press2_buf.get()
            yield from self.pre_press2_buf.process_part(p)
            yield from self.press2.process_part(p)

        # Common post-press buffer
        yield self.post_press_buf.put(p)

        # Post Press buffer
        p2 = yield self.post_press_buf.get()
        yield from self.post_press_buf.process_part(p2)

        # Quality station (Press 2 -> Quality, but Press 1 joins here in parallel)
        yield from self.quality.process_part(p2)

        # Defect check
        if random.random() < 0.089:
            # defect -> sink
            self.total_defects += 1
        else:
            # good part leaves system
            if self.env.now >= WARMUP:
                self.total_finished += 1

        self.change_wip(-1)


# ---------------------------------------------------------------------
# Run replications
# ---------------------------------------------------------------------

throughputs = []
wips = []
energies = []

for r in range(REPLICATIONS):
    random.seed(RANDOM_SEED + r)
    env = simpy.Environment()
    model = ProductionLine(env)
    env.run(until=SIM_TIME)

    effective_time = SIM_TIME - WARMUP
    throughput_per_hour = model.total_finished / (effective_time / 3600.0)
    mean_wip = model.area_num_in_system / SIM_TIME
    energy_per_part = (model.energy.work_energy /
                       model.total_finished) if model.total_finished > 0 else 0.0

    throughputs.append(throughput_per_hour)
    wips.append(mean_wip)
    energies.append(energy_per_part)

mean_throughput = statistics.mean(throughputs)
mean_wip = statistics.mean(wips)
mean_energy = statistics.mean(energies)

print("=== Mean Overall KPIs over 10 runs ===")
print("Throughput = {} parts/hour".format(mean_throughput))
print("WIP = {} parts".format(mean_wip))
print("Mean Energy Consumption per Part = {} kWh/part".format(mean_energy))