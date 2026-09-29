import simpy
import random
import statistics
import math

RANDOM_SEED = 110

SIM_TIME = 691200          # total simulation time [s]
WARMUP = 86400             # warm-up [s]
N_REPS = 10                # replications

# ---------- Calendar: production time ----------
DAY = 24 * 3600
WEEK = 7 * DAY

def in_production_time(t):
    """Return True if time t (in seconds) is inside production calendar."""
    t_week = t % WEEK
    day = int(t_week // DAY)          # 0=Mon ... 6=Sun
    t_day = t_week % DAY              # seconds since midnight
    # production days: Mon–Fri (0–4)
    if day > 4:
        return False
    # production hours: 07:00–17:00
    start = 7 * 3600
    end = 17 * 3600
    return start <= t_day < end

def time_to_next_open(t):
    """If currently closed, return time until next open; otherwise 0."""
    if in_production_time(t):
        return 0.0
    t_week = t % WEEK
    day = int(t_week // DAY)
    t_day = t_week % DAY
    start = 7 * 3600
    end = 17 * 3600

    # If weekend, next open is next Monday 07:00
    if day >= 5:
        days_ahead = (7 - day) % 7  # until Monday
        return (days_ahead * DAY + start) - t_day

    # Weekday but outside window
    if t_day < start:
        return start - t_day
    if t_day >= end:
        # next day 07:00
        return (DAY - t_day) + start

    return 0.0


def calendar_process(env, calendar_event):
    """Process that controls production on/off according to calendar."""
    while True:
        # Wait until we are in production time
        dt = time_to_next_open(env.now)
        if dt > 0:
            calendar_event.succeed(False)   # signal stop
            calendar_event = env.event()
            yield env.timeout(dt)
        else:
            # we are at opening time, let production run until close
            calendar_event.succeed(True)    # signal start
            calendar_event = env.event()
            # time to close
            t_day = (env.now % DAY)
            end = 17 * 3600
            yield env.timeout(end - t_day)


def calendar_wait(env, calendar_event, duration):
    """Wait for 'duration' seconds, respecting the production calendar."""
    remaining = duration
    while remaining > 0:
        if not in_production_time(env.now):
            # wait until production starts again
            dt = time_to_next_open(env.now)
            yield env.timeout(dt)
            continue
        # we are in production time: limit next chunk to end of current window
        t_day = (env.now % DAY)
        end = 17 * 3600
        available = end - t_day
        dt = min(remaining, available)
        yield env.timeout(dt)
        remaining -= dt


# ---------- Machine / Buffer definitions ----------

class Machine:
    def __init__(self, env, name, mean_time, availability, mttr,
                 energy_idle, energy_work, calendar_event):
        self.env = env
        self.name = name
        self.mean_time = mean_time
        self.availability = availability / 100.0
        self.mttr = mttr
        self.energy_idle = energy_idle
        self.energy_work = energy_work
        self.calendar_event = calendar_event

        self.resource = simpy.Resource(env, capacity=1)

        # statistics
        self.energy = 0.0
        self.last_state_change = env.now
        self.idle = True

        # breakdown logic: simple up/down with exponential times
        self.up = True
        self.process = env.process(self.breakdown_process())

    def _mtbf(self):
        if self.availability <= 0 or self.availability >= 1:
            return 1e9
        return self.mttr * self.availability / (1 - self.availability)

    def breakdown_process(self):
        while True:
            # time until next failure
            mtbf = self._mtbf()
            ttf = random.expovariate(1.0 / mtbf)
            yield from calendar_wait(self.env, self.calendar_event, ttf)
            # go down
            self.up = False
            down_time = random.expovariate(1.0 / self.mttr)
            yield from calendar_wait(self.env, self.calendar_event, down_time)
            self.up = True

    def add_energy(self):
        now = self.env.now
        dt = now - self.last_state_change
        if dt < 0:
            dt = 0
        if self.idle:
            self.energy += self.energy_idle * dt / 3600.0
        else:
            self.energy += self.energy_work * dt / 3600.0
        self.last_state_change = now

    def set_state(self, idle):
        if idle != self.idle:
            self.add_energy()
            self.idle = idle

    def process_part(self, part):
        # wait for machine to be up and in calendar
        while not (self.up and in_production_time(self.env.now)):
            yield self.env.timeout(1)
        with self.resource.request() as req:
            yield req
            # processing
            self.set_state(False)
            pt = self.mean_time
            yield from calendar_wait(self.env, self.calendar_event, pt)
            self.set_state(True)


class Buffer:
    def __init__(self, env, name, capacity, process_time, calendar_event):
        self.env = env
        self.name = name
        self.process_time = process_time
        self.calendar_event = calendar_event
        self.store = simpy.Store(env, capacity=capacity)

    def put(self, item):
        return self.store.put(item)

    def get(self):
        item = yield self.store.get()
        if self.process_time > 0:
            yield from calendar_wait(self.env, self.calendar_event,
                                self.process_time)
        return item


# ---------- Simulation model ----------

def production_line_run(rep_results, seed_offset=0):
    random.seed(RANDOM_SEED + seed_offset)
    env = simpy.Environment()

    calendar_event = env.event()
    env.process(calendar_process(env, calendar_event))

    # Machines
    conveyor = Machine(env, "Conveyor belt", 6.0, 100.0, 1.0, 0.0, 0.0,
                       calendar_event)
    hantering = Machine(env, "Hantering cell", 25.0, 97.79, 74.0, 0.50, 0.74,
                        calendar_event)
    loading_robot = Machine(env, "Loading robot", 12.0, 90.49, 68.0,
                            0.25, 0.72, calendar_event)
    press1 = Machine(env, "Presses cell 1", 175.0, 87.79, 73.0,
                     1.25, 1.28, calendar_event)
    press2 = Machine(env, "Presses cell 2", 176.0, 87.69, 74.0,
                     1.25, 1.27, calendar_event)
    quality = Machine(env, "Quality station cell", 41.0, 85.87, 66.0,
                      0.58, 0.84, calendar_event)
    washing = Machine(env, "Washing machine", 14.0, 80.89, 269.0,
                      4.28, 35.24, calendar_event)

    # Buffers with given capacities and times
    post_loading = Buffer(env, "PostLoadingBuffer", 2, 10, calendar_event)
    post_conveyor = Buffer(env, "PostConveyorBuffer", 2, 10, calendar_event)
    post_washing = Buffer(env, "PostWashingBuffer", 2, 10, calendar_event)
    pre_press1 = Buffer(env, "PrePress1Buffer", 3, 32, calendar_event)
    pre_press2 = Buffer(env, "PrePress2Buffer", 3, 32, calendar_event)
    post_press12 = Buffer(env, "PostPress1&Press2Buffer", 3, 32,
                          calendar_event)

    defect_rate = 0.089

    # stats
    completed = []
    defected = []
    parts_in_system = []

    def wip_monitor():
        while True:
            yield env.timeout(60)  # sample every minute
            parts_in_system.append(
                (len(post_loading.store.items) +
                 len(post_conveyor.store.items) +
                 len(post_washing.store.items) +
                 len(pre_press1.store.items) +
                 len(pre_press2.store.items) +
                 len(post_press12.store.items))
            )

    env.process(wip_monitor())

    def source():
        i = 0
        while True:
            i += 1
            part = {'id': i, 'birth': env.now}
            env.process(part_flow(part))
            # simple interarrival: push as fast as calendar allows
            yield from calendar_wait(env, calendar_event, 6.0)

    def part_flow(part):
        # Loading robot -> Conveyor belt
        yield from loading_robot.process_part(part)
        yield post_loading.put(part)

        part = yield post_loading.get()
        yield from conveyor.process_part(part)
        yield post_conveyor.put(part)

        # Conveyor belt -> Washing machine
        part = yield post_conveyor.get()
        yield from washing.process_part(part)
        yield post_washing.put(part)

        # Washing machine -> Hantering cell
        part = yield post_washing.get()
        yield from hantering.process_part(part)

        # Split evenly to two presses (parallel presses)
        if part['id'] % 2 == 0:
            yield pre_press1.put(part)
            part = yield pre_press1.get()
            yield from press1.process_part(part)
        else:
            yield pre_press2.put(part)
            part = yield pre_press2.get()
            yield from press2.process_part(part)

        # Merge after Press 1 & 2
        yield post_press12.put(part)
        part = yield post_press12.get()

        # Presses cell 1/2 -> Quality station cell
        yield from quality.process_part(part)

        # Defect sink at Quality station
        if random.random() < defect_rate:
            defected.append(part)
        else:
            completed.append(part)

    env.process(source())
    env.run(until=SIM_TIME)

    # Remove warm-up effects for throughput and WIP
    completed_after_warmup = [
        p for p in completed if p['birth'] >= WARMUP
    ]
    thp = len(completed_after_warmup) / ((SIM_TIME - WARMUP) / 3600.0)

    if parts_in_system:
        mean_wip = statistics.mean(parts_in_system)
    else:
        mean_wip = 0.0

    total_energy = (conveyor.energy + hantering.energy + loading_robot.energy +
                    press1.energy + press2.energy + quality.energy +
                    washing.energy)
    n_good = len(completed)
    if n_good > 0:
        energy_per_part = total_energy / n_good
    else:
        energy_per_part = 0.0

    rep_results.append((thp, mean_wip, energy_per_part))


# ---------- Run replications ----------

all_results = []
for r in range(N_REPS):
    rep_results = []
    production_line_run(rep_results, seed_offset=r * 1000)
    all_results.extend(rep_results)

throughputs = [x[0] for x in all_results]
wips = [x[1] for x in all_results]
energies = [x[2] for x in all_results]

mean_throughput = statistics.mean(throughputs) if throughputs else 0.0
mean_wip = statistics.mean(wips) if wips else 0.0
mean_energy = statistics.mean(energies) if energies else 0.0

print("=== Mean Overall KPIs over 10 runs ===")
print("Throughput = {:.3f} parts/hour".format(mean_throughput))
print("WIP = {:.3f} parts".format(mean_wip))
print("Mean Energy Consumption per Part = {:.3f} kWh/part".format(mean_energy))