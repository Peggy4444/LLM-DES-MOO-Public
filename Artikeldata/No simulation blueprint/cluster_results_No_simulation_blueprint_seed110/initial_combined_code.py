import simpy
import multiprocessing
import random
import statistics
import math
import os
import csv

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

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
        def _get_proc():
            item = yield self.store.get()
            if self.process_time > 0:
                yield from calendar_wait(self.env, self.calendar_event,
                                         self.process_time)
            return item
        return self.env.process(_get_proc())


# ---------- Simulation model ----------

def production_line_run(rep_results, seed_offset=0, caps=None):
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

    # Buffer capacities from caps if provided, else defaults
    if caps is None:
        post_loading_cap = 2
        post_conveyor_cap = 2
        post_washing_cap = 2
        pre_press1_cap = 3
        pre_press2_cap = 3
        post_press12_cap = 3
    else:
        post_loading_cap, post_conveyor_cap, post_washing_cap, \
        pre_press1_cap, pre_press2_cap, post_press12_cap = caps

    # Buffers with given capacities and times
    post_loading = Buffer(env, "PostLoadingBuffer", post_loading_cap, 10, calendar_event)
    post_conveyor = Buffer(env, "PostConveyorBuffer", post_conveyor_cap, 10, calendar_event)
    post_washing = Buffer(env, "PostWashingBuffer", post_washing_cap, 10, calendar_event)
    pre_press1 = Buffer(env, "PrePress1Buffer", pre_press1_cap, 32, calendar_event)
    pre_press2 = Buffer(env, "PrePress2Buffer", pre_press2_cap, 32, calendar_event)
    post_press12 = Buffer(env, "PostPress1&Press2Buffer", post_press12_cap, 32,
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


# ---------- MOO integration ----------

def run_simulation_for_caps(caps, n_replications=10, base_seed=RANDOM_SEED):
    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed_offset = r * 1000 + local_rng.randint(0, 1000000)
        rep_results = []
        production_line_run(rep_results, seed_offset=seed_offset, caps=caps)
        thp, wip, _ = rep_results[0]
        throughputs.append(thp)
        wips.append(wip)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    Decision variables (all integer in [1,10]):
        x[0] = PostLoadingBuffer capacity
        x[1] = PostConveyorBuffer capacity
        x[2] = PostWashingBuffer capacity
        x[3] = PrePress1Buffer capacity
        x[4] = PrePress2Buffer capacity
        x[5] = PostPress1&Press2Buffer capacity
    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    Constraint:
        Sum of all buffer capacities <= 30.
        If violated, return a large penalty so solution is dominated and
        will not be exported.
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Constraint: total capacity <= 30
    if sum(caps) > 30:
        # Penalize heavily; these will be filtered out when exporting
        return [1e6, 1e6]

    avg_throughput, avg_wip = run_simulation_for_caps(caps, n_replications, base_seed)

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables:
        x[0] = PostLoadingBuffer capacity
        x[1] = PostConveyorBuffer capacity
        x[2] = PostWashingBuffer capacity
        x[3] = PrePress1Buffer capacity
        x[4] = PrePress2Buffer capacity
        x[5] = PostPress1&Press2Buffer capacity

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 xl=None,
                 xu=None,
                 n_replications=10,
                 base_seed=RANDOM_SEED,
                 n_cores=50):
        if xl is None:
            xl = np.array([1] * n_var, dtype=int)
        if xu is None:
            xu = np.array([10] * n_var, dtype=int)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=xl,
                         xu=xu,
                         elementwise_evaluation=False)

        self.n_replications = n_replications
        self.base_seed = base_seed
        self.n_cores = n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.n_replications, self.base_seed)
            for i in range(n_individuals)
        ]

        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=10,
    base_seed=RANDOM_SEED,
    verbose=True
):
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1] * 6),
        xu=np.array([10] * 6),
        n_replications=n_replications,
        base_seed=base_seed,
        n_cores=50
    )

    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(prob=1.0 / problem.n_var, eta=20)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=base_seed,
        save_history=True,
        verbose=verbose
    )

    return res


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export all feasible solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.
    Infeasible points (sum of capacities > 30) are not written.
    Penalized points (with very large objective values) are also skipped.
    """

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "generation",
        "individual_index",
        "post_loading_capacity",
        "post_conveyor_capacity",
        "post_washing_capacity",
        "pre_press1_capacity",
        "pre_press2_capacity",
        "post_press12_capacity",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        for ind_idx, (x, f) in enumerate(zip(X, F)):
            caps = [int(v) for v in x[:6]]
            # Apply same constraint check: skip infeasible
            if sum(caps) > 30:
                continue
            wip = float(f[0])
            throughput = float(-f[1])
            # Skip penalized points
            if wip >= 1e6 or throughput <= -1e6:
                continue
            row = {
                "generation": gen_idx,
                "individual_index": ind_idx,
                "post_loading_capacity": caps[0],
                "post_conveyor_capacity": caps[1],
                "post_washing_capacity": caps[2],
                "pre_press1_capacity": caps[3],
                "pre_press2_capacity": caps[4],
                "post_press12_capacity": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=10,
        verbose=True
    )
    export_history_to_csv(result, filename="moo_simulation_results.csv")