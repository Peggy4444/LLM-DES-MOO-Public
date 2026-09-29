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
                # time to next failure (if any)
                if self.mttf is not None:
                    ttf = random.expovariate(1.0 / self.mttf)
                else:
                    ttf = float('inf')

                # time until end of current working window
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
        def _get_process():
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
        return self.env.process(_get_process())


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


def wip_monitor(env, stats):
    prev_time = env.now
    prev_wip = stats['wip']
    while True:
        yield env.timeout(60)  # sample every minute
        now = env.now
        stats['wip_time_area'] += prev_wip * (now - prev_time)
        prev_time = now
        prev_wip = stats['wip']


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
    throughput_per_hour = stats['good'] / (effective_time / 3600.0)
    avg_wip = stats['wip_time_area'] / SIM_TIME
    total_parts = max(1, stats['good'])  # avoid div0
    energy_per_part = (stats['energy_idle'] + stats['energy_work']) / total_parts / 3600.0

    results['throughput'].append(throughput_per_hour)
    results['wip'].append(avg_wip)
    results['energy_per_part'].append(energy_per_part)


# =========================
# MOO-related functionality
# =========================

def _single_replication_with_caps(seed, caps):
    """
    Single replication of the provided simulation model with modified buffer capacities.
    Returns dict with keys: throughput, wip.
    """
    random.seed(seed)
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

    # Unpack capacities
    (cap_post_load,
     cap_post_conv,
     cap_post_wash,
     cap_pre_p1,
     cap_pre_p2,
     cap_post_p12) = [int(v) for v in caps]

    machines = {
        'Conveyor belt': Machine(env, 'Conveyor belt', 6.0, 100.0, 1.0, 0.0, 0.0),
        'Hantering cell': Machine(env, 'Hantering cell', 25.0, 97.79, 74.0, 0.50, 0.74),
        'Loading robot': Machine(env, 'Loading robot', 12.0, 90.49, 68.0, 0.25, 0.72),
        'Presses cell 1': Machine(env, 'Presses cell 1', 175.0, 87.79, 73.0, 1.25, 1.28),
        'Presses cell 2': Machine(env, 'Presses cell 2', 176.0, 87.69, 74.0, 1.25, 1.27),
        'Quality station cell': Machine(env, 'Quality station cell', 41.0, 85.87, 66.0, 0.58, 0.84),
        'Washing machine': Machine(env, 'Washing machine', 14.0, 80.89, 269.0, 4.28, 35.24),
    }

    buffers = {
        'PostLoadingBuffer': BufferWithDelay(env, 'PostLoadingBuffer', capacity=cap_post_load, delay=10),
        'PostConveyorBuffer': BufferWithDelay(env, 'PostConveyorBuffer', capacity=cap_post_conv, delay=10),
        'PostWashingBuffer': BufferWithDelay(env, 'PostWashingBuffer', capacity=cap_post_wash, delay=10),
        'PrePress1Buffer': BufferWithDelay(env, 'PrePress1Buffer', capacity=cap_pre_p1, delay=32),
        'PrePress2Buffer': BufferWithDelay(env, 'PrePress2Buffer', capacity=cap_pre_p2, delay=32),
        'PostPress1&Press2Buffer': BufferWithDelay(env, 'PostPress1&Press2Buffer', capacity=cap_post_p12, delay=32),
    }

    env.process(source(env, machines, buffers, stats, defect_rate=0.089, interarrival=30.0))
    env.process(wip_monitor(env, stats))

    env.run(until=SIM_TIME)

    effective_time = SIM_TIME - WARMUP
    throughput_per_hour = stats['good'] / (effective_time / 3600.0) if effective_time > 0 else 0.0
    avg_wip = stats['wip_time_area'] / SIM_TIME if SIM_TIME > 0 else 0.0

    return {
        "throughput": throughput_per_hour,
        "wip": avg_wip
    }


def run_simulation_for_caps(seed, caps, n_replications=REPS):
    """
    Run the given simulation model for a specific set of buffer capacities
    over multiple replications and return average KPIs.
    caps: [PostLoadingBuffer, PostConveyorBuffer, PostWashingBuffer,
           PrePress1Buffer, PrePress2Buffer, PostPress1&Press2Buffer]
    """
    caps = [int(v) for v in caps[:6]]

    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(seed + sum(caps))

    for r in range(n_replications):
        rep_seed = seed + r + local_rng.randint(0, 1_000_000)
        res = _single_replication_with_caps(rep_seed, caps)
        throughputs.append(res["throughput"])
        wips.append(res["wip"])

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_wip, avg_throughput


def evaluate_single_individual(args):
    """
    Evaluate a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Constraint: total buffer capacity <= 50
    if sum(caps) > 50:
        return [np.nan, np.nan]

    local_seed = base_seed + sum(caps)
    avg_wip, avg_throughput = run_simulation_for_caps(local_seed, caps, n_replications)

    # Objectives: minimize wip, maximize throughput (so use negative)
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
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
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 n_replications=REPS,
                 base_seed=RANDOM_SEED,
                 n_cores=50):
        if xl is None:
            xl = np.array([1] * n_var, dtype=int)
        if xu is None:
            xu = np.array([10] * n_var, dtype=int)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
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

        # Use exactly 50 cores
        with multiprocessing.Pool(processes=50) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=REPS,
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
    Infeasible (constraint-violating) individuals with NaN objectives are skipped.
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "generation",
        "individual_index",
        "PostLoadingBuffer_capacity",
        "PostConveyorBuffer_capacity",
        "PostWashingBuffer_capacity",
        "PrePress1Buffer_capacity",
        "PrePress2Buffer_capacity",
        "PostPress1AndPress2Buffer_capacity",
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
            wip = float(f[0])
            throughput = float(-f[1])

            # Skip individuals that were not evaluated (constraint violations)
            if np.isnan(wip) or np.isnan(throughput):
                continue

            caps = [int(v) for v in x[:6]]
            row = {
                "generation": gen_idx,
                "individual_index": ind_idx,
                "PostLoadingBuffer_capacity": caps[0],
                "PostConveyorBuffer_capacity": caps[1],
                "PostWashingBuffer_capacity": caps[2],
                "PrePress1Buffer_capacity": caps[3],
                "PrePress2Buffer_capacity": caps[4],
                "PostPress1AndPress2Buffer_capacity": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization on the simulation model
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPS,
        verbose=True
    )

    # Export all feasible solutions to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")