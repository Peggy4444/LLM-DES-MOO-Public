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
    if avail >= 1.0:
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
        yield self.env.process(next_uptime(self.env))
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
                    yield self.env.process(next_uptime(self.env))
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
        yield self.env.process(next_uptime(self.env))
        pt = self.process_time
        remaining = pt
        while remaining > 0:
            if in_downtime(self.env.now):
                yield self.env.process(next_uptime(self.env))
            else:
                step = min(remaining, 60)
                yield self.env.timeout(step)
                remaining -= step
        if output_store is not None:
            yield output_store.put(part)


class ProductionLine:
    def __init__(self, env, rng, caps=None):
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

        # Decision-variable capacities
        if caps is None:
            caps = [2, 2, 2, 3, 3, 3]
        post_loading_cap, post_conveyor_cap, post_washing_cap, pre_press1_cap, pre_press2_cap, post_press12_cap = caps

        # Buffers with capacities and process times
        self.post_loading_buf = Buffer(env, "PostLoadingBuffer", post_loading_cap, 10, rng)
        self.post_conveyor_buf = Buffer(env, "PostConveyorBuffer", post_conveyor_cap, 10, rng)
        self.post_washing_buf = Buffer(env, "PostWashingBuffer", post_washing_cap, 10, rng)
        self.pre_press1_buf = Buffer(env, "PrePress1Buffer", pre_press1_cap, 32, rng)
        self.pre_press2_buf = Buffer(env, "PrePress2Buffer", pre_press2_cap, 32, rng)
        self.post_press12_buf = Buffer(env, "PostPress1&Press2Buffer", post_press12_cap, 32, rng)

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
            yield self.env.process(next_uptime(self.env))
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


def run_replication(seed_offset, caps=None):
    random.seed(RANDOM_SEED + seed_offset)
    rng = random.Random(RANDOM_SEED + seed_offset)
    env = simpy.Environment()
    line = ProductionLine(env, rng, caps=caps)
    env.run(until=SIM_TIME)

    # KPIs after warmup
    sim_duration = SIM_TIME - WARMUP
    completed_after_warm = 0
    for part in list(line.final_buffer.items):
        if part['birth'] >= WARMUP:
            completed_after_warm += 1

    throughput_per_hour = completed_after_warm / (sim_duration / 3600.0) if sim_duration > 0 else 0.0

    # WIP: average over time, use area from t>=WARMUP
    wip_area = line.wip_time_area
    avg_wip = wip_area / SIM_TIME if SIM_TIME > 0 else 0.0

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


def run_simulation_for_caps(seed, caps, n_replications=REPLICATIONS):
    """
    Run the given production line simulation for a specific set of buffer capacities
    over multiple replications and return average throughput and WIP.

    caps: list or array of 6 integers:
        [post_loading_cap,
         post_conveyor_cap,
         post_washing_cap,
         pre_press1_cap,
         pre_press2_cap,
         post_press12_cap]
    """
    throughputs = []
    wips = []

    # Local RNG to generate different seeds per replication
    local_rng = random.Random(seed)

    for _ in range(n_replications):
        rep_seed_offset = local_rng.randint(0, 10**9)
        tp, wip, _ = run_replication(rep_seed_offset, caps=caps)
        throughputs.append(tp)
        wips.append(wip)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [f1, f2] = [avg_wip, -avg_throughput]
    """
    x, n_replications, base_seed = args
    # Decision variables: 6 buffer capacities, integers in [1,10]
    caps = [int(v) for v in x[:6]]

    # Constraint: total capacity <= 40.
    if sum(caps) > 40:
        return [1e9, 1e9]

    # Create a local random instance
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    seed = base_seed + local_rng.randint(0, 10**9)

    avg_throughput, avg_wip = run_simulation_for_caps(seed, caps, n_replications)

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integers in [1,10]):
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
                 n_replications=REPLICATIONS,
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

        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=REPLICATIONS,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    """
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        xl=np.array([1] * 6),
        xu=np.array([10] * 6),
        n_replications=n_replications,
        base_seed=base_seed,
        n_cores=n_cores
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

    Infeasible points (constraint-violating) are not written to the CSV.
    Here, infeasibility is detected via the large penalty value (1e9).
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
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives

            # Skip infeasible/penalized points
            if wip >= 1e9 or throughput <= -1e9:
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
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")