import simpy
import multiprocessing
import random
import statistics
import os
import csv
import math

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

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

# Decision variables: all 6 buffer capacities in BUFFERS (1–10, integer)
BUFFER_NAMES = [
    "PostLoading",
    "PostConveyor",
    "PostWashing",
    "PrePress1",
    "PrePress2",
    "PostPresses",
]

def run_simulation_with_caps(seed, caps):
    """
    Run the given production line simulation with modified buffer capacities.

    caps: list/array of 6 integers (1–10) in the order of BUFFER_NAMES.
    Returns:
        throughput_per_hour, mean_wip, energy_per_part
    """
    global RANDOM_SEED, SIM_TIME, WARMUP, REPLICATIONS
    global MACHINES, BUFFERS, Machine, Buffer
    global part_generator, run_replication

    # Set the capacities in BUFFERS according to caps
    for i, bname in enumerate(BUFFER_NAMES):
        BUFFERS[bname]["cap"] = int(caps[i])

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

    return mean_throughput, mean_wip, mean_energy

def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications (handled inside run_simulation_with_caps).

    Returns objectives: [f1 (wip), f2 (-throughput)]
    """
    x, base_seed = args
    caps = [int(v) for v in x[:6]]

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    seed = base_seed + local_rng.randint(0, 10**6)

    throughput, wip, _ = run_simulation_with_caps(seed, caps)

    return [wip, -throughput]

class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (integer, 1–10):
        x[0] = PostLoading cap
        x[1] = PostConveyor cap
        x[2] = PostWashing cap
        x[3] = PrePress1 cap
        x[4] = PrePress2 cap
        x[5] = PostPresses cap

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
                 base_seed=88,
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

        self.base_seed = base_seed
        self.n_cores = n_cores

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        tasks = [
            (X[i], self.base_seed)
            for i in range(n_individuals)
        ]

        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)

def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    base_seed=88,
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
    Export all solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Only feasible points are exported. In this setup, all points within bounds
    are feasible, and no objective is evaluated for infeasible points.
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "generation",
        "individual_index",
        "PostLoading_capacity",
        "PostConveyor_capacity",
        "PostWashing_capacity",
        "PrePress1_capacity",
        "PrePress2_capacity",
        "PostPresses_capacity",
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
            # If any objective is NaN, treat as infeasible and skip
            if np.any(np.isnan(f)):
                continue

            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives

            row = {
                "generation": gen_idx,
                "individual_index": ind_idx,
                "PostLoading_capacity": caps[0],
                "PostConveyor_capacity": caps[1],
                "PostWashing_capacity": caps[2],
                "PrePress1_capacity": caps[3],
                "PrePress2_capacity": caps[4],
                "PostPresses_capacity": caps[5],
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
        base_seed=88,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")