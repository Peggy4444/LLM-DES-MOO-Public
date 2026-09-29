import simpy
import multiprocessing
import random
import statistics
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

RANDOM_SEED = 77

SIM_TIME = 691200          # total simulation time (s)
WARMUP = 86400             # warm-up (s)
REPLICATIONS = 10

# Interarrival rate – not given, assume one part every 60s
INTERARRIVAL = 60.0

# Station data (times in seconds)
STATIONS = {
    "Conveyor":      {"mean": 6.0,   "avail": 1.00,  "mttr": 1.0,   "idle_e": 0.0,  "work_e": 0.0},
    "Handling":      {"mean": 25.0,  "avail": 0.9779,"mttr": 74.0,  "idle_e": 0.50, "work_e": 0.74},
    "Loading":       {"mean": 12.0,  "avail": 0.9049,"mttr": 68.0,  "idle_e": 0.25, "work_e": 0.72},
    "Press1":        {"mean": 175.0, "avail": 0.8779,"mttr": 73.0,  "idle_e": 1.25, "work_e": 1.28},
    "Press2":        {"mean": 176.0, "avail": 0.8769,"mttr": 74.0,  "idle_e": 1.25, "work_e": 1.27},
    "Quality":       {"mean": 41.0,  "avail": 0.8587,"mttr": 66.0,  "idle_e": 0.58, "work_e": 0.84},
    "Washing":       {"mean": 14.0,  "avail": 0.8089,"mttr": 269.0, "idle_e": 4.28, "work_e": 35.24},
}

# Buffers with capacity and process time
BUF_CONF = {
    "PostLoading":  {"cap": 2, "ptime": 10.0},
    "PostConveyor": {"cap": 2, "ptime": 10.0},
    "PostWashing":  {"cap": 2, "ptime": 10.0},
    "PrePress1":    {"cap": 3, "ptime": 32.0},
    "PrePress2":    {"cap": 3, "ptime": 32.0},
    "PostPress12":  {"cap": 3, "ptime": 32.0},
}

DEFECT_RATE = 0.089  # defect sink at Quality

# ---------------------------------------------------------------------------

class Machine:
    def __init__(self, env, name, conf, energy_tracker):
        self.env = env
        self.name = name
        self.conf = conf
        self.resource = simpy.Resource(env, capacity=1)
        self.mtbf = conf["mean"] * conf["avail"] / (1 - conf["avail"]) if conf["avail"] < 1.0 else None
        self.mttr = conf["mttr"]
        self.energy_idle = conf["idle_e"]     # kW
        self.energy_work = conf["work_e"]     # kW
        self.energy_tracker = energy_tracker
        self.broken = False
        if self.mtbf is not None:
            env.process(self.breakdown_process())

    def breakdown_process(self):
        while True:
            ttf = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(ttf)
            self.broken = True
            # Wait until machine is idle
            while self.resource.count > 0:
                yield self.env.timeout(1)
            ttr = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(ttr)
            self.broken = False

    def process_part(self, part_id):
        with self.resource.request() as req:
            yield req
            while self.broken:
                yield self.env.timeout(1)
            start = self.env.now
            mean = self.conf["mean"]
            pt = random.expovariate(1.0 / mean)
            work_start = self.env.now
            yield self.env.timeout(pt)
            work_time = self.env.now - work_start
            idle_time = (start - work_start) if start > work_start else 0.0
            self.energy_tracker.add(self.energy_work * work_time / 3600.0)
            self.energy_tracker.add(self.energy_idle * idle_time / 3600.0)


class Buffer:
    def __init__(self, env, name, cap, ptime):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=cap)
        self.ptime = ptime

    def put(self, item):
        return self.store.put(item)

    def get(self):
        return self.store.get()

    def process_delay(self):
        dt = random.expovariate(1.0 / self.ptime)
        return self.env.timeout(dt)


class EnergyTracker:
    def __init__(self):
        self.energy = 0.0

    def add(self, e):
        self.energy += e


def is_working_time(t):
    day = int(t // 86400) % 7
    sec_of_day = t % 86400
    # Stop: Fri 17:00–Sat 07:00, Sat 17:00–Sun 07:00
    if day == 4:  # Friday
        if 17*3600 <= sec_of_day < 24*3600:
            return False
    if day == 5:  # Saturday
        if 0 <= sec_of_day < 7*3600:
            return False
        if 17*3600 <= sec_of_day < 24*3600:
            return False
    if day == 6:  # Sunday
        if 0 <= sec_of_day < 7*3600:
            return False
    return True


def wait_until_working(env):
    while not is_working_time(env.now):
        yield env.timeout(60)


def part_generator(env, buffers, overall_wip, machines):
    post_loading, post_conv, post_wash, pre_p1, pre_p2, post_p12 = buffers
    m_loading, m_conveyor, m_washing, m_handling = machines
    part_id = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / INTERARRIVAL))
        yield env.process(wait_until_working(env))
        part_id += 1
        overall_wip[0] += 1
        env.process(part_flow(env, part_id, buffers, overall_wip, machines))


def part_flow(env, part_id, buffers, overall_wip, machines):
    post_loading, post_conv, post_wash, pre_p1, pre_p2, post_p12 = buffers
    m_loading, m_conveyor, m_washing, m_handling = machines

    # Loading robot -> PostLoading buffer
    yield env.process(m_loading.process_part(part_id))
    yield post_loading.process_delay()
    yield post_loading.put(part_id)

    # PostLoading buffer -> Conveyor belt
    item = yield post_loading.get()
    yield env.process(m_conveyor.process_part(item))
    yield post_conv.process_delay()
    yield post_conv.put(item)

    # PostConveyor buffer -> Washing machine
    item = yield post_conv.get()
    yield env.process(m_washing.process_part(item))
    yield post_wash.process_delay()
    yield post_wash.put(item)

    # PostWashing buffer -> Handling cell
    item = yield post_wash.get()
    yield env.process(m_handling.process_part(item))

    # Split evenly (random but unbiased) to two presses
    target = random.choice(["P1", "P2"])
    if target == "P1":
        yield pre_p1.process_delay()
        yield pre_p1.put(item)
    else:
        yield pre_p2.process_delay()
        yield pre_p2.put(item)
    # From here on, presses pull from pre_p1/pre_p2 and push to post_p12,
    # then postpress_to_quality handles Quality and sinks (good/defect).


def press_process(env, name, machine, in_buffer, out_buffer):
    while True:
        item = yield in_buffer.get()
        yield in_buffer.process_delay()
        yield env.process(machine.process_part(item))
        yield out_buffer.process_delay()
        yield out_buffer.put(item)


def postpress_to_quality(env, machine_quality, post_p12_buf, defect_sink, overall_wip, throughput_counter, warmup):
    while True:
        item = yield post_p12_buf.get()
        yield post_p12_buf.process_delay()
        yield env.process(machine_quality.process_part(item))
        if random.random() < DEFECT_RATE:
            defect_sink.append(item)
            overall_wip[0] -= 1
        else:
            throughput_counter.append((env.now, item))
            overall_wip[0] -= 1


def run_replication(rep):
    random.seed(RANDOM_SEED + rep)
    env = simpy.Environment()
    energy_tracker = EnergyTracker()

    # Machines
    m_loading = Machine(env, "Loading", STATIONS["Loading"], energy_tracker)
    m_conveyor = Machine(env, "Conveyor", STATIONS["Conveyor"], energy_tracker)
    m_washing = Machine(env, "Washing", STATIONS["Washing"], energy_tracker)
    m_handling = Machine(env, "Handling", STATIONS["Handling"], energy_tracker)
    m_press1 = Machine(env, "Press1", STATIONS["Press1"], energy_tracker)
    m_press2 = Machine(env, "Press2", STATIONS["Press2"], energy_tracker)
    m_quality = Machine(env, "Quality", STATIONS["Quality"], energy_tracker)

    # Buffers (all have explicit finite capacities)
    post_loading = Buffer(env, "PostLoading", BUF_CONF["PostLoading"]["cap"], BUF_CONF["PostLoading"]["ptime"])
    post_conv = Buffer(env, "PostConveyor", BUF_CONF["PostConveyor"]["cap"], BUF_CONF["PostConveyor"]["ptime"])
    post_wash = Buffer(env, "PostWashing", BUF_CONF["PostWashing"]["cap"], BUF_CONF["PostWashing"]["ptime"])
    pre_p1 = Buffer(env, "PrePress1", BUF_CONF["PrePress1"]["cap"], BUF_CONF["PrePress1"]["ptime"])
    pre_p2 = Buffer(env, "PrePress2", BUF_CONF["PrePress2"]["cap"], BUF_CONF["PrePress2"]["ptime"])
    post_p12 = Buffer(env, "PostPress12", BUF_CONF["PostPress12"]["cap"], BUF_CONF["PostPress12"]["ptime"])

    buffers = (post_loading, post_conv, post_wash, pre_p1, pre_p2, post_p12)
    machines_for_flow = (m_loading, m_conveyor, m_washing, m_handling)

    overall_wip = [0]
    throughput_counter = []
    defect_sink = []

    env.process(part_generator(env, buffers, overall_wip, machines_for_flow))
    env.process(press_process(env, "Press1", m_press1, pre_p1, post_p12))
    env.process(press_process(env, "Press2", m_press2, pre_p2, post_p12))
    env.process(postpress_to_quality(env, m_quality, post_p12, defect_sink, overall_wip, throughput_counter, WARMUP))

    wip_samples = []

    def wip_monitor(env, wip_list, sample_list):
        while True:
            if env.now >= WARMUP:
                sample_list.append(wip_list[0])
            yield env.timeout(300)

    env.process(wip_monitor(env, overall_wip, wip_samples))

    env.run(until=SIM_TIME)

    completed_after_warmup = [t for (t, _) in throughput_counter if t >= WARMUP]
    hours_after_warmup = (SIM_TIME - WARMUP) / 3600.0
    throughput_per_hour = len(completed_after_warmup) / hours_after_warmup if hours_after_warmup > 0 else 0.0
    mean_wip = statistics.mean(wip_samples) if wip_samples else 0.0
    total_energy = energy_tracker.energy
    energy_per_part = total_energy / len(completed_after_warmup) if completed_after_warmup else 0.0

    return throughput_per_hour, mean_wip, energy_per_part


def run_simulation_for_caps(seed, caps, n_replications=REPLICATIONS):
    """
    Run the given production line simulation for a specific set of buffer capacities
    over multiple replications and return average throughput and WIP.
    """
    # caps is a list/array of 6 integers in order:
    # [PostLoading, PostConveyor, PostWashing, PrePress1, PrePress2, PostPress12]
    caps = [int(v) for v in caps[:6]]

    # Backup original capacities
    original_caps = {
        name: BUF_CONF[name]["cap"]
        for name in BUF_CONF.keys()
    }

    # Apply new capacities
    buf_names = ["PostLoading", "PostConveyor", "PostWashing", "PrePress1", "PrePress2", "PostPress12"]
    for name, cap in zip(buf_names, caps):
        BUF_CONF[name]["cap"] = cap

    throughputs = []
    wips = []

    # Local RNG to generate distinct seeds per replication
    local_rng = random.Random(seed)

    for r in range(n_replications):
        rep_seed_offset = local_rng.randint(0, 10**9)
        # run_replication uses RANDOM_SEED + rep internally, so we temporarily
        # adjust RANDOM_SEED to incorporate our offset
        global RANDOM_SEED
        old_rs = RANDOM_SEED
        RANDOM_SEED = old_rs + rep_seed_offset

        th, w, _ = run_replication(r)
        throughputs.append(th)
        wips.append(w)

        RANDOM_SEED = old_rs

    # Restore original capacities
    for name in BUF_CONF.keys():
        BUF_CONF[name]["cap"] = original_caps[name]

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [f1, f2] = [avg_wip, -avg_throughput].
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Create a local seed based on base_seed and decision variables
    seed = base_seed + sum(caps)

    avg_throughput, avg_wip = run_simulation_for_caps(seed, caps, n_replications=n_replications)

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integer in [1,10]):
        x[0] = PostLoading cap
        x[1] = PostConveyor cap
        x[2] = PostWashing cap
        x[3] = PrePress1 cap
        x[4] = PrePress2 cap
        x[5] = PostPress12 cap

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

        with multiprocessing.Pool(processes=50) as pool:
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
    Export all solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Only feasible/evaluated points are exported.
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
        "PostPress12_capacity",
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
            # If F contains NaN or inf, skip (treat as non-evaluated / infeasible)
            if not np.all(np.isfinite(f)):
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
                "PostPress12_capacity": caps[5],
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