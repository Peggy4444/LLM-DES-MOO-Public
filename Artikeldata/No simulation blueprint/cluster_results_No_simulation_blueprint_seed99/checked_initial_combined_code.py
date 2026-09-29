import simpy
import random
import math
import statistics
import multiprocessing
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

RANDOM_SEED = 99

SIM_TIME = 691200
WARM_UP = 86400
REPLICATIONS = 10

# Interarrival time (assumed)
MEAN_INTERARRIVAL = 60.0  # seconds, adjust as needed

DEFECT_RATE = 0.089

# Machine parameters: (mean_proc_time, availability, MTTR, idle_energy, work_energy)
MACHINES = {
    "Conveyor":        (6.0,   1.0000,   1.0,   0.00,  0.00),
    "Handling":        (25.0,  0.9779,  74.0,   0.50,  0.74),
    "Loading":         (12.0,  0.9049,  68.0,   0.25,  0.72),
    "Press1":          (175.0, 0.8779,  73.0,   1.25,  1.28),
    "Press2":          (176.0, 0.8769,  74.0,   1.25,  1.27),
    "Quality":         (41.0,  0.8587,  66.0,   0.58,  0.84),
    "Washing":         (14.0,  0.8089, 269.0,   4.28, 35.24),
}

# Buffer specs: (capacity, process_time)
BUFFERS = {
    "PostLoadingBuffer":      (2, 10.0),
    "PostConveyorBuffer":     (2, 10.0),
    "PostWashingBuffer":      (2, 10.0),
    "PrePress1Buffer":        (3, 32.0),
    "PrePress2Buffer":        (3, 32.0),
    "PostPress1_2Buffer":     (3, 32.0),
}

DAY = 24 * 3600

def is_open(t):
    t_week = t % (7 * DAY)
    day = int(t_week // DAY)
    sec = t_week % DAY
    if day == 4 and sec >= 17 * 3600:
        return False
    if day == 5 and sec < 7 * 3600:
        return False
    if day == 5 and sec >= 17 * 3600:
        return False
    if day == 6 and sec < 7 * 3600:
        return False
    return True

def wait_until_open(env):
    while not is_open(env.now):
        yield env.timeout(60)

class Machine:
    def __init__(self, env, name, mean_proc, availability, mttr,
                 idle_energy, work_energy, stats):
        self.env = env
        self.name = name
        self.mean_proc = mean_proc
        self.mttr = mttr
        self.idle_energy = idle_energy
        self.work_energy = work_energy
        self.stats = stats

        self.resource = simpy.Resource(env, capacity=1)

        if availability >= 1.0:
            self.mttf = float("inf")
        elif availability <= 0.0:
            self.mttf = 0.0
        else:
            self.mttf = mttr * availability / (1.0 - availability)

        self.up = True
        self.proc_interrupt = env.event()

        env.process(self.failure_process())

    def failure_process(self):
        while True:
            if not self.up:
                yield self.env.timeout(1)
                continue
            ttf = random.expovariate(1.0 / self.mttf) if self.mttf > 0.0 and self.mttf < float("inf") else float("inf")
            yield self.env.timeout(ttf)
            if not is_open(self.env.now):
                continue
            self.up = False
            ttr = random.expovariate(1.0 / self.mttr)
            while ttr > 0:
                if is_open(self.env.now):
                    dt = min(60, ttr)
                else:
                    dt = 60
                yield self.env.timeout(dt)
                if is_open(self.env.now):
                    ttr -= dt
            self.up = True

    def process(self, part):
        with self.resource.request() as req:
            yield req
            yield self.env.process(self._run_job(part))

    def _run_job(self, part):
        yield from wait_until_open(self.env)

        start = self.env.now
        remaining = random.expovariate(1.0 / self.mean_proc)

        if self.env.now >= WARM_UP:
            self.stats['energy'] += self.idle_energy * 0

        while remaining > 0:
            if not is_open(self.env.now) or not self.up:
                yield self.env.timeout(60)
                continue
            dt = min(1.0, remaining)
            yield self.env.timeout(dt)
            remaining -= dt
            if self.env.now >= WARM_UP:
                self.stats['energy'] += self.work_energy * (dt / 3600.0)

        end = self.env.now
        if self.env.now >= WARM_UP:
            self.stats['machine_busy_time'][self.name] += (end - start)

class Buffer:
    def __init__(self, env, name, capacity, proc_time):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.proc_time = proc_time
        self.store = simpy.Store(env, capacity=capacity)

    def put(self, item):
        return self.store.put(item)

    def get_and_process(self):
        item = yield self.store.get()
        remaining = random.expovariate(1.0 / self.proc_time)
        while remaining > 0:
            if not is_open(self.env.now):
                yield self.env.timeout(60)
                continue
            dt = min(1.0, remaining)
            yield self.env.timeout(dt)
            remaining -= dt
        return item

def source(env, loading, post_loading_buffer, stats):
    i = 0
    while True:
        if is_open(env.now):
            inter = random.expovariate(1.0 / MEAN_INTERARRIVAL)
        else:
            inter = 60.0
        yield env.timeout(inter)
        if not is_open(env.now):
            continue
        i += 1
        part = {'id': i, 'birth': env.now}
        env.process(part_flow(env, part, loading, post_loading_buffer, stats))

def part_flow(env, part, loading, post_loading_buffer, stats):
    yield env.process(loading.process(part))
    yield post_loading_buffer.put(part)

    part = yield env.process(post_loading_buffer.get_and_process())
    yield env.process(stats['machines']['Conveyor'].process(part))

    pcb = stats['buffers']['PostConveyorBuffer']
    yield pcb.put(part)
    part = yield env.process(pcb.get_and_process())

    yield env.process(stats['machines']['Washing'].process(part))

    pwb = stats['buffers']['PostWashingBuffer']
    yield pwb.put(part)
    part = yield env.process(pwb.get_and_process())

    yield env.process(stats['machines']['Handling'].process(part))

    stats['press_counter'] += 1
    if stats['press_counter'] % 2 == 1:
        target_pre = stats['buffers']['PrePress1Buffer']
        press_machine = stats['machines']['Press1']
    else:
        target_pre = stats['buffers']['PrePress2Buffer']
        press_machine = stats['machines']['Press2']

    yield target_pre.put(part)
    part = yield env.process(target_pre.get_and_process())

    yield env.process(press_machine.process(part))

    postp = stats['buffers']['PostPress1_2Buffer']
    yield postp.put(part)
    part = yield env.process(postp.get_and_process())

    yield env.process(stats['machines']['Quality'].process(part))

    if random.random() < DEFECT_RATE:
        if env.now >= WARM_UP:
            stats['defects'] += 1
        return

    if env.now >= WARM_UP:
        stats['throughput'] += 1
        stats['flow_times'].append(env.now - part['birth'])

def run_replication(rep, buffers_config, seed):
    random.seed(seed)
    env = simpy.Environment()

    stats = {
        'throughput': 0,
        'defects': 0,
        'flow_times': [],
        'energy': 0.0,
        'machine_busy_time': {m: 0.0 for m in MACHINES.keys()},
        'press_counter': 0,
        'buffers': {},
        'machines': {}
    }

    for name, (cap, pt) in buffers_config.items():
        stats['buffers'][name] = Buffer(env, name, cap, pt)

    for name, (pt, avail, mttr, idle_e, work_e) in MACHINES.items():
        stats['machines'][name] = Machine(env, name, pt, avail, mttr,
                                          idle_e, work_e, stats)

    loading = stats['machines']['Loading']
    post_loading_buffer = stats['buffers']['PostLoadingBuffer']

    env.process(source(env, loading, post_loading_buffer, stats))

    env.run(until=SIM_TIME)

    run_time_hours = max(1.0, (SIM_TIME - WARM_UP) / 3600.0)
    throughput_rate = stats['throughput'] / run_time_hours
    wip = 0
    for buf in stats['buffers'].values():
        wip += len(buf.store.items)
    mean_energy_per_part = (stats['energy'] / stats['throughput']) if stats['throughput'] > 0 else 0.0

    return {
        'throughput_rate': throughput_rate,
        'wip': wip,
        'mean_energy_per_part': mean_energy_per_part
    }

def evaluate_single_individual(args):
    x, n_replications, base_seed = args

    caps = [int(v) for v in x]

    original_buffers = BUFFERS
    new_buffers = {}
    for (name, (_, pt)), cap in zip(original_buffers.items(), caps):
        new_buffers[name] = (cap, pt)

    throughputs = []
    wips = []

    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        res = run_replication(r, new_buffers, seed)
        throughputs.append(res['throughput_rate'])
        wips.append(res['wip'])

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return [avg_wip, -avg_throughput]

class BufferCapacityProblem(Problem):
    def __init__(self,
                 n_obj=2,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 n_replications=REPLICATIONS,
                 base_seed=RANDOM_SEED,
                 n_cores=50):
        self.buffer_names = list(BUFFERS.keys())
        n_var = len(self.buffer_names)

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
        X = np.asarray(X, dtype=int)
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
    n_var = len(BUFFERS.keys())

    problem = BufferCapacityProblem(
        n_obj=2,
        xl=np.array([1] * n_var, dtype=int),
        xu=np.array([10] * n_var, dtype=int),
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
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    buffer_names = list(BUFFERS.keys())
    fieldnames = ["generation", "individual_index"] + [f"{name}_capacity" for name in buffer_names] + ["wip", "throughput"]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")

        for ind_idx, (x, f) in enumerate(zip(X, F)):
            if f is None or any(np.isnan(f)):
                continue

            caps = [int(v) for v in x]
            wip = float(f[0])
            throughput = float(-f[1])

            row = {
                "generation": gen_idx,
                "individual_index": ind_idx,
                "wip": wip,
                "throughput": throughput
            }

            for name, cap in zip(buffer_names, caps):
                row[f"{name}_capacity"] = cap

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