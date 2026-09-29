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

RANDOM_SEED = 22

SIM_TIME = 691200        # total simulation time [s]
WARMUP = 86400           # warm-up [s]
REPLICATIONS = 10

# --- Production calendar ---

def is_work_time(t):
    """Return True if time t (in seconds) is within working hours.
       Work stops from Friday 17:00 to Saturday 07:00 and Saturday 17:00 to Sunday 07:00.
       Assume week starts at Monday 00:00."""
    sec_per_day = 24 * 3600
    day = int(t // sec_per_day) % 7  # 0=Mon, ..., 4=Fri, 5=Sat, 6=Sun
    sec_of_day = t % sec_per_day
    h = sec_of_day // 3600
    # Base working hours: 07:00–17:00
    base_work = (7 <= h < 17)
    # Weekend stops:
    # Friday (4) after 17:00 until Saturday 07:00
    # Saturday (5) after 17:00 until Sunday 07:00 (but Sun is fully off)
    if day == 4 and h >= 17:
        return False
    if day == 5:
        return 7 <= h < 17  # Sat 07:00–17:00 only
    if day == 6:
        return False        # Sun off
    return base_work


def wait_until_work_time(env):
    """Block until the calendar is in working time."""
    while not is_work_time(env.now):
        # sleep in 15-minute chunks
        yield env.timeout(900)


# --- Utility: machine with availability and failures ---

class Machine:
    def __init__(self, env, name, mean_proc_time, availability, mttr,
                 idle_power, work_power):
        self.env = env
        self.name = name
        self.mean_proc_time = mean_proc_time
        self.mtbf = self._calc_mtbf(availability, mttr)
        self.mttr = mttr
        self.idle_power = idle_power  # kW
        self.work_power = work_power  # kW
        self.resource = simpy.Resource(env, capacity=1)
        self.working = 0  # 1 if currently processing
        self.failed = False
        self.process = env.process(self._breakdown_process())
        self.power_log = []  # (t, power)

    def _calc_mtbf(self, A, mttr):
        # Availability A = MTBF / (MTBF + MTTR)
        # => MTBF = A*MTTR/(1-A)
        if A >= 1.0:
            return 1e12
        return A * mttr / (1.0 - A)

    def _breakdown_process(self):
        while True:
            # time to next failure
            ttf = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(ttf)
            if self.failed:
                continue
            # fail
            self.failed = True
            # down time
            ttr = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(ttr)
            self.failed = False

    def log_power(self, power):
        if self.power_log and self.power_log[-1][1] == power:
            return
        self.power_log.append((self.env.now, power))

    def start_work(self):
        self.working = 1
        self.log_power(self.work_power)

    def stop_work(self):
        self.working = 0
        self.log_power(self.idle_power)

    def current_power(self):
        return self.work_power if self.working else self.idle_power

    def energy_kwh(self):
        """Integrate power over time."""
        if not self.power_log:
            return 0.0
        energy = 0.0
        last_t, last_p = self.power_log[0]
        for t, p in self.power_log[1:]:
            dt = t - last_t
            energy += last_p * dt / 3600.0
            last_t, last_p = t, p
        # extend to end of simulation
        dt = SIM_TIME - last_t
        energy += last_p * dt / 3600.0
        return energy


# --- Buffers with capacity and process time (delay) ---

class Buffer:
    def __init__(self, env, name, capacity, proc_time):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.proc_time = proc_time

    def put(self, part):
        return self.store.put(part)

    def get(self):
        return self.store.get()

    def process_part(self, part):
        yield self.env.timeout(self.proc_time)


# --- Model for one replication (base version, not used by MOO but kept for completeness) ---

def run_replication(rep):

    random.seed(RANDOM_SEED + rep)

    env = simpy.Environment()

    # Machines
    conveyor = Machine(env, 'Conveyor belt', 6.0, 1.00, 1.0, 0.0, 0.0)
    handling = Machine(env, 'Hantering cell', 25.0, 0.9779, 74.0, 0.50, 0.74)
    loading = Machine(env, 'Loading robot', 12.0, 0.9049, 68.0, 0.25, 0.72)
    press1 = Machine(env, 'Presses cell 1', 175.0, 0.8779, 73.0, 1.25, 1.28)
    press2 = Machine(env, 'Presses cell 2', 176.0, 0.8769, 74.0, 1.25, 1.27)
    quality = Machine(env, 'Quality station cell', 41.0, 0.8587, 66.0, 0.58, 0.84)
    washing = Machine(env, 'Washing machine', 14.0, 0.8089, 269.0, 4.28, 35.24)

    machines = [conveyor, handling, loading, press1, press2, quality, washing]
    for m in machines:
        m.log_power(m.idle_power)

    # Buffers
    post_loading = Buffer(env, 'PostLoadingBuffer', capacity=2, proc_time=10)
    post_conveyor = Buffer(env, 'PostConveyorBuffer', capacity=2, proc_time=10)
    post_washing = Buffer(env, 'PostWashingBuffer', capacity=2, proc_time=10)
    pre_press1 = Buffer(env, 'PrePress1Buffer', capacity=3, proc_time=32)
    pre_press2 = Buffer(env, 'PrePress2Buffer', capacity=3, proc_time=32)
    post_press = Buffer(env, 'PostPress1&Press2Buffer', capacity=3, proc_time=32)

    # WIP tracking (excluding raw source)
    wip = 0
    wip_time = 0.0
    last_wip_change = 0.0

    completed_parts = 0
    defect_parts = 0
    defect_rate = 0.089

    def wip_increase():
        nonlocal wip, wip_time, last_wip_change
        now = env.now
        wip_time += wip * (now - last_wip_change)
        wip += 1
        last_wip_change = now

    def wip_decrease():
        nonlocal wip, wip_time, last_wip_change
        now = env.now
        wip_time += wip * (now - last_wip_change)
        wip -= 1
        last_wip_change = now

    # Source: infinite raw parts, modeled as arrival to loading robot
    interarrival = 20.0  # arbitrary; can be adjusted

    def source():
        i = 0
        while True:
            yield env.timeout(interarrival)
            i += 1
            env.process(part_process(i))

    def process_on_machine(part_id, machine, mean_proc_time):
        # wait for work time and machine availability (no processing during stops)
        with machine.resource.request() as req:
            yield req
            # wait for calendar open
            yield from wait_until_work_time(env)
            # processing with simple availability/repair: if machine fails during proc,
            # we just extend time (simplification: failures handled by downtime only)
            remaining = random.expovariate(1.0 / mean_proc_time)
            machine.start_work()
            while remaining > 0:
                # work only in work time and when not failed
                if not is_work_time(env.now) or machine.failed:
                    machine.stop_work()
                    yield env.timeout(60)  # check every 60 s
                    if not machine.working:
                        machine.log_power(machine.idle_power)
                    continue
                dt = min(60, remaining)
                yield env.timeout(dt)
                remaining -= dt
            machine.stop_work()

    def part_process(part_id):
        nonlocal completed_parts, defect_parts

        # enter system WIP after source
        wip_increase()

        # Loading robot
        yield from process_on_machine(part_id, loading, loading.mean_proc_time)

        # Conveyor belt
        yield from process_on_machine(part_id, conveyor, conveyor.mean_proc_time)

        # Washing machine
        yield from process_on_machine(part_id, washing, washing.mean_proc_time)

        # Hantering cell
        yield from process_on_machine(part_id, handling, handling.mean_proc_time)

        # Split evenly to presses via buffers (round-robin)
        if part_id % 2 == 0:
            buf = pre_press1
            target_press = press1
            press_mean = press1.mean_proc_time
        else:
            buf = pre_press2
            target_press = press2
            press_mean = press2.mean_proc_time

        # Put in pre-press buffer
        yield buf.put(part_id)
        yield from buf.process_part(part_id)
        yield buf.get()

        # Press (1 or 2)
        yield from process_on_machine(part_id, target_press, press_mean)

        # PostPress shared buffer
        yield post_press.put(part_id)
        yield from post_press.process_part(part_id)
        yield post_press.get()

        # Quality station
        yield from process_on_machine(part_id, quality, quality.mean_proc_time)

        # Defect decision (defect sink vs good sink)
        if random.random() < defect_rate:
            defect_parts += 1
            wip_decrease()
            return
        else:
            completed_parts += 1
            wip_decrease()
            return

    env.process(source())
    env.run(until=SIM_TIME)

    # KPIs excluding warm-up
    eff_time = SIM_TIME - WARMUP
    # throughput parts/hour
    th = (completed_parts / eff_time) * 3600.0
    # average WIP over time
    avg_wip = wip_time / SIM_TIME
    # energy
    total_energy = sum(m.energy_kwh() for m in machines)
    mean_energy_per_part = total_energy / max(completed_parts, 1)

    return th, avg_wip, mean_energy_per_part


# --- MOO-related simulation wrapper using variable buffer capacities ---

def run_simulation_for_caps(seed, caps, replications=REPLICATIONS):
    """
    Run the production line simulation for a given buffer capacity
    configuration over multiple replications and return average KPIs.
    Decision variables (all integer in [1,10]):
        caps[0] = post_loading capacity
        caps[1] = post_conveyor capacity
        caps[2] = post_washing capacity
        caps[3] = pre_press1 capacity
        caps[4] = pre_press2 capacity
        caps[5] = post_press capacity
    """

    def single_rep(rep, caps_local):
        random.seed(seed + rep)

        env = simpy.Environment()

        # Machines (same as in original run_replication)
        conveyor = Machine(env, 'Conveyor belt', 6.0, 1.00, 1.0, 0.0, 0.0)
        handling = Machine(env, 'Hantering cell', 25.0, 0.9779, 74.0, 0.50, 0.74)
        loading = Machine(env, 'Loading robot', 12.0, 0.9049, 68.0, 0.25, 0.72)
        press1 = Machine(env, 'Presses cell 1', 175.0, 0.8779, 73.0, 1.25, 1.28)
        press2 = Machine(env, 'Presses cell 2', 176.0, 0.8769, 74.0, 1.25, 1.27)
        quality = Machine(env, 'Quality station cell', 41.0, 0.8587, 66.0, 0.58, 0.84)
        washing = Machine(env, 'Washing machine', 14.0, 0.8089, 269.0, 4.28, 35.24)

        machines = [conveyor, handling, loading, press1, press2, quality, washing]
        for m in machines:
            m.log_power(m.idle_power)

        # Unpack capacities
        cap_post_loading, cap_post_conveyor, cap_post_washing, cap_pre_press1, cap_pre_press2, cap_post_press = caps_local

        # Buffers with decision-variable capacities
        post_loading = Buffer(env, 'PostLoadingBuffer', capacity=cap_post_loading, proc_time=10)
        post_conveyor = Buffer(env, 'PostConveyorBuffer', capacity=cap_post_conveyor, proc_time=10)
        post_washing = Buffer(env, 'PostWashingBuffer', capacity=cap_post_washing, proc_time=10)
        pre_press1 = Buffer(env, 'PrePress1Buffer', capacity=cap_pre_press1, proc_time=32)
        pre_press2 = Buffer(env, 'PrePress2Buffer', capacity=cap_pre_press2, proc_time=32)
        post_press = Buffer(env, 'PostPress1&Press2Buffer', capacity=cap_post_press, proc_time=32)

        # WIP tracking (excluding raw source)
        wip = 0
        wip_time = 0.0
        last_wip_change = 0.0

        completed_parts = 0
        defect_parts = 0
        defect_rate = 0.089

        def wip_increase():
            nonlocal wip, wip_time, last_wip_change
            now = env.now
            wip_time += wip * (now - last_wip_change)
            wip += 1
            last_wip_change = now

        def wip_decrease():
            nonlocal wip, wip_time, last_wip_change
            now = env.now
            wip_time += wip * (now - last_wip_change)
            wip -= 1
            last_wip_change = now

        interarrival = 20.0

        def source():
            i = 0
            while True:
                yield env.timeout(interarrival)
                i += 1
                env.process(part_process(i))

        def process_on_machine(part_id, machine, mean_proc_time):
            with machine.resource.request() as req:
                yield req
                yield from wait_until_work_time(env)
                remaining = random.expovariate(1.0 / mean_proc_time)
                machine.start_work()
                while remaining > 0:
                    if not is_work_time(env.now) or machine.failed:
                        machine.stop_work()
                        yield env.timeout(60)
                        if not machine.working:
                            machine.log_power(machine.idle_power)
                        continue
                    dt = min(60, remaining)
                    yield env.timeout(dt)
                    remaining -= dt
                machine.stop_work()

        def part_process(part_id):
            nonlocal completed_parts, defect_parts

            wip_increase()

            # Loading robot
            yield from process_on_machine(part_id, loading, loading.mean_proc_time)

            # PostLoading buffer
            yield post_loading.put(part_id)
            yield from post_loading.process_part(part_id)
            yield post_loading.get()

            # Conveyor belt
            yield from process_on_machine(part_id, conveyor, conveyor.mean_proc_time)

            # PostConveyor buffer
            yield post_conveyor.put(part_id)
            yield from post_conveyor.process_part(part_id)
            yield post_conveyor.get()

            # Washing machine
            yield from process_on_machine(part_id, washing, washing.mean_proc_time)

            # PostWashing buffer
            yield post_washing.put(part_id)
            yield from post_washing.process_part(part_id)
            yield post_washing.get()

            # Hantering cell
            yield from process_on_machine(part_id, handling, handling.mean_proc_time)

            # Split evenly to presses via buffers (round-robin)
            if part_id % 2 == 0:
                buf = pre_press1
                target_press = press1
                press_mean = press1.mean_proc_time
            else:
                buf = pre_press2
                target_press = press2
                press_mean = press2.mean_proc_time

            yield buf.put(part_id)
            yield from buf.process_part(part_id)
            yield buf.get()

            # Press (1 or 2)
            yield from process_on_machine(part_id, target_press, press_mean)

            # PostPress shared buffer
            yield post_press.put(part_id)
            yield from post_press.process_part(part_id)
            yield post_press.get()

            # Quality station
            yield from process_on_machine(part_id, quality, quality.mean_proc_time)

            # Defect decision
            if random.random() < defect_rate:
                defect_parts += 1
                wip_decrease()
                return
            else:
                completed_parts += 1
                wip_decrease()
                return

        env.process(source())
        env.run(until=SIM_TIME)

        eff_time = SIM_TIME - WARMUP
        th = (completed_parts / eff_time) * 3600.0 if eff_time > 0 else 0.0
        avg_wip = wip_time / SIM_TIME if SIM_TIME > 0 else 0.0

        return th, avg_wip

    throughputs = []
    wips = []

    for r in range(replications):
        th, wip = single_rep(r, caps)
        throughputs.append(th)
        wips.append(wip)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return avg_throughput, avg_wip


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    # Constraint handling:
    # Example constraint: total buffer capacity must be <= 40.
    # If violated, return a large penalty so that the point is dominated.
    if sum(caps) > 40:
        # Penalize: high WIP, very low throughput
        return [1e6, 1e6]

    # Local RNG for seed diversification
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    throughputs = []
    wips = []

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        avg_th, avg_wip = run_simulation_for_caps(seed, caps, replications=1)
        throughputs.append(avg_th)
        wips.append(avg_wip)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    # Objectives: minimize WIP, maximize throughput -> minimize -throughput
    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables (integer in [1,10]):
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

        # Use up to n_cores cores (do not exceed available CPUs)
        n_procs = min(self.n_cores, multiprocessing.cpu_count())
        with multiprocessing.Pool(processes=n_procs) as pool:
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
    Points that violate the constraint (sum of capacities > 40) are excluded.
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
        "post_press_capacity",
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
            # Apply the same constraint filter used in evaluation:
            if sum(caps) > 40:
                continue  # skip infeasible solution
            wip_val = float(f[0])
            throughput_val = float(-f[1])  # stored as -throughput in objectives
            row = {
                "generation": gen_idx,
                "individual_index": ind_idx,
                "post_loading_capacity": caps[0],
                "post_conveyor_capacity": caps[1],
                "post_washing_capacity": caps[2],
                "pre_press1_capacity": caps[3],
                "pre_press2_capacity": caps[4],
                "post_press_capacity": caps[5],
                "wip": wip_val,
                "throughput": throughput_val
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")