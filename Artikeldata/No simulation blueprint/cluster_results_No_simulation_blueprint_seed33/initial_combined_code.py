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

RANDOM_SEED = 33
SIM_TIME = 691200        # total simulation time (s)
WARMUP = 86400           # warmup time (s)
REPLICATIONS = 10

# Shift pattern: production stops
# - Friday 17:00 to Saturday 07:00
# - Saturday 17:00 to Sunday 07:00
# Time origin: Monday 00:00 = t=0, week = 7*24*3600 = 604800 s
WEEK = 7 * 24 * 3600
FRI_17 = (4 * 24 + 17) * 3600
SAT_07 = (5 * 24 + 7) * 3600
SAT_17 = (5 * 24 + 17) * 3600
SUN_07 = (6 * 24 + 7) * 3600


def is_work_time(t):
    """Return True if time t (in seconds from week start) is in working period."""
    t_week = t % WEEK
    # two stop intervals: [Fri17, Sat07) and [Sat17, Sun07)
    if FRI_17 <= t_week < SAT_07:
        return False
    if SAT_17 <= t_week < SUN_07:
        return False
    return True


def wait_until_work_time(env):
    """If in stop period, wait until work resumes."""
    while not is_work_time(env.now):
        t = env.now % WEEK
        if FRI_17 <= t < SAT_07:
            yield env.timeout(SAT_07 - t)
        elif SAT_17 <= t < SUN_07:
            yield env.timeout(SUN_07 - t)
        else:
            break


class Machine:
    def __init__(self, env, name, avg_time, availability, mttr,
                 energy_idle, energy_work):
        self.env = env
        self.name = name
        self.avg_time = avg_time
        self.mtbf = self.calc_mtbf(availability, mttr)
        self.mttr = mttr
        self.energy_idle = energy_idle       # kW when idle (or available but not working)
        self.energy_work = energy_work       # kW when processing
        self.resource = simpy.Resource(env, capacity=1)

        # breakdown tracking
        self.working = True
        self.processes = []  # active processing coroutines that should be interrupted on breakdown
        self.break_proc = env.process(self.breakdown_process())

        # energy tracking
        self.last_state_change = env.now
        self.last_load = 0      # 0 idle, 1 busy (processing)
        self.energy_integral = 0.0  # kW * h (time-weighted integral, but time in seconds)
        self.env.process(self.energy_tracker())

    @staticmethod
    def calc_mtbf(avail, mttr):
        # Availability = MTBF / (MTBF + MTTR) -> MTBF = A*MTTR/(1-A)
        a = avail / 100.0
        if a >= 0.9999:
            return 1e12  # practically never fails
        return a * mttr / (1.0 - a)

    def energy_tracker(self):
        while True:
            yield self.env.timeout(60)  # integrate every minute
            now = self.env.now
            dt = now - self.last_state_change
            power = self.energy_idle
            if self.last_load > 0:
                power = self.energy_work
            # convert kW * (s) to kWh: multiply by dt/3600
            self.energy_integral += power * dt / 3600.0
            self.last_state_change = now

    def breakdown_process(self):
        while True:
            if self.mtbf > 1e9:
                return
            # time until next failure
            ttf = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(ttf)
            # breakdown occurs
            self.working = False
            # interrupt all active processing
            for p in list(self.processes):
                if not p.triggered and not p.processed:
                    p.proc.interrupt()
            # repair time
            ttr = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(ttr)
            self.working = True

    def process_part(self, part_id):
        """Context manager to use machine with breakdowns and energy measurement."""
        with self.resource.request() as req:
            yield req
            # mark as busy
            self.set_load(+1)
            while not self.working or not is_work_time(self.env.now):
                self.set_load(0)
                yield from wait_until_work_time(self.env)
                while not self.working:
                    yield self.env.timeout(1)
                self.set_load(+1)

            # actual processing with possible breakdowns
            proc = self.env.process(self._do_processing(part_id))
            wrapper = _ProcWrapper(proc)
            self.processes.append(wrapper)
            try:
                yield proc
            except simpy.Interrupt:
                # interrupted by breakdown
                while not self.working:
                    yield self.env.timeout(1)
                yield from wait_until_work_time(self.env)
                # resume full processing time anew
                proc = self.env.process(self._do_processing(part_id))
                wrapper2 = _ProcWrapper(proc)
                self.processes.append(wrapper2)
                try:
                    yield proc
                except simpy.Interrupt:
                    pass
                finally:
                    if wrapper2 in self.processes:
                        self.processes.remove(wrapper2)
            finally:
                if wrapper in self.processes:
                    self.processes.remove(wrapper)
                self.set_load(-1)

    def set_load(self, delta):
        # update integral to now with old state, then change load
        now = self.env.now
        dt = now - self.last_state_change
        power = self.energy_idle if self.last_load == 0 else self.energy_work
        self.energy_integral += power * dt / 3600.0
        self.last_state_change = now
        self.last_load += delta
        if self.last_load < 0:
            self.last_load = 0

    def _do_processing(self, part_id):
        pt = random.expovariate(1.0 / self.avg_time)
        remaining = pt
        try:
            start = self.env.now
            yield self.env.timeout(remaining)
        except simpy.Interrupt:
            raise


class _ProcWrapper:
    def __init__(self, proc):
        self.proc = proc
        self.triggered = False
        self.processed = False
        self.proc.callbacks.append(self._callback)

    def _callback(self, event):
        self.processed = True


class BufferWithTime:
    def __init__(self, env, name, capacity, proc_time):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.capacity = capacity
        self.proc_time = proc_time
        self.energy_idle = 0.0
        self.energy_work = 0.0
        self.last_state_change = env.now
        self.last_load = 0
        self.energy_integral = 0.0
        self.env.process(self.energy_tracker())

    def energy_tracker(self):
        while True:
            yield self.env.timeout(60)
            now = self.env.now
            dt = now - self.last_state_change
            power = self.energy_idle if self.last_load == 0 else self.energy_work
            self.energy_integral += power * dt / 3600.0
            self.last_state_change = now

    def put(self, item):
        return self.store.put(item)

    def get(self):
        return self.store.get()

    def hold(self):
        self.set_load(+1)
        yield self.env.timeout(self.proc_time)
        self.set_load(-1)

    def set_load(self, delta):
        now = self.env.now
        dt = now - self.last_state_change
        power = self.energy_idle if self.last_load == 0 else self.energy_work
        self.energy_integral += power * dt / 3600.0
        self.last_state_change = now
        self.last_load += delta
        if self.last_load < 0:
            self.last_load = 0


class ProductionSystem:
    def __init__(self, env, interarrival):
        self.env = env
        self.interarrival = interarrival

        # Stations
        self.loading_robot = Machine(env, "Loading robot",
                                     avg_time=12.0,
                                     availability=90.49,
                                     mttr=68.0,
                                     energy_idle=0.25,
                                     energy_work=0.72)
        self.conveyor = Machine(env, "Conveyor belt",
                                avg_time=6.0,
                                availability=100.0,
                                mttr=1.0,
                                energy_idle=0.0,
                                energy_work=0.0)
        self.washing = Machine(env, "Washing machine",
                               avg_time=14.0,
                               availability=80.89,
                               mttr=269.0,
                               energy_idle=4.28,
                               energy_work=35.24)
        self.hantering = Machine(env, "Hantering cell",
                                 avg_time=25.0,
                                 availability=97.79,
                                 mttr=74.0,
                                 energy_idle=0.50,
                                 energy_work=0.74)
        self.press1 = Machine(env, "Presses cell 1",
                              avg_time=175.0,
                              availability=87.79,
                              mttr=73.0,
                              energy_idle=1.25,
                              energy_work=1.28)
        self.press2 = Machine(env, "Presses cell 2",
                              avg_time=176.0,
                              availability=87.69,
                              mttr=74.0,
                              energy_idle=1.25,
                              energy_work=1.27)
        self.quality = Machine(env, "Quality station cell",
                               avg_time=41.0,
                               availability=85.87,
                               mttr=66.0,
                               energy_idle=0.58,
                               energy_work=0.84)

        # Buffers with given capacities and process times
        self.post_loading_buffer = BufferWithTime(env, "PostLoadingBuffer", 2, 10)
        self.post_conveyor_buffer = BufferWithTime(env, "PostConveyorBuffer", 2, 10)
        self.post_washing_buffer = BufferWithTime(env, "PostWashingBuffer", 2, 10)
        self.pre_press1_buffer = BufferWithTime(env, "PrePress1Buffer", 3, 32)
        self.pre_press2_buffer = BufferWithTime(env, "PrePress2Buffer", 3, 32)
        self.post_press_buffer = BufferWithTime(env, "PostPress1&Press2Buffer", 3, 32)

        # WIP tracking
        self.wip = 0
        self.wip_time_integral = 0.0
        self.last_wip_change = env.now
        env.process(self.wip_tracker())

        # results
        self.throughput_count = 0
        self.defect_count = 0
        self.completed_parts = 0

        # generator (source, infinite capacity implicit)
        env.process(self.source())

    def wip_tracker(self):
        while True:
            yield self.env.timeout(60)
            now = self.env.now
            dt = now - self.last_wip_change
            self.wip_time_integral += self.wip * dt
            self.last_wip_change = now

    def change_wip(self, delta):
        now = self.env.now
        dt = now - self.last_wip_change
        self.wip_time_integral += self.wip * dt
        self.last_wip_change = now
        self.wip += delta

    def source(self):
        i = 0
        while True:
            yield from wait_until_work_time(self.env)
            i += 1
            self.change_wip(+1)
            self.env.process(self.part_flow(i))
            ia = random.expovariate(1.0 / self.interarrival)
            yield self.env.timeout(ia)

    def part_flow(self, part_id):
        # Loading robot
        yield self.env.process(self.loading_robot.process_part(part_id))

        # buffer after loading
        yield self.post_loading_buffer.put(part_id)
        yield self.env.process(self.post_loading_buffer.hold())
        part_id = yield self.post_loading_buffer.get()

        # Conveyor
        yield self.env.process(self.conveyor.process_part(part_id))

        # buffer after conveyor
        yield self.post_conveyor_buffer.put(part_id)
        yield self.env.process(self.post_conveyor_buffer.hold())
        part_id = yield self.post_conveyor_buffer.get()

        # Washing machine
        yield self.env.process(self.washing.process_part(part_id))

        # buffer after washing
        yield self.post_washing_buffer.put(part_id)
        yield self.env.process(self.post_washing_buffer.hold())
        part_id = yield self.post_washing_buffer.get()

        # Hantering cell
        yield self.env.process(self.hantering.process_part(part_id))

        # split to two parallel press buffers evenly by part id
        if part_id % 2 == 0:
            # to press1 path
            yield self.pre_press1_buffer.put(part_id)
            yield self.env.process(self.pre_press1_buffer.hold())
            part_id = yield self.pre_press1_buffer.get()
            # Press1
            yield self.env.process(self.press1.process_part(part_id))
        else:
            # to press2 path
            yield self.pre_press2_buffer.put(part_id)
            yield self.env.process(self.pre_press2_buffer.hold())
            part_id = yield self.pre_press2_buffer.get()
            # Press2
            yield self.env.process(self.press2.process_part(part_id))

        # merge after presses
        yield self.post_press_buffer.put(part_id)
        yield self.env.process(self.post_press_buffer.hold())
        part_id = yield self.post_press_buffer.get()

        # Quality station
        yield self.env.process(self.quality.process_part(part_id))

        # defect decision at quality
        if random.random() < 0.089:
            self.defect_count += 1  # goes to defect sink
        else:
            self.throughput_count += 1
        self.completed_parts += 1
        self.change_wip(-1)


def run_replication(seed):
    random.seed(seed)
    env = simpy.Environment()
    # Assume interarrival chosen such that system has work; not specified, so set to 60 s
    system = ProductionSystem(env, interarrival=60.0)
    env.run(until=SIM_TIME)

    # KPIs after warmup
    effective_time = SIM_TIME - WARMUP
    # throughput in parts/hour
    # approximate: assume throughput during full run, ignore warmup removal at part-level
    th = system.throughput_count * 3600.0 / SIM_TIME

    # WIP average during full run
    avg_wip = system.wip_time_integral / SIM_TIME

    # energy: sum of all stations and buffers
    total_energy = (
        system.loading_robot.energy_integral +
        system.conveyor.energy_integral +
        system.washing.energy_integral +
        system.hantering.energy_integral +
        system.press1.energy_integral +
        system.press2.energy_integral +
        system.quality.energy_integral +
        system.post_loading_buffer.energy_integral +
        system.post_conveyor_buffer.energy_integral +
        system.post_washing_buffer.energy_integral +
        system.pre_press1_buffer.energy_integral +
        system.pre_press2_buffer.energy_integral +
        system.post_press_buffer.energy_integral
    )  # kWh

    if system.throughput_count > 0:
        energy_per_part = total_energy / system.throughput_count
    else:
        energy_per_part = 0.0

    return th, avg_wip, energy_per_part


def run_simulation(seed, caps):
    """
    Run one full simulation replication for a given set of buffer capacities.

    caps: list/array of 6 integers in [1,10] for:
        0: post_loading_buffer
        1: post_conveyor_buffer
        2: post_washing_buffer
        3: pre_press1_buffer
        4: pre_press2_buffer
        5: post_press_buffer
    """
    random.seed(seed)
    env = simpy.Environment()

    # Create system with default interarrival
    system = ProductionSystem(env, interarrival=60.0)

    # Override buffer capacities with decision variables
    caps = list(caps)[:6]
    (
        cap_post_loading,
        cap_post_conveyor,
        cap_post_washing,
        cap_pre_press1,
        cap_pre_press2,
        cap_post_press,
    ) = caps

    system.post_loading_buffer.capacity = int(cap_post_loading)
    system.post_loading_buffer.store.capacity = int(cap_post_loading)

    system.post_conveyor_buffer.capacity = int(cap_post_conveyor)
    system.post_conveyor_buffer.store.capacity = int(cap_post_conveyor)

    system.post_washing_buffer.capacity = int(cap_post_washing)
    system.post_washing_buffer.store.capacity = int(cap_post_washing)

    system.pre_press1_buffer.capacity = int(cap_pre_press1)
    system.pre_press1_buffer.store.capacity = int(cap_pre_press1)

    system.pre_press2_buffer.capacity = int(cap_pre_press2)
    system.pre_press2_buffer.store.capacity = int(cap_pre_press2)

    system.post_press_buffer.capacity = int(cap_post_press)
    system.post_press_buffer.store.capacity = int(cap_post_press)

    env.run(until=SIM_TIME)

    # KPIs over full run (consistent with provided simulation code)
    th = system.throughput_count * 3600.0 / SIM_TIME
    avg_wip = system.wip_time_integral / SIM_TIME

    return {
        "throughput": th,
        "wip": avg_wip,
        "produced_parts": system.throughput_count,
    }


def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.

    Returns objectives [f1, f2] = [avg_wip, -avg_throughput].
    """
    x, n_replications, base_seed = args
    caps = [int(v) for v in x[:6]]

    throughputs = []
    wips = []

    # Local RNG to decorrelate seeds across processes
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps))

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1_000_000)
        res = run_simulation(seed, caps)
        throughputs.append(res["throughput"])
        wips.append(res["wip"])

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return [avg_wip, -avg_throughput]


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integer in [1,10]):
        x[0] = post_loading_buffer capacity
        x[1] = post_conveyor_buffer capacity
        x[2] = post_washing_buffer capacity
        x[3] = pre_press1_buffer capacity
        x[4] = pre_press2_buffer capacity
        x[5] = post_press_buffer capacity

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(
        self,
        n_var=6,
        n_obj=2,
        n_constr=0,
        xl=None,
        xu=None,
        n_replications=REPLICATIONS,
        base_seed=RANDOM_SEED,
        n_cores=50,
    ):
        if xl is None:
            xl = np.array([1] * n_var, dtype=int)
        if xu is None:
            xu = np.array([10] * n_var, dtype=int)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
        )

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
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        out["F"] = np.array(results, dtype=float)


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=REPLICATIONS,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50,
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
        n_cores=n_cores,
    )

    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(prob=1.0 / problem.n_var, eta=20)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=base_seed,
        save_history=True,
        verbose=verbose,
    )

    return res


def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export all solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    Only feasible points are exported (all here are feasible since bounds are enforced).
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "generation_index",
        "individual_index",
        "post_loading_buffer_capacity",
        "post_conveyor_buffer_capacity",
        "post_washing_buffer_capacity",
        "pre_press1_buffer_capacity",
        "pre_press2_buffer_capacity",
        "post_press_buffer_capacity",
        "wip",
        "throughput",
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
            throughput = float(-f[1])

            # All solutions are feasible due to enforced bounds; no additional constraints to check.
            row = {
                "generation_index": gen_idx,
                "individual_index": ind_idx,
                "post_loading_buffer_capacity": caps[0],
                "post_conveyor_buffer_capacity": caps[1],
                "post_washing_buffer_capacity": caps[2],
                "pre_press1_buffer_capacity": caps[3],
                "pre_press2_buffer_capacity": caps[4],
                "post_press_buffer_capacity": caps[5],
                "wip": wip,
                "throughput": throughput,
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization on the production line simulation model
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=REPLICATIONS,
        verbose=True,
        n_cores=50,
    )

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")