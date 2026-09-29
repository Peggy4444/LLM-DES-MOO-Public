import simpy
import multiprocessing
import random
import statistics
import os

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv

# Ensure compatibility when integrated with the existing simulation code
try:
    RANDOM_SEED
except NameError:
    RANDOM_SEED = 33

try:
    WARMUP_SECONDS
except NameError:
    WARMUP_SECONDS = 86400

try:
    MEASURE_UNTIL
except NameError:
    SIM_TIME = 691200
    MEASURE_UNTIL = SIM_TIME

# The following simulation dependencies are expected to be provided by the existing
# simulation code when integrating. They must be available in the runtime environment.

def run_simulation(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Machine topology example including both serial and parallel section:
    #   M1  ->  M2  ->  [ M3 || M4 ]  ->  M5
    #
    # Serial segments:
    #   raw_input -> M1 -> buffer1 -> M2 -> buffer2
    #
    # Parallel segment (Pattern: splitter + separate pre/post buffers, depending on machine count):
    #   splitter(buffer2, preM3buffer, preM4buffer)
    #   preM3buffer -> M3 -> postM3buffer
    #   preM4buffer -> M4 -> postM4buffer
    #   add buffers if needed for more parallel machines
    #   merger(postM3buffer, postM4buffer) -> buffer3
    #
    # final serial segment:
    #   buffer3 -> M5 -> sink

    # Set changable parameters, in this case buffer capacities
    caps = list(caps)[:3]
    cap_buffer1, cap_buffer2, cap_buffer3 = caps

    #Buffers between serial machines
    buffer1 = DelayBuffer(env, cap=cap_buffer1, delay=10)  # between M1 and M2
    buffer2 = DelayBuffer(env, cap=cap_buffer2, delay=10)  # between M2 and parallel section
    buffer3 = DelayBuffer(env, cap=cap_buffer3, delay=10)  # between parallel section and M5

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000) # large to avoid starvation
    sink = simpy.Store(env, capacity=100000)   # final sink
    defects = simpy.Store(env, capacity=100000)  # defect sink

    # Helper stores for routing in parallel section
    # - Helper stores are simple simpy.Store, not DelayBuffer.
    # - They can be used:
    #     - as outputs of parallel machines before a merger, or
    #     - as temporary queues for splitters/mergers.
    # The capacity of helper stores can not be changed
    branch1_out = simpy.Store(env, capacity=2)  # output of M3
    branch2_out = simpy.Store(env, capacity=2)  # output of M4

    M1 = Machine(env, "M1", input_buffer=raw_input, output_buffer=buffer1,
        process_time=5, availability=97.79, mttr=74, 
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    M2 = Machine(env, "M2", input_buffer=buffer1, output_buffer=buffer2,
        process_time=20, availability=95.0, mttr=100,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    M3_parallel = Machine(env, "M3parallel", input_buffer=buffer2, output_buffer=branch1_out,
        process_time=15, availability=90.0, mttr=80,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    M4_parallel = Machine(env, "M4parallel", input_buffer=buffer2, output_buffer=branch2_out,
        process_time=15, availability=90.0, mttr=80,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    # Merge outputs of parallel machines into buffer3.
    merger(env, branch1_out, branch2_out, buffer3)

    M5 = Machine(env, "M5", input_buffer=buffer3, output_buffer=sink,
        process_time=25, availability=92.0, mttr=90,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
        defect_rate=0.089, defect_sink=defects,
    )

    machines_list = [M1, M2, M3_parallel, M4_parallel, M5]

    # Start part generation.
    env.process(part_generator(env, raw_input))

    # Run the model to fill pipelines/buffers and reach steady-state
    env.run(until=warmup)

    # Zero machine counters so everything after is measured stats
    for m in machines_list:
        reset_machine_stats(m)

    # Zero sinks for measured production counts
    produced_count_before = len(sink.items)

    wip_samples = []
    delay_buffers = [buffer1, buffer2, buffer3]

    def sample_wip(env):
        while True:
            # WIP definition: items in delay buffers + items in process
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)

            wip_samples.append(ready + in_transit + in_machines)
            yield env.timeout(60)

    env.process(sample_wip(env))

    env.run(until=measure_until)
    
    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0
 
    result = {"overall": {
            "throughput": throughput,
            "wip": avg_wip,
            "produced_parts":total_produced},
        "machine_energy": {}}
 
    for m in machines_list:
        waiting_energy = m.waiting_energy_consumption()
        working_energy = m.working_energy_consumption()
        total_energy = waiting_energy + working_energy
        result["machine_energy"][m.name] = {
            "working_time": m.working_time,
            "waiting_time": m.failed_time_total + m.blocked_time,
            "working_energy": working_energy,
            "waiting_energy": waiting_energy,
            "total_energy": total_energy}
        
    return result

def evaluate_single_individual(args):
    """
    Evaluates a single individual (one set of buffer capacities)
    over multiple simulation replications.
    """
    x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x[:3]]

    throughputs = []
    wips = []

    # Create a local random instance to prevent multiple CPU processes 
    # from pulling from the same global random state and generating identical seeds
    local_rng = random.Random()
    local_rng.seed(base_seed + sum(caps)) 

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        res = run_simulation(seed, caps, warmup, measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs)
    avg_wip = statistics.mean(wips)

    # Return the objectives: [f1 (wip), f2 (-throughput)]
    return [avg_wip, -avg_throughput]

class BufferCapacityProblem(Problem):

    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.
    Decision variables:
        x[0] = buffer1 cap
        x[1] = buffer2 cap
        x[2] = buffer3 cap

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(self, n_var=3, n_obj=2, n_constr=0,
                 xl=None, xu=None,
                 n_replications=5,
                 base_seed=RANDOM_SEED,
                 n_cores=1):
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

        # 1. Package the arguments for each individual
        
        tasks = [
            (X[i], self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL)
            for i in range(n_individuals)
        ]

        # 2. Map the tasks across all available CPU cores
        # Using a context manager (with) ensures processes are cleaned up properly
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_individual, tasks)

        # 3. Assign the compiled results matrix back to out["F"]
        out["F"] = np.array(results, dtype=float)

def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=RANDOM_SEED,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    """
    problem = BufferCapacityProblem(
        n_var=3,
        n_obj=2,
        xl=np.array([1, 1, 1]),
        xu=np.array([10, 10, 10]),
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
    """

    # 1. Define the target directory
    output_dir = "result"
    
    # 2. Safely create the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Join the directory name and the filename together
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "buffer1_cap",
        "buffer2_cap",
        "buffer3_cap",
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
            caps = [int(v) for v in x[:3]]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "buffer1_cap": caps[0],
                "buffer2_cap": caps[1],
                "buffer3_cap": caps[2],
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
    result = run_nsga2_optimization(pop_size=50, n_gen=50, n_replications=5, verbose=True, n_cores=50)

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")