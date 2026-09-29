import multiprocessing
import random
import statistics
import os
import math

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv

# Assumes the simulation model (DelayBuffer, Machine, splitter, merger, part_generator,
# production_wait_time, WARMUP_SECONDS, MEASURE_UNTIL, RANDOM_SEED, run_simulation implementation
# or helper primitives) are available in the same execution environment / module. This MOO file
# only defines the optimizer and a wrapper run_simulation that constructs the production line
# using those existing primitives.

# Wrapper simulation builder that uses the existing DelayBuffer, Machine, splitter, merger,
# part_generator, kwh_per_sec helpers available in the simulation module. This function
# expects those symbols to be defined (e.g., by placing this MOO code in the same file
# below the simulation model) or imported prior to running the optimizer.
def run_simulation_with_caps(seed, caps, warmup=None, measure_until=None):
    """
    Wrapper that builds the full production model using the provided buffer capacities (list of 6 ints)
    and then runs the simulation. Returns the same result dict structure as the simulation.
    This function uses the Machine, DelayBuffer, splitter, merger, part_generator, kwh_per_sec,
    WARMUP_SECONDS and MEASURE_UNTIL symbols which must exist in the environment where this code is run.
    """
    # Local imports from the simulation namespace - assumes they are available.
    # If this file is placed below the simulation code, these names will be in global scope.
    global DelayBuffer, Machine, splitter, merger, part_generator, kwh_per_sec
    global WARMUP_SECONDS, MEASURE_UNTIL

    if warmup is None:
        warmup = WARMUP_SECONDS
    if measure_until is None:
        measure_until = MEASURE_UNTIL

    random.seed(seed)
    # Build model using the same topology and parameters as the provided simulation model,
    # but replace the six delay buffer capacities with values from `caps`.
    # Expect caps to be sequence-like with 6 integer entries:
    #   post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
    #   pre_press1_buffer, pre_press2_buffer, post_press12_buffer

    caps = list(caps)
    cap_post_loading = int(caps[0])
    cap_post_conveyor = int(caps[1])
    cap_post_washing = int(caps[2])
    cap_pre_press1 = int(caps[3])
    cap_pre_press2 = int(caps[4])
    cap_post_press12 = int(caps[5])

    # The remainder of the model construction mirrors the original simulation model.
    # We construct a fresh environment and create the machines and buffers with the
    # provided capacities. This function relies on the existing implementation of
    # DelayBuffer, Machine, splitter, merger, part_generator, kwh_per_sec to be present.

    # Create a new simpy environment via the Machine / DelayBuffer constructors expecting one.
    # The Machine and DelayBuffer constructors require a simpy.Environment object; the
    # original run_simulation has that creation inside itself. To avoid duplicating simpy
    # here we re-create the environment by calling DelayBuffer / Machine constructors
    # which will implicitly require access to simpy.Environment in their scope.
    # The simplest approach (and compatible with integrating this MOO file into the same
    # script as the simulation) is to call the original run_simulation builder but supplying
    # the caps -- however because the original run_simulation is not parameterized we rebuild
    # the model here using the same primitives.

    # Create environment and other objects by invoking the same constructors as the simulation.
    # To avoid re-importing simpy here we rely on the constructors to create environments themselves
    # if required. Many model primitives expect a simpy.Environment to be passed in; therefore we
    # create one through the Machine/DelayBuffer/part_generator constructors by importing simpy.
    import simpy
    env = simpy.Environment()

    # Create buffers (DelayBuffer requires env, cap, delay)
    post_loading_buffer = DelayBuffer(env, cap=cap_post_loading, delay=10)
    post_conveyor_buffer = DelayBuffer(env, cap=cap_post_conveyor, delay=10)
    post_washing_buffer = DelayBuffer(env, cap=cap_post_washing, delay=10)
    pre_press1_buffer = DelayBuffer(env, cap=cap_pre_press1, delay=32)
    pre_press2_buffer = DelayBuffer(env, cap=cap_pre_press2, delay=32)
    post_press12_buffer = DelayBuffer(env, cap=cap_post_press12, delay=32)

    # Stores and helpers
    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)
    pre_press1 = simpy.Store(env, capacity=3)
    pre_press2 = simpy.Store(env, capacity=3)
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    # Instantiate machines with same parameters as in the provided simulation code
    loading_robot = Machine(
        env, "Loading robot", input_buffer=raw_input, output_buffer=post_loading_buffer,
        process_time=12.0, availability=90.49, mttr=68.0,
        working_power=kwh_per_sec(0.72), waiting_power=kwh_per_sec(0.25),
    )

    conveyor_belt = Machine(
        env, "Conveyor belt", input_buffer=post_loading_buffer, output_buffer=post_conveyor_buffer,
        process_time=6.0, availability=100.0, mttr=1.0,
        working_power=kwh_per_sec(0.0), waiting_power=kwh_per_sec(0.0),
    )

    washing_machine = Machine(
        env, "Washing machine", input_buffer=post_conveyor_buffer, output_buffer=post_washing_buffer,
        process_time=14.0, availability=80.89, mttr=269.0,
        working_power=kwh_per_sec(35.24), waiting_power=kwh_per_sec(4.28),
    )

    hantering_cell = Machine(
        env, "Hantering cell", input_buffer=post_washing_buffer, output_buffer=pre_press1,
        process_time=25.0, availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74), waiting_power=kwh_per_sec(0.50),
    )

    # Split stream from pre_press1 into two pre-press delay buffers
    env.process(splitter(env, pre_press1, pre_press1_buffer, pre_press2_buffer))

    presses_cell1 = Machine(
        env, "Presses cell 1", input_buffer=pre_press1_buffer, output_buffer=press1_out,
        process_time=175.0, availability=87.79, mttr=73.0,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    presses_cell2 = Machine(
        env, "Presses cell 2", input_buffer=pre_press2_buffer, output_buffer=press2_out,
        process_time=176.0, availability=87.69, mttr=74.0,
        working_power=kwh_per_sec(1.27), waiting_power=kwh_per_sec(1.25),
    )

    merger(env, press1_out, press2_out, post_press12_buffer)

    quality_station = Machine(
        env, "Quality station cell", input_buffer=post_press12_buffer, output_buffer=sink,
        process_time=41.0, availability=85.87, mttr=66.0,
        working_power=kwh_per_sec(0.84), waiting_power=kwh_per_sec(0.58),
        defect_rate=0.089, defect_sink=defects,
    )

    machines_list = [
        loading_robot, conveyor_belt, washing_machine,
        hantering_cell, presses_cell1, presses_cell2, quality_station,
    ]

    # Start part generation
    env.process(part_generator(env, raw_input))

    # Run warmup
    env.run(until=warmup)

    # Reset machine stats
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    delay_buffers = [
        post_loading_buffer, post_conveyor_buffer, post_washing_buffer,
        pre_press1_buffer, pre_press2_buffer, post_press12_buffer
    ]

    def sample_wip(env_inner):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env_inner.timeout(600)

    env.process(sample_wip(env))
    env.run(until=measure_until)

    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0

    result = {"overall": {
        "throughput": throughput,
        "wip": avg_wip,
        "produced_parts": total_produced},
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
    args: (index, x, n_replications, base_seed, warmup, measure_until)
    Returns (index, [avg_wip, -avg_throughput]) or (index, [np.nan, np.nan]) if infeasible.
    """
    idx, x, n_replications, base_seed, warmup, measure_until = args
    caps = [int(v) for v in x]

    # Feasibility check: ensure integer and within 1..10
    feasible = True
    for c in caps:
        if not (isinstance(c, (int, np.integer)) or (isinstance(c, float) and float(c).is_integer())):
            feasible = False
            break
        if c < 1 or c > 10:
            feasible = False
            break

    if not feasible:
        return (idx, [np.nan, np.nan])

    throughputs = []
    wips = []

    local_rng = random.Random()
    # seed derivation depends on caps to ensure variety across different designs
    local_rng.seed(base_seed + sum(caps) + idx)

    for r in range(n_replications):
        seed = base_seed + r + local_rng.randint(0, 1000000)
        res = run_simulation_with_caps(seed, caps, warmup, measure_until)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0

    return (idx, [avg_wip, -avg_throughput])


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for all six buffer capacities using NSGA-II.
    Decision variables:
        x[0] = post_loading_buffer cap
        x[1] = post_conveyor_buffer cap
        x[2] = post_washing_buffer cap
        x[3] = pre_press1_buffer cap
        x[4] = pre_press2_buffer cap
        x[5] = post_press12_buffer cap

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)

    Feasibility:
        Each capacity must be integer in [1, 10]. Infeasible individuals will receive NaN objectives
        and a positive constraint value (so they will be treated as infeasible by pymoo).
    """

    def __init__(self, n_var=6, n_obj=2, n_constr=1,
                 xl=None, xu=None,
                 n_replications=3,
                 base_seed=0,
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

        # Prepare output arrays
        F = np.full((n_individuals, self.n_obj), np.nan, dtype=float)
        G = np.zeros((n_individuals, self.n_constr), dtype=float)

        # Determine feasible individuals (bounds are provided by pymoo but we re-check)
        feasible_indices = []
        tasks = []
        for i in range(n_individuals):
            xi = X[i]
            caps = [int(round(v)) for v in xi]
            # Check integer and bounds (1..10)
            is_feasible = all((1 <= c <= 10) for c in caps)
            if not is_feasible:
                # Mark constraint violation > 0 (infeasible)
                G[i, 0] = 1.0
                F[i, :] = np.nan
            else:
                feasible_indices.append(i)
                tasks.append((i, caps, self.n_replications, self.base_seed, WARMUP_SECONDS, MEASURE_UNTIL))

        if tasks:
            # Run evaluations in parallel using exactly self.n_cores processes
            # Use a Pool context manager to ensure clean termination
            with multiprocessing.Pool(processes=self.n_cores) as pool:
                results = pool.map(evaluate_single_individual, tasks)

            # Fill F with results
            for idx, vals in results:
                if vals is None:
                    F[idx, :] = np.nan
                    G[idx, 0] = 1.0
                else:
                    # If evaluator returns NaN objectives, treat as infeasible
                    if math.isnan(vals[0]) or math.isnan(vals[1]):
                        F[idx, :] = np.array([np.nan, np.nan], dtype=float)
                        G[idx, 0] = 1.0
                    else:
                        F[idx, :] = np.array(vals, dtype=float)
                        G[idx, 0] = 0.0

        out["F"] = F
        out["G"] = G


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=3,
    base_seed=0,
    verbose=True,
    n_cores=50
):
    """
    Run NSGA-II on the buffer capacity optimization problem.
    pop_size and n_gen are set by caller (we will use pop_size=50, n_gen=50).
    This function configures the problem and performs the optimization.
    """

    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        n_constr=1,
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
    with their KPIs and decision variables to a CSV file. Infeasible individuals (G>0
    or NaN objectives) are skipped and not written to the CSV per user request.
    """
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "gen",
        "ind",
        "post_loading_buffer",
        "post_conveyor_buffer",
        "post_washing_buffer",
        "pre_press1_buffer",
        "pre_press2_buffer",
        "post_press12_buffer",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G") if pop.has("G") else None

        for ind_idx, (x, f_row) in enumerate(zip(X, F)):
            # Skip infeasible or NaN objective rows
            if G is not None and G[ind_idx, 0] > 0:
                continue
            if np.any(np.isnan(f_row)):
                continue

            caps = [int(round(v)) for v in x[:6]]
            wip = float(f_row[0])
            throughput = float(-f_row[1])  # stored as -throughput in objectives
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "post_loading_buffer": caps[0],
                "post_conveyor_buffer": caps[1],
                "post_washing_buffer": caps[2],
                "pre_press1_buffer": caps[3],
                "pre_press2_buffer": caps[4],
                "post_press12_buffer": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Configure and run NSGA-II with population size 50, 50 generations, and exactly 50 cores.
    POP_SIZE = 50
    N_GEN = 50
    N_REPLICATIONS = 3  # per-individual simulation replications (adjust for runtime/variance)
    BASE_SEED = 77
    N_CORES = 50  # exactly 50 cores as requested

    result = run_nsga2_optimization(
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        n_replications=N_REPLICATIONS,
        base_seed=BASE_SEED,
        verbose=True,
        n_cores=N_CORES
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")