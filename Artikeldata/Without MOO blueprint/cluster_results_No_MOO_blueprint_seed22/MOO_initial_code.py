import multiprocessing as mp
import random
import csv
import os
from functools import partial

# Assumes run_simulation is imported from the simulation module:
# from simulation_module import run_simulation, RANDOM_SEED

POP_SIZE = 50
N_GEN = 100
N_CORES = 50

BUFFER_NAMES = [
    "post_loading_buffer",
    "post_conveyor_buffer",
    "post_washing_buffer",
    "pre_press1_buffer",
    "pre_press2_buffer",
    "post_press12_buffer",
]

VAR_MIN = 1
VAR_MAX = 10
N_VARS = len(BUFFER_NAMES)


def evaluate_individual(ind, seed_offset=0):
    """
    Evaluate one individual.
    ind: list of capacities for the 6 delay buffers in the same order as BUFFER_NAMES.
    Returns (throughput, wip) or None if constraint violated.
    """
    # Constraint: all capacities must be within [VAR_MIN, VAR_MAX]
    if any((c < VAR_MIN or c > VAR_MAX) for c in ind):
        return None

    # Run simulation with modified capacities.
    # We assume run_simulation can be adapted to accept capacities as an argument.
    # Here we wrap it by monkey-patching or by assuming a modified version:
    # run_simulation_with_caps(seed, caps_dict)
    caps_dict = {name: cap for name, cap in zip(BUFFER_NAMES, ind)}
    seed = RANDOM_SEED + seed_offset

    res = run_simulation_with_caps(seed, caps_dict)
    throughput = res["overall"]["throughput"]
    wip = res["overall"]["wip"]

    # Additional constraint example (optional): discard if throughput <= 0
    if throughput <= 0:
        return None

    return throughput, wip


def dominates(a_obj, b_obj):
    """
    Return True if a dominates b.
    a_obj, b_obj: (throughput, wip)
    Maximize throughput, minimize wip.
    """
    a_t, a_w = a_obj
    b_t, b_w = b_obj

    not_worse = (a_t >= b_t) and (a_w <= b_w)
    strictly_better = (a_t > b_t) or (a_w < b_w)
    return not_worse and strictly_better


def fast_non_dominated_sort(pop_objs):
    """
    NSGA-II fast non-dominated sort.
    pop_objs: list of objective tuples or None for infeasible.
    Returns: list of fronts, each front is list of indices.
    """
    S = [[] for _ in pop_objs]
    n = [0 for _ in pop_objs]
    fronts = [[]]

    for p in range(len(pop_objs)):
        if pop_objs[p] is None:
            continue
        for q in range(len(pop_objs)):
            if pop_objs[q] is None or p == q:
                continue
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(front, pop_objs):
    """
    Compute crowding distance for a front.
    front: list of indices
    pop_objs: list of (throughput, wip)
    Returns: dict index -> distance
    """
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance

    # For each objective
    for m in range(2):
        front_sorted = sorted(front, key=lambda i: pop_objs[i][m])
        f_min = pop_objs[front_sorted[0]][m]
        f_max = pop_objs[front_sorted[-1]][m]
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        if f_max == f_min:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_val = pop_objs[front_sorted[k - 1]][m]
            next_val = pop_objs[front_sorted[k + 1]][m]
            distance[front_sorted[k]] += (next_val - prev_val) / (f_max - f_min)
    return distance


def tournament_selection(pop, pop_objs, k=2):
    """
    Binary tournament selection based on rank and crowding distance.
    pop: list of individuals
    pop_objs: list of objective tuples
    Returns: selected individual (deep copy not needed for ints).
    """
    i, j = random.sample(range(len(pop)), 2)
    # Compare feasibility first
    obj_i = pop_objs[i]
    obj_j = pop_objs[j]
    if obj_i is None and obj_j is not None:
        return pop[j]
    if obj_j is None and obj_i is not None:
        return pop[i]
    if obj_i is None and obj_j is None:
        return pop[i]

    # If both feasible, use dominance
    if dominates(obj_i, obj_j):
        return pop[i]
    if dominates(obj_j, obj_i):
        return pop[j]

    # If neither dominates, pick randomly
    return pop[i] if random.random() < 0.5 else pop[j]


def crossover(parent1, parent2, pc=0.9):
    """
    One-point crossover for integer vectors.
    """
    if random.random() > pc or len(parent1) < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    """
    Uniform mutation for integer variables in [VAR_MIN, VAR_MAX].
    """
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def create_initial_population():
    pop = []
    for _ in range(POP_SIZE):
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
        pop.append(ind)
    return pop


def run_simulation_with_caps(seed, caps_dict):
    """
    Wrapper around run_simulation that applies custom capacities.
    This function assumes that run_simulation is modified to accept
    a 'caps_dict' argument, or that the simulation module reads these
    capacities from a global configuration.
    """
    # Example assuming run_simulation(seed, caps_dict=...) exists:
    return run_simulation(seed, caps_dict=caps_dict)


def evaluate_population(pop, pool, gen):
    """
    Evaluate all individuals in population using multiprocessing pool.
    Returns list of objective tuples or None for infeasible.
    """
    func = partial(evaluate_individual)
    # Use unique seed offsets per individual and generation
    tasks = [(ind, gen * POP_SIZE + i) for i, ind in enumerate(pop)]

    # Unpack because pool.map only passes one argument
    def wrapper(args):
        ind, seed_offset = args
        return evaluate_individual(ind, seed_offset)

    results = pool.map(wrapper, tasks)
    return results


def nsga2():
    random.seed(RANDOM_SEED)
    if N_CORES > mp.cpu_count():
        raise RuntimeError(f"Requested {N_CORES} cores, but only {mp.cpu_count()} available.")

    pop = create_initial_population()

    # Prepare CSV file
    csv_file = "moo_results.csv"
    write_header = not os.path.exists(csv_file)
    csv_f = open(csv_file, mode="w", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        header = ["gen", "idx"] + BUFFER_NAMES + ["throughput", "wip"]
        writer.writerow(header)

    with mp.Pool(processes=N_CORES) as pool:
        for gen in range(N_GEN):
            pop_objs = evaluate_population(pop, pool, gen)

            # Filter feasible individuals for logging and selection
            for idx, (ind, obj) in enumerate(zip(pop, pop_objs)):
                if obj is None:
                    continue
                throughput, wip = obj
                row = [gen, idx] + ind + [throughput, wip]
                writer.writerow(row)

            csv_f.flush()

            # NSGA-II selection
            fronts = fast_non_dominated_sort(pop_objs)
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) > POP_SIZE:
                    distances = crowding_distance(front, pop_objs)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    remaining = POP_SIZE - len(new_pop)
                    new_pop.extend([pop[i] for i in sorted_front[:remaining]])
                    break
                else:
                    new_pop.extend([pop[i] for i in front])

            # If due to infeasibility we have fewer than POP_SIZE, fill randomly
            while len(new_pop) < POP_SIZE:
                ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
                new_pop.append(ind)

            # Create offspring
            offspring = []
            while len(offspring) < POP_SIZE:
                p1 = tournament_selection(new_pop, pop_objs)
                p2 = tournament_selection(new_pop, pop_objs)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                if len(offspring) < POP_SIZE:
                    offspring.append(c2)

            pop = offspring

    csv_f.close()


if __name__ == "__main__":
    nsga2()