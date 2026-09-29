import multiprocessing as mp
import random
import csv
import os
from functools import partial

# Import the simulation function from the existing simulation module
# from your_simulation_module import run_simulation, RANDOM_SEED
# Assuming run_simulation and RANDOM_SEED are available in the same namespace

POP_SIZE = 50
N_GEN = 100
N_CORES = 50

# Decision variables: capacities of all buffers (1-10, integers)
# Order: [raw_input, post_loading, post_conveyor, post_washing,
#         pre_press1_delay, pre_press2_delay, post_press12_delay,
#         pre_press1_store, pre_press2_store, press1_out_store, press2_out_store]
VAR_BOUNDS = [(1, 10)] * 11

def evaluate_individual(ind, seed_offset=0, runs=3):
    """
    Evaluate an individual by running the simulation multiple times and averaging.
    Objectives:
      - f1: WIP (to minimize)
      - f2: -Throughput (to minimize, since we want to maximize throughput)
    Constraint:
      - If any simulation run produces zero parts, treat as infeasible.
    """
    wip_vals = []
    thr_vals = []

    for r in range(runs):
        seed = RANDOM_SEED + seed_offset + r
        res = run_simulation_with_capacities(ind, seed)
        produced = res["overall"]["produced_parts"]
        if produced <= 0:
            # Infeasible
            return None
        wip_vals.append(res["overall"]["wip"])
        thr_vals.append(res["overall"]["throughput"])

    avg_wip = sum(wip_vals) / len(wip_vals)
    avg_thr = sum(thr_vals) / len(thr_vals)

    return (avg_wip, -avg_thr)


def run_simulation_with_capacities(ind, seed):
    """
    Wrapper around run_simulation that applies the individual's buffer capacities.
    This assumes that run_simulation is modified to accept a 'capacities' argument
    or that there is a global way to inject capacities before calling it.
    Here we assume a signature:
        run_simulation(seed, capacities=None)
    where capacities is a dict with the following keys:
        "raw_input", "post_loading", "post_conveyor", "post_washing",
        "pre_press1_delay", "pre_press2_delay", "post_press12_delay",
        "pre_press1_store", "pre_press2_store", "press1_out_store", "press2_out_store"
    """
    capacities = {
        "raw_input": ind[0],
        "post_loading": ind[1],
        "post_conveyor": ind[2],
        "post_washing": ind[3],
        "pre_press1_delay": ind[4],
        "pre_press2_delay": ind[5],
        "post_press12_delay": ind[6],
        "pre_press1_store": ind[7],
        "pre_press2_store": ind[8],
        "press1_out_store": ind[9],
        "press2_out_store": ind[10],
    }
    return run_simulation(seed, capacities=capacities)


def dominates(a, b):
    """
    Return True if objective vector a dominates b (strict Pareto dominance).
    Both a and b are tuples (f1, f2) to be minimized.
    """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(pop_objs):
    """
    Perform fast non-dominated sorting.
    pop_objs: list of objective tuples.
    Returns: list of fronts, each front is a list of indices.
    """
    S = [[] for _ in range(len(pop_objs))]
    n = [0] * len(pop_objs)
    rank = [0] * len(pop_objs)
    fronts = [[]]

    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if p == q:
                continue
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, rank


def crowding_distance(front, pop_objs):
    """
    Compute crowding distance for a front.
    front: list of indices
    pop_objs: list of objective tuples
    Returns: dict index -> distance
    """
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance

    num_obj = len(pop_objs[0])
    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda i: pop_objs[i][m])
        f_min = pop_objs[front_sorted[0]][m]
        f_max = pop_objs[front_sorted[-1]][m]
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        if f_max == f_min:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_f = pop_objs[front_sorted[k - 1]][m]
            next_f = pop_objs[front_sorted[k + 1]][m]
            distance[front_sorted[k]] += (next_f - prev_f) / (f_max - f_min)
    return distance


def tournament_selection(pop, pop_objs, k=2):
    """
    Binary tournament selection based on rank and crowding distance.
    pop: list of individuals
    pop_objs: list of objective tuples
    Returns: selected individual (deep copy not necessary for ints).
    """
    i, j = random.sample(range(len(pop)), 2)
    # Assume rank and crowding_distance are stored in attributes
    a = pop[i]
    b = pop[j]
    if a["rank"] < b["rank"]:
        return a["ind"]
    elif a["rank"] > b["rank"]:
        return b["ind"]
    else:
        if a["crowding"] > b["crowding"]:
            return a["ind"]
        else:
            return b["ind"]


def crossover(parent1, parent2, pc=0.9):
    """
    Single-point crossover for integer vectors.
    """
    if random.random() > pc:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    """
    Uniform mutation for integer variables within bounds.
    """
    for i, (low, up) in enumerate(VAR_BOUNDS):
        if random.random() < pm:
            ind[i] = random.randint(low, up)
    return ind


def init_population():
    pop = []
    for _ in range(POP_SIZE):
        ind = [random.randint(low, up) for (low, up) in VAR_BOUNDS]
        pop.append({"ind": ind, "objs": None, "rank": None, "crowding": None})
    return pop


def evaluate_population(pop, gen_seed_offset):
    """
    Evaluate all individuals in the population in parallel.
    Only feasible individuals (non-None objectives) are kept.
    """
    with mp.Pool(processes=N_CORES) as pool:
        func = partial(evaluate_individual, seed_offset=gen_seed_offset)
        inds = [p["ind"] for p in pop]
        results = pool.map(func, inds)

    new_pop = []
    for p, objs in zip(pop, results):
        if objs is not None:
            p["objs"] = objs
            new_pop.append(p)
    return new_pop


def assign_rank_and_crowding(pop):
    """
    Assign rank and crowding distance to population.
    """
    pop_objs = [p["objs"] for p in pop]
    fronts, rank = fast_non_dominated_sort(pop_objs)
    for i, p in enumerate(pop):
        p["rank"] = rank[i]
        p["crowding"] = 0.0

    for front in fronts:
        dist = crowding_distance(front, pop_objs)
        for i in front:
            pop[i]["crowding"] = dist[i]


def make_offspring(pop):
    """
    Create offspring population using selection, crossover, and mutation.
    """
    offspring = []
    while len(offspring) < POP_SIZE:
        parent1 = tournament_selection(pop, [p["objs"] for p in pop])
        parent2 = tournament_selection(pop, [p["objs"] for p in pop])
        c1, c2 = crossover(parent1, parent2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        offspring.append({"ind": c1, "objs": None, "rank": None, "crowding": None})
        if len(offspring) < POP_SIZE:
            offspring.append({"ind": c2, "objs": None, "rank": None, "crowding": None})
    return offspring


def environmental_selection(pop):
    """
    NSGA-II environmental selection to form next generation.
    """
    pop_objs = [p["objs"] for p in pop]
    fronts, rank = fast_non_dominated_sort(pop_objs)
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) <= POP_SIZE:
            for i in front:
                new_pop.append(pop[i])
        else:
            dist = crowding_distance(front, pop_objs)
            sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
            for i in sorted_front:
                if len(new_pop) < POP_SIZE:
                    new_pop.append(pop[i])
                else:
                    break
            break
    return new_pop


def run_moo(output_csv="moo_results.csv"):
    """
    Run NSGA-II for the given number of generations.
    Only feasible individuals are evaluated and stored.
    """
    random.seed(RANDOM_SEED)

    # Initialize population
    pop = init_population()

    # Evaluate initial population
    pop = evaluate_population(pop, gen_seed_offset=0)
    if not pop:
        return

    assign_rank_and_crowding(pop)

    # Prepare CSV
    header = [
        "gen",
        "ind_id",
        "raw_input",
        "post_loading",
        "post_conveyor",
        "post_washing",
        "pre_press1_delay",
        "pre_press2_delay",
        "post_press12_delay",
        "pre_press1_store",
        "pre_press2_store",
        "press1_out_store",
        "press2_out_store",
        "wip",
        "throughput"
    ]
    if os.path.exists(output_csv):
        os.remove(output_csv)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Log initial population
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for idx, p in enumerate(pop):
            ind = p["ind"]
            wip, neg_thr = p["objs"]
            row = [
                0,
                idx
            ] + ind + [
                wip,
                -neg_thr
            ]
            writer.writerow(row)

    # Generational loop
    for gen in range(1, N_GEN + 1):
        offspring = make_offspring(pop)
        offspring = evaluate_population(offspring, gen_seed_offset=gen * 10000)
        if not offspring:
            continue

        # Combine and select
        combined = pop + offspring
        assign_rank_and_crowding(combined)
        pop = environmental_selection(combined)

        # Log current population
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for idx, p in enumerate(pop):
                ind = p["ind"]
                wip, neg_thr = p["objs"]
                row = [
                    gen,
                    idx
                ] + ind + [
                    wip,
                    -neg_thr
                ]
                writer.writerow(row)


if __name__ == "__main__":
    run_moo()