import multiprocessing as mp
import random
import csv
import os
from functools import partial

# Assumes run_simulation is imported from the simulation module:
# from simulation_module import run_simulation, RANDOM_SEED, WARMUP_SECONDS, MEASURE_UNTIL

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

# Constraint: example total capacity limit (can be adjusted as needed)
MAX_TOTAL_CAPACITY = 40


def is_feasible(individual):
    total_cap = sum(individual)
    return total_cap <= MAX_TOTAL_CAPACITY


def evaluate_individual(individual, base_seed, warmup, measure_until):
    if not is_feasible(individual):
        return None

    # Map capacities to a configuration dict
    config = dict(zip(BUFFER_NAMES, individual))

    # We need a wrapper around run_simulation that accepts capacities.
    # It is assumed that the user will modify run_simulation to accept a
    # "buffer_caps" dict and use it when constructing DelayBuffer instances.
    seed = base_seed + random.randint(0, 10_000_000)
    res = run_simulation_with_caps(seed, config, warmup, measure_until)

    throughput = res["overall"]["throughput"]
    wip = res["overall"]["wip"]

    # Multi-objective: (wip to minimize, -throughput to minimize)
    return (wip, -throughput)


def run_simulation_with_caps(seed, buffer_caps, warmup, measure_until):
    # This function wraps the original run_simulation, assuming it has been
    # extended to accept a "buffer_caps" argument. If not, the user should
    # modify run_simulation accordingly.
    return run_simulation(seed=seed, warmup=warmup, measure_until=measure_until, buffer_caps=buffer_caps)


def dominates(a_obj, b_obj):
    return all(a <= b for a, b in zip(a_obj, b_obj)) and any(a < b for a, b in zip(a_obj, b_obj))


def fast_non_dominated_sort(pop_objs):
    S = [[] for _ in range(len(pop_objs))]
    n = [0 for _ in range(len(pop_objs))]
    rank = [0 for _ in range(len(pop_objs))]
    fronts = [[]]

    for p in range(len(pop_objs)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop_objs)):
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
    distance = [0.0 for _ in front]
    n_obj = len(pop_objs[0])
    for m in range(n_obj):
        obj_values = [pop_objs[i][m] for i in front]
        sorted_idx = sorted(range(len(front)), key=lambda k: obj_values[k])
        distance[sorted_idx[0]] = float("inf")
        distance[sorted_idx[-1]] = float("inf")
        if obj_values[sorted_idx[-1]] == obj_values[sorted_idx[0]]:
            continue
        for k in range(1, len(front) - 1):
            prev_val = obj_values[sorted_idx[k - 1]]
            next_val = obj_values[sorted_idx[k + 1]]
            distance[sorted_idx[k]] += (next_val - prev_val) / (obj_values[sorted_idx[-1]] - obj_values[sorted_idx[0]])
    return distance


def tournament_selection(pop, pop_objs, k=2):
    selected = []
    for _ in range(len(pop)):
        a, b = random.sample(range(len(pop)), k)
        if dominates(pop_objs[a], pop_objs[b]):
            selected.append(pop[a])
        elif dominates(pop_objs[b], pop_objs[a]):
            selected.append(pop[b])
        else:
            selected.append(pop[a] if random.random() < 0.5 else pop[b])
    return selected


def crossover(parent1, parent2, cx_prob=0.9):
    if random.random() > cx_prob:
        return parent1[:], parent2[:]
    point = random.randint(1, N_VARS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual, mut_prob=0.1):
    for i in range(N_VARS):
        if random.random() < mut_prob:
            individual[i] = random.randint(VAR_MIN, VAR_MAX)
    return individual


def create_individual():
    while True:
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
        if is_feasible(ind):
            return ind


def evaluate_population(pop, base_seed, warmup, measure_until, pool):
    eval_func = partial(evaluate_individual, base_seed=base_seed, warmup=warmup, measure_until=measure_until)
    results = pool.map(eval_func, pop)
    new_pop = []
    new_objs = []
    for ind, obj in zip(pop, results):
        if obj is not None:
            new_pop.append(ind)
            new_objs.append(obj)
    return new_pop, new_objs


def nsga2_optimization(output_csv_path, base_seed=RANDOM_SEED, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    with mp.Pool(processes=N_CORES) as pool, open(output_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = BUFFER_NAMES + ["wip", "throughput"]
        writer.writerow(header)

        pop = [create_individual() for _ in range(POP_SIZE)]
        pop, pop_objs = evaluate_population(pop, base_seed, warmup, measure_until, pool)

        for gen in range(N_GEN):
            if not pop:
                break

            fronts, rank = fast_non_dominated_sort(pop_objs)
            new_pop = []
            while len(new_pop) < POP_SIZE:
                offspring = tournament_selection(pop, pop_objs)
                children = []
                for i in range(0, len(offspring), 2):
                    if i + 1 < len(offspring):
                        c1, c2 = crossover(offspring[i], offspring[i + 1])
                        children.append(mutate(c1))
                        children.append(mutate(c2))
                    else:
                        children.append(mutate(offspring[i]))
                new_pop.extend(children)
            new_pop = new_pop[:POP_SIZE]

            combined_pop = pop + new_pop
            combined_pop, combined_objs = evaluate_population(combined_pop, base_seed, warmup, measure_until, pool)

            if not combined_pop:
                break

            fronts, rank = fast_non_dominated_sort(combined_objs)
            next_pop = []
            next_objs = []

            for front in fronts:
                if len(next_pop) + len(front) > POP_SIZE:
                    distances = crowding_distance(front, combined_objs)
                    sorted_front = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
                    for idx in sorted_front:
                        if len(next_pop) >= POP_SIZE:
                            break
                        p_idx = front[idx]
                        next_pop.append(combined_pop[p_idx])
                        next_objs.append(combined_objs[p_idx])
                    break
                else:
                    for p_idx in front:
                        next_pop.append(combined_pop[p_idx])
                        next_objs.append(combined_objs[p_idx])

            pop = next_pop
            pop_objs = next_objs

            for ind, obj in zip(pop, pop_objs):
                wip = obj[0]
                throughput = -obj[1]
                row = ind + [wip, throughput]
                writer.writerow(row)
            csvfile.flush()


if __name__ == "__main__":
    nsga2_optimization(output_csv_path="moo_results.csv")