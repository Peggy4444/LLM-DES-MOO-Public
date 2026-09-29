import multiprocessing as mp
import random
import csv
import os
from functools import partial

# Import the simulation function from the existing simulation module
# from simulation_module import run_simulation, RANDOM_SEED  # Example import
# Assuming run_simulation and RANDOM_SEED are available in the same namespace

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
N_VARS = len(BUFFER_NAMES)
VAR_MIN = 1
VAR_MAX = 10

def run_simulation_with_buffers(seed, buffer_caps):
    # buffer_caps is a list of 6 integers in [1,10]
    # This function must call a modified version of run_simulation that accepts buffer capacities.
    # Here we assume run_simulation has been adapted to accept an optional buffer_caps argument.
    return run_simulation(seed, buffer_caps=buffer_caps)


def evaluate_individual(ind, base_seed):
    # Constraint: sum of all buffer capacities <= 30 (example constraint)
    # If violated, return None to indicate infeasible (do not evaluate)
    if sum(ind) > 30:
        return None

    runs = 3
    throughputs = []
    wips = []
    for i in range(runs):
        seed = base_seed + i
        res = run_simulation_with_buffers(seed, ind)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])
    avg_throughput = sum(throughputs) / runs
    avg_wip = sum(wips) / runs
    return avg_wip, -avg_throughput  # minimize wip, maximize throughput -> minimize -throughput


def dominates(a, b):
    # a and b are tuples (f1, f2), lower is better
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


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
    distance = {i: 0.0 for i in front}
    if len(front) == 0:
        return distance
    n_obj = len(pop_objs[0])
    for m in range(n_obj):
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
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None:
            best = i
        else:
            # Compare rank and crowding later; here we just use objective sum as a simple proxy
            if sum(pop_objs[i]) < sum(pop_objs[best]):
                best = i
    return pop[best]


def crossover(p1, p2, pc=0.9):
    if random.random() > pc:
        return p1[:], p2[:]
    point = random.randint(1, N_VARS - 1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    for i in range(N_VARS):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def nsga2_optimize():
    random.seed(RANDOM_SEED)
    base_seed = RANDOM_SEED + 10000

    # Initialize population
    population = []
    while len(population) < POP_SIZE:
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]
        if sum(ind) <= 30:
            population.append(ind)

    pool = mp.Pool(processes=N_CORES)
    eval_func = partial(evaluate_individual, base_seed=base_seed)

    results_file = "moo_results.csv"
    if os.path.exists(results_file):
        os.remove(results_file)
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = BUFFER_NAMES + ["wip", "throughput"]
        writer.writerow(header)

    for gen in range(N_GEN):
        # Evaluate population
        objs = pool.map(eval_func, population)
        valid_indices = [i for i, o in enumerate(objs) if o is not None]
        population = [population[i] for i in valid_indices]
        objs = [objs[i] for i in valid_indices]

        # Write valid individuals to CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            for ind, (wip, neg_throughput) in zip(population, objs):
                writer.writerow(ind + [wip, -neg_throughput])

        # NSGA-II selection
        fronts, rank = fast_non_dominated_sort(objs)
        new_population = []
        while len(new_population) < POP_SIZE:
            for front in fronts:
                if len(new_population) + len(front) > POP_SIZE:
                    dist = crowding_distance(front, objs)
                    sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                    for i in sorted_front:
                        if len(new_population) < POP_SIZE:
                            new_population.append(population[i])
                        else:
                            break
                    break
                else:
                    for i in front:
                        new_population.append(population[i])
            if len(new_population) >= POP_SIZE:
                break

        population = new_population

        # Create offspring
        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(population, objs)
            p2 = tournament_selection(population, objs)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            if sum(c1) <= 30:
                offspring.append(c1)
            if len(offspring) < POP_SIZE and sum(c2) <= 30:
                offspring.append(c2)

        population = offspring

    pool.close()
    pool.join()


if __name__ == "__main__":
    nsga2_optimize()