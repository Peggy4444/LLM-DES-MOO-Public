import multiprocessing as mp
import random
import csv
import os
from functools import partial

# Assumes run_simulation is imported from the simulation module:
# from simulation_module import run_simulation, RANDOM_SEED, REPLICATIONS

POP_SIZE = 50
N_GEN = 100
N_CORES = 50
RANDOM_SEED_MOO = 1234

# Decision variables: capacities of all buffers (1-10, integers)
# Order:
# 0: post_loading_buffer.cap
# 1: post_conveyor_buffer.cap
# 2: post_washing_buffer.cap
# 3: pre_press1_buffer.cap
# 4: pre_press2_buffer.cap
# 5: post_press12_buffer.cap
# 6: pre_press_split.capacity
# 7: press1_out.capacity
# 8: press2_out.capacity
# 9: raw_input.capacity
N_VAR = 10
VAR_MIN = 1
VAR_MAX = 10

# Constraint: example constraint on total capacity (can be adapted as needed)
# Here we use a simple constraint: sum of all capacities <= 60
def is_feasible(x):
    return sum(x) <= 60


def evaluate_individual(x):
    if not is_feasible(x):
        # Return None to indicate infeasible; caller must skip
        return None

    # We need to call run_simulation with modified buffer capacities.
    # To keep compatibility, we assume run_simulation can accept an optional
    # "buffer_caps" argument: a dict with buffer names to capacities.
    # If not present in your simulation, you must adapt run_simulation accordingly.
    buffer_caps = {
        "post_loading_buffer": x[0],
        "post_conveyor_buffer": x[1],
        "post_washing_buffer": x[2],
        "pre_press1_buffer": x[3],
        "pre_press2_buffer": x[4],
        "post_press12_buffer": x[5],
        "pre_press_split": x[6],
        "press1_out": x[7],
        "press2_out": x[8],
        "raw_input": x[9],
    }

    # Multi-replication evaluation: average throughput and WIP
    throughputs = []
    wips = []

    for i in range(REPLICATIONS):
        seed = RANDOM_SEED + i
        res = run_simulation(seed, buffer_caps=buffer_caps)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    avg_throughput = sum(throughputs) / len(throughputs)
    avg_wip = sum(wips) / len(wips)

    # Objectives: f1 = WIP (minimize), f2 = -throughput (minimize)
    return (avg_wip, -avg_throughput)


def init_population():
    pop = []
    for _ in range(POP_SIZE):
        ind = [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VAR)]
        pop.append(ind)
    return pop


def tournament_selection(pop, fitnesses, k=2):
    # Binary tournament
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None or dominates(fitnesses[i], fitnesses[best]):
            best = i
    return pop[best]


def dominates(f1, f2):
    # Minimization: f1 dominates f2 if f1 is no worse in all and better in at least one
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1[0] < f2[0] or f1[1] < f2[1])


def crossover(parent1, parent2, pc=0.9):
    if random.random() > pc:
        return parent1[:], parent2[:]
    point = random.randint(1, N_VAR - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    for i in range(N_VAR):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def non_dominated_sort(pop, fitnesses):
    S = [[] for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    rank = [0 for _ in range(len(pop))]
    fronts = [[]]

    for p in range(len(pop)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop)):
            if dominates(fitnesses[p], fitnesses[q]):
                S[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]):
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


def crowding_distance(front, fitnesses):
    distance = [0.0 for _ in front]
    if len(front) == 0:
        return distance
    num_obj = len(fitnesses[0])

    for m in range(num_obj):
        front_sorted = sorted(range(len(front)), key=lambda i: fitnesses[front[i]][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        f_min = fitnesses[front[front_sorted[0]]][m]
        f_max = fitnesses[front[front_sorted[-1]]][m]
        if f_max == f_min:
            continue
        for i in range(1, len(front) - 1):
            prev_f = fitnesses[front[front_sorted[i - 1]]][m]
            next_f = fitnesses[front[front_sorted[i + 1]]][m]
            distance[front_sorted[i]] += (next_f - prev_f) / (f_max - f_min)
    return distance


def select_next_generation(pop, fitnesses):
    fronts, _ = non_dominated_sort(pop, fitnesses)
    new_pop = []
    new_fit = []

    for front in fronts:
        if len(new_pop) + len(front) > POP_SIZE:
            dist = crowding_distance(front, fitnesses)
            sorted_front = sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
            for idx in sorted_front:
                if len(new_pop) >= POP_SIZE:
                    break
                new_pop.append(pop[front[idx]])
                new_fit.append(fitnesses[front[idx]])
            break
        else:
            for idx in front:
                new_pop.append(pop[idx])
                new_fit.append(fitnesses[idx])

    return new_pop, new_fit


def evaluate_population(pop, pool):
    # Evaluate in parallel, skip infeasible individuals
    results = pool.map(evaluate_individual, pop)
    new_pop = []
    new_fit = []
    for ind, fit in zip(pop, results):
        if fit is not None:
            new_pop.append(ind)
            new_fit.append(fit)
    return new_pop, new_fit


def main():
    random.seed(RANDOM_SEED_MOO)

    if N_CORES > mp.cpu_count():
        raise RuntimeError(f"Requested {N_CORES} cores, but only {mp.cpu_count()} available.")

    with mp.Pool(processes=N_CORES) as pool:
        pop = init_population()
        pop, fitnesses = evaluate_population(pop, pool)

        # If all individuals are infeasible, reinitialize until we get some feasible ones
        while len(pop) == 0:
            pop = init_population()
            pop, fitnesses = evaluate_population(pop, pool)

        for gen in range(N_GEN):
            offspring = []
            while len(offspring) < POP_SIZE:
                p1 = tournament_selection(pop, fitnesses)
                p2 = tournament_selection(pop, fitnesses)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                if len(offspring) < POP_SIZE:
                    offspring.append(c2)

            offspring, off_fit = evaluate_population(offspring, pool)

            # If all offspring infeasible, keep current population
            if len(offspring) == 0:
                continue

            combined_pop = pop + offspring
            combined_fit = fitnesses + off_fit

            pop, fitnesses = select_next_generation(combined_pop, combined_fit)

        # Final non-dominated set
        fronts, _ = non_dominated_sort(pop, fitnesses)
        pareto_front = fronts[0]

        # Write results to CSV (only feasible, evaluated points)
        out_file = "moo_results.csv"
        header = [
            "post_loading_buffer_cap",
            "post_conveyor_buffer_cap",
            "post_washing_buffer_cap",
            "pre_press1_buffer_cap",
            "pre_press2_buffer_cap",
            "post_press12_buffer_cap",
            "pre_press_split_cap",
            "press1_out_cap",
            "press2_out_cap",
            "raw_input_cap",
            "wip",
            "throughput"
        ]

        with open(out_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ind, fit in zip(pop, fitnesses):
                wip = fit[0]
                throughput = -fit[1]
                row = ind + [wip, throughput]
                writer.writerow(row)


if __name__ == "__main__":
    main()