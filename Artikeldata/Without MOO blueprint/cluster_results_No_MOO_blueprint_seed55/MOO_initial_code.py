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

# Decision variables: capacities of all buffers (1-10, integer)
# Order:
# 0: raw_input
# 1: post_loading_buffer
# 2: post_conveyor_buffer
# 3: post_washing_buffer
# 4: pre_press1_buffer
# 5: pre_press2_buffer
# 6: post_press12_buffer
# 7: pre_press_split_buffer
# 8: press1_out
# 9: press2_out

VAR_MIN = 1
VAR_MAX = 10
N_VARS = 10

# Constraint: example – total capacity budget (can be adapted as needed)
# Here we use a simple constraint: sum of capacities <= 60
def is_feasible(ind):
    return sum(ind) <= 60


def decode_and_run(ind, base_seed):
    """
    Wrapper that runs the simulation with the given buffer capacities.
    This function assumes that run_simulation is modified to accept
    buffer capacities as an argument, or that there is a separate
    function that builds the model with these capacities.
    If not yet present, you must adapt run_simulation accordingly.
    """
    # Example: assume run_simulation has been extended to accept a
    # 'buffer_caps' argument (list of 10 ints) and uses them internally.
    # Also assume it still uses REPLICATIONS and RANDOM_SEED as before.
    from simulation_module import run_simulation, REPLICATIONS, RANDOM_SEED  # adjust module name

    throughputs = []
    wips = []

    for r in range(REPLICATIONS):
        seed = base_seed + r
        res = run_simulation(seed, buffer_caps=ind)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

    # Objectives: minimize wip, maximize throughput
    # For NSGA-II we typically minimize both, so we minimize -throughput
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_wip = sum(wips) / len(wips)
    return avg_wip, -avg_throughput


def init_individual():
    return [random.randint(VAR_MIN, VAR_MAX) for _ in range(N_VARS)]


def tournament_selection(pop, k=2):
    best = None
    for _ in range(k):
        ind = random.choice(pop)
        if best is None:
            best = ind
        else:
            if dominates(ind, best):
                best = ind
    return best


def dominates(a, b):
    # a dominates b if a is no worse in all objectives and better in at least one
    not_worse = all(x <= y for x, y in zip(a["objs"], b["objs"]))
    strictly_better = any(x < y for x, y in zip(a["objs"], b["objs"]))
    return not_worse and strictly_better


def fast_non_dominated_sort(pop):
    fronts = []
    S = {}
    n = {}
    rank = {}

    for p in range(len(pop)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop)):
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
    front = [i for i in range(len(pop)) if n[i] == 0]
    fronts.append(front)

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
    return fronts


def crowding_distance_assignment(front, pop):
    l = len(front)
    if l == 0:
        return {}
    distances = {i: 0.0 for i in front}
    n_obj = len(pop[0]["objs"])

    for m in range(n_obj):
        front_sorted = sorted(front, key=lambda i: pop[i]["objs"][m])
        distances[front_sorted[0]] = float("inf")
        distances[front_sorted[-1]] = float("inf")
        obj_min = pop[front_sorted[0]]["objs"][m]
        obj_max = pop[front_sorted[-1]]["objs"][m]
        if obj_max == obj_min:
            continue
        for k in range(1, l - 1):
            prev_obj = pop[front_sorted[k - 1]]["objs"][m]
            next_obj = pop[front_sorted[k + 1]]["objs"][m]
            distances[front_sorted[k]] += (next_obj - prev_obj) / (obj_max - obj_min)
    return distances


def crossover(parent1, parent2, pc=0.9):
    if random.random() > pc:
        return parent1[:], parent2[:]
    point = random.randint(1, N_VARS - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(ind, pm=0.1):
    for i in range(N_VARS):
        if random.random() < pm:
            ind[i] = random.randint(VAR_MIN, VAR_MAX)
    return ind


def evaluate_population(pop, pool, base_seed):
    # Evaluate only individuals that are feasible and not yet evaluated
    tasks = []
    idx_map = []
    for i, ind in enumerate(pop):
        if ind["objs"] is None and ind["feasible"]:
            tasks.append(ind["vars"])
            idx_map.append(i)

    if tasks:
        func = partial(decode_and_run, base_seed=base_seed)
        results = pool.map(func, tasks)
        for idx, objs in zip(idx_map, results):
            pop[idx]["objs"] = objs


def create_initial_population():
    pop = []
    while len(pop) < POP_SIZE:
        vars_ = init_individual()
        feasible = is_feasible(vars_)
        if not feasible:
            continue
        pop.append({"vars": vars_, "objs": None, "feasible": True})
    return pop


def make_offspring(pop):
    offspring = []
    while len(offspring) < POP_SIZE:
        p1 = tournament_selection(pop)
        p2 = tournament_selection(pop)
        c1_vars, c2_vars = crossover(p1["vars"], p2["vars"])
        c1_vars = mutate(c1_vars)
        c2_vars = mutate(c2_vars)

        for child_vars in (c1_vars, c2_vars):
            if len(offspring) >= POP_SIZE:
                break
            if not is_feasible(child_vars):
                continue
            offspring.append({"vars": child_vars, "objs": None, "feasible": True})
    return offspring


def nsga2():
    random.seed(RANDOM_SEED_MOO)

    with mp.Pool(processes=N_CORES) as pool:
        pop = create_initial_population()
        evaluate_population(pop, pool, base_seed=10000)

        for gen in range(N_GEN):
            offspring = make_offspring(pop)
            evaluate_population(offspring, pool, base_seed=10000 + (gen + 1) * 1000)

            combined = pop + offspring
            fronts = fast_non_dominated_sort(combined)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) > POP_SIZE:
                    distances = crowding_distance_assignment(front, combined)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    needed = POP_SIZE - len(new_pop)
                    new_pop.extend([combined[i] for i in sorted_front[:needed]])
                    break
                else:
                    new_pop.extend([combined[i] for i in front])
            pop = new_pop

        # Final non-dominated front
        fronts = fast_non_dominated_sort(pop)
        best_front = fronts[0]
        pareto_set = [pop[i] for i in best_front]
        return pareto_set


def save_results(pareto_set, filename="moo_results.csv"):
    # Only feasible individuals are in the population by construction
    # and only evaluated ones (objs not None) are saved.
    fieldnames = [f"var_{i}" for i in range(N_VARS)] + ["wip", "throughput"]
    # throughput was minimized as -throughput, so convert back
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ind in pareto_set:
            if ind["objs"] is None:
                continue
            wip = ind["objs"][0]
            throughput = -ind["objs"][1]
            row = {f"var_{i}": ind["vars"][i] for i in range(N_VARS)}
            row["wip"] = wip
            row["throughput"] = throughput
            writer.writerow(row)


if __name__ == "__main__":
    pareto = nsga2()
    save_results(pareto, filename="moo_results.csv")