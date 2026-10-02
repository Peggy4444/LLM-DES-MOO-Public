import random
import statistics
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv


class BufferCapacityProblem(Problem):
    """
    Multi-objective optimization problem for buffer capacities using NSGA-II.

    Decision variables (all integer in [1,10]):
        x[0] = PostLoadingBuffer cap
        x[1] = PostConveyorBuffer cap
        x[2] = PostWashingBuffer cap
        x[3] = PrePress1Buffer cap
        x[4] = PrePress2Buffer cap
        x[5] = PostPress12Buffer cap

    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)

    Constraint:
        g1(x) <= 0  (user-defined; here we use a simple example:
                     sum(capacities) <= 40; if violated, solution is infeasible
                     and will not be exported to CSV)
    """

    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 n_constr=1,
                 xl=None,
                 xu=None,
                 n_replications=5,
                 base_seed=11):
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

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        n_individuals = X.shape[0]

        F = np.zeros((n_individuals, 2), dtype=float)
        G = np.zeros((n_individuals, self.n_constr), dtype=float)

        for i in range(n_individuals):
            x = X[i]
            caps = [int(v) for v in x[:6]]

            # Example constraint: total buffer capacity <= 40
            # g1(x) = sum(caps) - 40 <= 0 is feasible
            g1 = sum(caps) - 40
            G[i, 0] = g1

            if g1 > 0:
                # Infeasible: assign very bad objective values so NSGA-II discards it
                F[i, 0] = 1e6   # very high WIP
                F[i, 1] = 1e6   # very low throughput (since we minimize -throughput)
                continue

            throughputs = []
            wips = []

            for r in range(self.n_replications):
                seed = self.base_seed + r + random.randint(0, 1000000)
                res = run_simulation(seed, caps, WARMUP_SECONDS, MEASURE_UNTIL)
                throughputs.append(res["overall"]["throughput"])
                wips.append(res["overall"]["wip"])

            avg_throughput = statistics.mean(throughputs)
            avg_wip = statistics.mean(wips)

            F[i, 0] = avg_wip
            F[i, 1] = -avg_throughput

        out["F"] = F
        out["G"] = G


def run_simulation(seed, caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    # This function is assumed to be defined in the existing simulation code.
    # Here we only declare it so that the optimizer code is syntactically complete.
    # The actual implementation from the provided simulation must be used.
    raise NotImplementedError("Use the run_simulation function from the simulation code.")


def run_nsga2_optimization(
    pop_size=50,
    n_gen=50,
    n_replications=5,
    base_seed=11,
    verbose=True
):
    problem = BufferCapacityProblem(
        n_var=6,
        n_obj=2,
        n_constr=1,
        xl=np.array([1, 1, 1, 1, 1, 1]),
        xu=np.array([10, 10, 10, 10, 10, 10]),
        n_replications=n_replications,
        base_seed=base_seed
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

    Only individuals that satisfy all constraints (G <= 0) are exported.
    """

    fieldnames = [
        "gen",
        "ind",
        "PostLoadingBuffer_cap",
        "PostConveyorBuffer_cap",
        "PostWashingBuffer_cap",
        "PrePress1Buffer_cap",
        "PrePress2Buffer_cap",
        "PostPress12Buffer_cap",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G") if "G" in pop.get_keys() else None

        for ind_idx, (x, f) in enumerate(zip(X, F)):
            # Check feasibility: all constraints <= 0
            feasible = True
            if G is not None:
                g_vals = G[ind_idx]
                if np.any(g_vals > 0):
                    feasible = False

            if not feasible:
                continue

            caps = [int(v) for v in x[:6]]
            wip = float(f[0])
            throughput = float(-f[1])

            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "PostLoadingBuffer_cap": caps[0],
                "PostConveyorBuffer_cap": caps[1],
                "PostWashingBuffer_cap": caps[2],
                "PrePress1Buffer_cap": caps[3],
                "PrePress2Buffer_cap": caps[4],
                "PostPress12Buffer_cap": caps[5],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = run_nsga2_optimization(
        pop_size=50,
        n_gen=50,
        n_replications=5,
        base_seed=11,
        verbose=True
    )

    export_history_to_csv(result, filename="moo_simulation_results.csv")