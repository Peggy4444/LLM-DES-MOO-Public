import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Assumes run_simulation is imported from the simulation module
# from your_simulation_module import run_simulation, RANDOM_SEED


BUFFER_NAMES = [
    "post_loading_buffer",
    "post_conveyor_buffer",
    "post_washing_buffer",
    "pre_press1_buffer",
    "pre_press2_buffer",
    "post_press12_buffer"
]


def run_simulation_with_buffers(seed, buffer_caps):
    """
    Wrapper around run_simulation that sets buffer capacities before running.
    Assumes that run_simulation reads capacities from global variables or
    is adapted to accept them. Here we assume it accepts a dict of capacities.
    """
    # Adaptation point: if run_simulation signature is changed to accept
    # buffer capacities, pass them here. For now, assume:
    # run_simulation(seed, buffer_caps=...)
    res = run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL)
    return res


def evaluate_individual(x, seed_offset):
    """
    Evaluate a single individual.
    x: array of 6 integers in [1,10] representing buffer capacities.
    Returns (f1, f2) = (wip, -throughput) for minimization.
    """
    buffer_caps = {name: int(cap) for name, cap in zip(BUFFER_NAMES, x)}

    # Constraint check (example: total capacity <= 40, can be adapted)
    # If violated, return None to indicate infeasible and skip.
    total_cap = sum(buffer_caps.values())
    if total_cap > 40:
        return None

    seed = RANDOM_SEED + seed_offset
    res = run_simulation_with_buffers(seed, buffer_caps)

    wip = res["overall"]["wip"]
    throughput = res["overall"]["throughput"]

    return np.array([wip, -throughput]), buffer_caps, res


class ProductionLineProblem(Problem):
    def __init__(self, pop_size, **kwargs):
        super().__init__(
            n_var=len(BUFFER_NAMES),
            n_obj=2,
            n_constr=0,
            xl=np.array([1] * len(BUFFER_NAMES)),
            xu=np.array([10] * len(BUFFER_NAMES)),
            type_var=int,
            **kwargs
        )
        self.pop_size = pop_size
        self.eval_counter = 0
        self.results = []

    def _evaluate(self, X, out, *args, **kwargs):
        n_individuals = X.shape[0]
        seeds = np.arange(self.eval_counter, self.eval_counter + n_individuals)
        self.eval_counter += n_individuals

        with Pool(50) as pool:
            func = partial(evaluate_individual)
            eval_results = pool.starmap(func, [(X[i], int(seeds[i])) for i in range(n_individuals)])

        F = []
        for i, res in enumerate(eval_results):
            if res is None:
                # Infeasible: assign large penalty values
                F.append([1e6, 1e6])
            else:
                f, buffer_caps, sim_res = res
                F.append(f.tolist())
                # Store only feasible results
                self.results.append({
                    **{name: int(X[i, j]) for j, name in enumerate(BUFFER_NAMES)},
                    "wip": sim_res["overall"]["wip"],
                    "throughput": sim_res["overall"]["throughput"],
                    "produced_parts": sim_res["overall"]["produced_parts"]
                })

        out["F"] = np.array(F)


def main():
    pop_size = 50
    n_gen = 50

    problem = ProductionLineProblem(pop_size=pop_size)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=RANDOM_SEED,
        save_history=False,
        verbose=True
    )

    # Save only feasible evaluated points (constraints already enforced)
    df = pd.DataFrame(problem.results)
    df.to_csv("moo_results.csv", index=False)

    # Optionally, save Pareto front
    pareto_X = res.X
    pareto_F = res.F
    pareto_data = []
    for i in range(pareto_X.shape[0]):
        row = {name: int(pareto_X[i, j]) for j, name in enumerate(BUFFER_NAMES)}
        row["wip"] = float(pareto_F[i, 0])
        row["throughput"] = float(-pareto_F[i, 1])
        pareto_data.append(row)
    pd.DataFrame(pareto_data).to_csv("moo_pareto_front.csv", index=False)


if __name__ == "__main__":
    main()