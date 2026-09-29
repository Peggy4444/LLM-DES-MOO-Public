"""
R2 -- Scenario-conditioned SCORE bottleneck analysis.

For a single Pareto-optimal configuration (the knee-point from R1), rank
each machine by throughput improvement obtained from a small "SCORE-style"
relaxation, under each of nominal + D1 + D2 + D3.

Relaxation applied per candidate machine:
    availability +5 pp   AND   MTTR x0.7   AND   process_time x0.9

Machines whose relaxation yields the largest TP gain in a given scenario
are that scenario's dominant bottlenecks. Comparing rankings across
scenarios distinguishes robust bottlenecks (top rank in all scenarios)
from scenario-specific bottlenecks (rank shifts).

Persists:
    r2_score_runs.csv          -- one row per (scenario, target machine, seed)
    r2_score_ranking.csv       -- per-scenario ranking matrix
"""

import csv
import copy
import time
from pathlib import Path

from des_perturbable import (
    run_perturbed_simulation, MACHINE_ORDER, BASELINE_MACHINES,
)


HERE = Path(__file__).parent
OUT_RUNS    = HERE / "r2_score_runs.csv"
OUT_RANKING = HERE / "r2_score_ranking.csv"

# Knee-point buffer configuration V2 from Section 5.2 Table 7:
# bufs = 1-1-1-3-2-3 (matches paper/figs/suggested_improvements_freezed.csv).
KNEE_CAPS = {
    "PostLoadingBuffer":  1,
    "PostConveyorBuffer": 1,
    "PostWashingBuffer":  1,
    "PrePress1Buffer":    3,
    "PrePress2Buffer":    2,
    "PostPress12Buffer":  3,
}

# Same disruption scenarios as R1.
SCENARIOS = {
    "nominal": {},
    "D1_press1_aging": {
        "Presses cell 1": {"mttr_mult": 2.0, "availability_delta_pp": -10.0},
    },
    "D2_systemic_degradation": {
        "Loading robot":        {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Washing machine":      {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Hantering cell":       {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Presses cell 1":       {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Presses cell 2":       {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Quality station cell": {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
    },
    "D3_feeder_failure": {
        "Hantering cell": {"availability_delta_pp": -15.0, "mttr_mult": 3.0},
    },
}

SCORE_RELAXATION = {
    "availability_delta_pp": +5.0,
    "mttr_mult":              0.7,
    "process_time_mult":      0.9,
}

SEEDS = [11, 22, 33, 44, 55]


def compose(scenario_pert, target_machine, relaxation):
    """Return a perturb dict combining scenario stress with a SCORE
    relaxation on target_machine."""
    out = copy.deepcopy(scenario_pert)
    base = dict(out.get(target_machine, {}))
    # Combine multiplicatively / additively.
    if "mttr_mult" in relaxation:
        base["mttr_mult"] = base.get("mttr_mult", 1.0) * relaxation["mttr_mult"]
    if "process_time_mult" in relaxation:
        base["process_time_mult"] = base.get("process_time_mult", 1.0) * relaxation["process_time_mult"]
    if "availability_delta_pp" in relaxation:
        base["availability_delta_pp"] = (
            base.get("availability_delta_pp", 0.0) + relaxation["availability_delta_pp"]
        )
    out[target_machine] = base
    return out


def _mean(vals):
    return sum(vals) / len(vals)


def main():
    rows = []
    total = len(SCENARIOS) * (len(MACHINE_ORDER) + 1) * len(SEEDS)
    done = 0
    t_start = time.time()

    per_scenario_baseline = {}

    for sc_name, sc_pert in SCENARIOS.items():
        # First, measure the scenario baseline (no relaxation).
        tp_base = []
        for seed in SEEDS:
            res = run_perturbed_simulation(seed=seed, buffer_caps=KNEE_CAPS, perturb=sc_pert)
            tp_base.append(res["throughput"])
            rows.append({
                "scenario": sc_name,
                "target_machine": "__baseline__",
                "seed": seed,
                "throughput": res["throughput"],
                "wip": res["wip"],
                "sec": res["sec"],
            })
            done += 1
        per_scenario_baseline[sc_name] = _mean(tp_base)
        print(f"  {sc_name:30s}  baseline TP = {per_scenario_baseline[sc_name]:.3f}")

        # Then, apply the SCORE relaxation to each machine in turn.
        for m in MACHINE_ORDER:
            if BASELINE_MACHINES[m]["availability"] >= 100.0 and BASELINE_MACHINES[m]["mttr"] <= 1.0:
                # Perfectly reliable, minimal-processing machine (conveyor).
                # Relaxation cannot improve it. Skip but record so ranking is complete.
                for seed in SEEDS:
                    rows.append({
                        "scenario": sc_name, "target_machine": m, "seed": seed,
                        "throughput": per_scenario_baseline[sc_name],
                        "wip": None, "sec": None,
                    })
                    done += 1
                continue
            for seed in SEEDS:
                pert = compose(sc_pert, m, SCORE_RELAXATION)
                res = run_perturbed_simulation(seed=seed, buffer_caps=KNEE_CAPS, perturb=pert)
                rows.append({
                    "scenario": sc_name, "target_machine": m, "seed": seed,
                    "throughput": res["throughput"],
                    "wip": res["wip"],
                    "sec": res["sec"],
                })
                done += 1
                if done % 20 == 0 or done == total:
                    eta = (time.time() - t_start) / done * (total - done)
                    print(f"  {done}/{total}  ETA ~{eta:.0f} s")

    with open(OUT_RUNS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Build ranking matrix.
    from collections import defaultdict
    tp_by_sc_m = defaultdict(list)
    for r in rows:
        tp_by_sc_m[(r["scenario"], r["target_machine"])].append(r["throughput"])
    mean_tp = {k: _mean(v) for k, v in tp_by_sc_m.items()}

    # Delta relative to scenario baseline.
    rank_rows = []
    for sc_name in SCENARIOS:
        base = per_scenario_baseline[sc_name]
        gains = []
        for m in MACHINE_ORDER:
            g = mean_tp[(sc_name, m)] - base
            gains.append((m, g))
        gains.sort(key=lambda x: -x[1])
        for rank, (m, g) in enumerate(gains, start=1):
            rank_rows.append({
                "scenario": sc_name,
                "rank": rank,
                "machine": m,
                "tp_gain": g,
                "tp_after_improvement": mean_tp[(sc_name, m)],
                "tp_baseline": base,
            })

    with open(OUT_RANKING, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        w.writeheader()
        w.writerows(rank_rows)

    print(f"\nTotal wall-clock: {time.time() - t_start:.0f} s")
    print(f"Persisted: {OUT_RUNS}\n           {OUT_RANKING}")


if __name__ == "__main__":
    main()
