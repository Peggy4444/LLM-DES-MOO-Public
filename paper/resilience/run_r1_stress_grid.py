"""
R1 -- Resilience of Pareto solutions under disruption scenarios.

For each unique buffer configuration on the 19-point S1 Pareto front
(paper/figs/moo_pareto_solutions.csv, dedup'd to 15 unique configs),
simulate the line under four scenarios:

    Nominal : baseline machine parameters
    D1      : Press 1 aging  -- MTTR of Presses cell 1 doubled,
                                availability -10 pp
    D2      : Systemic reliability degradation -- availability of every
              machine reduced by 5 pp, MTTR x1.3
    D3      : Feeder failure  -- Hantering cell availability -15 pp,
                                 MTTR x3

Each (config, scenario) pair is replicated with SEEDS seeds. Per-run KPIs
are persisted to r1_stress_runs.csv for downstream aggregation and plotting.
"""

import csv
import time
from pathlib import Path

from des_perturbable import run_perturbed_simulation


HERE = Path(__file__).parent
PARETO_CSV  = HERE.parent / "figs" / "moo_pareto_solutions.csv"
OUTPUT_CSV  = HERE / "r1_stress_runs.csv"

SEEDS = [11, 22, 33, 44, 55]

# -- Buffer name mapping from the Pareto CSV column headers to DES names. -----
# paper/figs/moo_pareto_solutions.csv uses CamelCase column names.
BUF_COLS = [
    ("PostLoadingBuffer_capacity",     "PostLoadingBuffer"),
    ("PostConveyorBuffer_capacity",    "PostConveyorBuffer"),
    ("PostWashingBuffer_capacity",     "PostWashingBuffer"),
    ("PrePress1Buffer_capacity",       "PrePress1Buffer"),
    ("PrePress2Buffer_capacity",       "PrePress2Buffer"),
    ("PostPress1_2_Buffer_capacity",   "PostPress12Buffer"),
]

# -- Disruption scenarios. ---------------------------------------------------
# Each scenario targets a distinct part of the line so that buffer
# configurations respond differently. The perturbation magnitudes are
# calibrated to produce meaningful (5-15%) throughput impact without
# stalling the line, so that resilience differences between Pareto
# configurations are observable rather than dominated by structural
# limits.
SCENARIOS = {
    "nominal": {},
    # D1: Press 1 severe aging. Combines a reliability hit (availability
    # drops 10 pp, MTTR doubles) on the dominant bottleneck. Configurations
    # that under-buffer the PrePress1 side or fail to reroute through
    # Press 2 will degrade more.
    "D1_press1_aging": {
        "Presses cell 1": {"mttr_mult": 2.0, "availability_delta_pp": -10.0},
    },
    # D2: Plant-wide reliability degradation. Simulates aging or staff
    # turnover across the line: availability -5 pp and MTTR x1.3 for every
    # non-perfect machine.
    "D2_systemic_degradation": {
        "Loading robot":        {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Washing machine":      {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Hantering cell":       {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Presses cell 1":       {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Presses cell 2":       {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
        "Quality station cell": {"availability_delta_pp": -5.0, "mttr_mult": 1.3},
    },
    # D3: Single-line feeder failure. The Hantering cell feeds both parallel
    # press cells via the splitter; when it degrades, both presses starve
    # simultaneously. Availability -15 pp and MTTR x3 makes this a serious
    # feeder disruption. Configurations that pre-load PostWashing and the
    # pre-press buffers should retain more throughput.
    "D3_feeder_failure": {
        "Hantering cell": {"availability_delta_pp": -15.0, "mttr_mult": 3.0},
    },
}


def load_pareto_configs():
    """Load configurations from the paper/figs 19-point Pareto CSV and
    deduplicate to unique buffer-capacity tuples (there are 15 unique
    configs across the 19 non-dominated rows)."""
    seen = {}
    with open(PARETO_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caps = {des_name: int(row[csv_col]) for csv_col, des_name in BUF_COLS}
            key = tuple(caps[n] for _, n in BUF_COLS)
            if key in seen:
                continue
            seen[key] = {
                "generation": int(row["generation_index"]),
                "individual": int(row["individual_index"]),
                "caps": caps,
                "wip_from_moo": float(row["wip"]),
                "tp_from_moo": float(row["throughput"]),
            }
    return list(seen.values())


def main():
    configs = load_pareto_configs()
    total = len(configs) * len(SCENARIOS) * len(SEEDS)
    print(f"Configs: {len(configs)}  Scenarios: {len(SCENARIOS)}  Seeds: {len(SEEDS)}  Total runs: {total}")

    rows = []
    done = 0
    t_start = time.time()

    for cfg_idx, cfg in enumerate(configs):
        for sc_name, sc_perturb in SCENARIOS.items():
            for seed in SEEDS:
                t0 = time.time()
                res = run_perturbed_simulation(
                    seed=seed,
                    buffer_caps=cfg["caps"],
                    perturb=sc_perturb,
                )
                rows.append({
                    "config_id":       cfg_idx,
                    "generation":      cfg["generation"],
                    "individual":      cfg["individual"],
                    "scenario":        sc_name,
                    "seed":            seed,
                    "cap_post_loading":  cfg["caps"]["PostLoadingBuffer"],
                    "cap_post_conveyor": cfg["caps"]["PostConveyorBuffer"],
                    "cap_post_washing":  cfg["caps"]["PostWashingBuffer"],
                    "cap_pre_press1":    cfg["caps"]["PrePress1Buffer"],
                    "cap_pre_press2":    cfg["caps"]["PrePress2Buffer"],
                    "cap_post_press12":  cfg["caps"]["PostPress12Buffer"],
                    "throughput":      res["throughput"],
                    "wip":             res["wip"],
                    "sec":             res["sec"],
                    "produced":        res["produced"],
                    "wall_seconds":    time.time() - t0,
                })
                done += 1
                if done % 20 == 0 or done == total:
                    eta = (time.time() - t_start) / done * (total - done)
                    print(f"  {done}/{total}  ETA ~{eta:.0f} s")

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nTotal wall-clock: {time.time() - t_start:.0f} s")
    print(f"Persisted: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
