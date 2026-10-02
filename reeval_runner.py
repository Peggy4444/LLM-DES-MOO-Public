"""
Re-evaluate every buffer configuration in reeval_configurations.csv under the
framework DES (paper/resilience/des_perturbable.py) across 10 random seeds.

Horizon and warm-up match tbl:hyper (8 days / 1 day), matching Kaveh's spec.
"""
import time
from pathlib import Path

import pandas as pd

from paper.resilience.des_perturbable import run_perturbed_simulation

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
IN_CSV = Path(__file__).parent / "reeval_configurations.csv"
OUT_CSV = Path(__file__).parent / "reeval_results.csv"

BUF_MAP = {
    "PostLoading":   "PostLoadingBuffer",
    "PostConveyor":  "PostConveyorBuffer",
    "PostWashing":   "PostWashingBuffer",
    "PrePress1":     "PrePress1Buffer",
    "PrePress2":     "PrePress2Buffer",
    "PostPress1_2":  "PostPress12Buffer",
}

def main():
    cfg = pd.read_csv(IN_CSV)
    results = []
    t_global = time.time()
    for i, row in cfg.iterrows():
        buffer_caps = {BUF_MAP[k]: int(row[k]) for k in BUF_MAP}
        per_seed = []
        t0 = time.time()
        for s in SEEDS:
            r = run_perturbed_simulation(seed=s, buffer_caps=buffer_caps)
            per_seed.append((s, r["throughput"], r["wip"], r["sec"]))
        dt = time.time() - t0
        th_vals = [p[1] for p in per_seed]
        wip_vals = [p[2] for p in per_seed]
        sec_vals = [p[3] for p in per_seed]
        rec = {
            "config_id": row["config_id"],
            "source":    row["source"],
            "note":      row.get("note", ""),
            "total":     int(row["total"]),
            "TH_mean":   sum(th_vals)/len(th_vals),
            "TH_std":    pd.Series(th_vals).std(ddof=0),
            "WIP_mean":  sum(wip_vals)/len(wip_vals),
            "WIP_std":   pd.Series(wip_vals).std(ddof=0),
            "SEC_mean":  sum(sec_vals)/len(sec_vals),
            "SEC_std":   pd.Series(sec_vals).std(ddof=0),
            "n_seeds":   len(SEEDS),
            "runtime_s": dt,
        }
        # also keep per-seed in wide columns for traceability
        for s, th, wip, sec in per_seed:
            rec[f"TH_s{s}"] = th
            rec[f"WIP_s{s}"] = wip
            rec[f"SEC_s{s}"] = sec
        results.append(rec)
        if (i + 1) % 10 == 0 or i == 0 or i == len(cfg) - 1:
            print(f"[{i+1}/{len(cfg)}] {row['config_id']}  TH={rec['TH_mean']:.3f}  WIP={rec['WIP_mean']:.3f}  SEC={rec['SEC_mean']:.4f}  ({dt:.1f}s)", flush=True)
        # periodic incremental save in case we abort
        if (i + 1) % 25 == 0 or i == len(cfg) - 1:
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nDone. {len(results)} configs x {len(SEEDS)} seeds in {time.time()-t_global:.1f}s.")
    print(f"Written {OUT_CSV}")

if __name__ == "__main__":
    main()
