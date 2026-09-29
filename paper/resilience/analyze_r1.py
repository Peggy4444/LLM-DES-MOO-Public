"""
R1 aggregation & plotting.

Reads r1_stress_runs.csv, aggregates per (config, scenario), computes
resilience metrics per config, and produces:

  * r1_config_scenario.csv  -- mean/std of TP, WIP, SEC per (config, scenario)
  * r1_resilience.csv       -- resilience metrics per config
  * r1_pareto_resilience.png    -- nominal Pareto colored by worst-case R
  * r1_scenario_kpi_matrix.png  -- TP drop heatmap (config x scenario)
  * r1_selected_bars.png        -- TP drop bar chart for 3 highlighted configs
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
RUNS_CSV      = HERE / "r1_stress_runs.csv"
CONFIG_SC_CSV = HERE / "r1_config_scenario.csv"
RESIL_CSV     = HERE / "r1_resilience.csv"

SCENARIO_LABEL = {
    "nominal":                     "Nominal",
    "D1_press1_aging":             "D1: Press 1 aging ($-10$ pp, MTTR $\\times$2)",
    "D2_systemic_degradation":     "D2: Systemic degradation ($-5$ pp, MTTR $\\times$1.3)",
    "D3_feeder_failure":           "D3: Feeder failure (Hantering $-15$ pp, MTTR $\\times$3)",
}
DISRUPTIONS = ["D1_press1_aging", "D2_systemic_degradation", "D3_feeder_failure"]


def aggregate(df):
    grp = df.groupby(["config_id", "scenario"], as_index=False).agg(
        tp_mean=("throughput", "mean"),
        tp_std=("throughput", "std"),
        wip_mean=("wip", "mean"),
        wip_std=("wip", "std"),
        sec_mean=("sec", "mean"),
        sec_std=("sec", "std"),
    )
    # Attach buffer caps for readability.
    caps = df.drop_duplicates("config_id")[
        ["config_id", "cap_post_loading", "cap_post_conveyor",
         "cap_post_washing", "cap_pre_press1", "cap_pre_press2",
         "cap_post_press12"]
    ]
    grp = grp.merge(caps, on="config_id", how="left")
    return grp


def resilience_metrics(agg):
    wide = agg.pivot(index="config_id", columns="scenario",
                     values=["tp_mean", "wip_mean", "sec_mean"])
    tp_nom = wide[("tp_mean", "nominal")]

    out = pd.DataFrame({"config_id": tp_nom.index})
    out = out.set_index("config_id")
    out["tp_nominal"]  = tp_nom
    out["wip_nominal"] = wide[("wip_mean", "nominal")]
    out["sec_nominal"] = wide[("sec_mean", "nominal")]

    tp_ratios = []
    for d in DISRUPTIONS:
        r = wide[("tp_mean", d)] / tp_nom
        out[f"tp_ratio_{d}"] = r
        out[f"tp_drop_pct_{d}"] = (1 - r) * 100
        out[f"wip_infl_pct_{d}"] = (
            wide[("wip_mean", d)] - out["wip_nominal"]
        ) / out["wip_nominal"] * 100
        tp_ratios.append(r)

    ratios = pd.concat(tp_ratios, axis=1)
    out["R_worst"] = ratios.min(axis=1)              # worst-case retention
    out["R_mean"]  = ratios.mean(axis=1)              # average retention
    out["R_area"]  = (1 - ratios).sum(axis=1)         # cumulative drop (lower=better)
    return out.reset_index()


def _bufstr(row):
    return (f"{int(row.cap_post_loading)}-{int(row.cap_post_conveyor)}-"
            f"{int(row.cap_post_washing)}-{int(row.cap_pre_press1)}-"
            f"{int(row.cap_pre_press2)}-{int(row.cap_post_press12)}")


def pick_highlight_configs(agg_nom, runs_df):
    """Explicitly locate V1/V2/V3 from Section 5.2 Table 7 by matching their
    buffer configurations. V1 = (1,1,1,1,1,1), V2 = (1,1,1,3,2,3), V3 =
    (1,3,4,3,4,2). Falls back to nearest-match if a config is missing."""
    V_CONFIGS = {
        "V1 (Lowest-WIP)": (1, 1, 1, 1, 1, 1),
        "V2 (Knee)":       (1, 1, 1, 3, 2, 3),
        "V3 (Highest-TH)": (1, 3, 4, 3, 4, 2),
    }
    caps_by_cid = (
        runs_df.drop_duplicates("config_id")
        .set_index("config_id")[
            ["cap_post_loading", "cap_post_conveyor", "cap_post_washing",
             "cap_pre_press1", "cap_pre_press2", "cap_post_press12"]
        ]
    )
    out = {}
    for lbl, target in V_CONFIGS.items():
        match = caps_by_cid[
            (caps_by_cid["cap_post_loading"]   == target[0]) &
            (caps_by_cid["cap_post_conveyor"]  == target[1]) &
            (caps_by_cid["cap_post_washing"]   == target[2]) &
            (caps_by_cid["cap_pre_press1"]     == target[3]) &
            (caps_by_cid["cap_pre_press2"]     == target[4]) &
            (caps_by_cid["cap_post_press12"]   == target[5])
        ]
        if len(match) == 0:
            raise RuntimeError(
                f"{lbl} config {target} not found in the current Pareto sample; "
                "check paper/figs/moo_pareto_solutions.csv and Table 7."
            )
        out[lbl] = int(match.index[0])
    return out


def plot_pareto_resilience(agg, resil, highlights, path):
    nom = agg[agg["scenario"] == "nominal"].merge(
        resil[["config_id", "R_worst"]], on="config_id"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(nom["wip_mean"], nom["tp_mean"],
                    c=nom["R_worst"], cmap="viridis",
                    s=90, edgecolor="k", linewidth=0.6, zorder=3)
    # Halos on the highlighted three.
    for lbl, cid in highlights.items():
        r = nom[nom["config_id"] == cid].iloc[0]
        ax.scatter(r["wip_mean"], r["tp_mean"], s=280, facecolor="none",
                   edgecolor="crimson", linewidth=1.8, zorder=4)
        ax.annotate(lbl, (r["wip_mean"], r["tp_mean"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="crimson")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Worst-case resilience $R_\\mathrm{worst}$\n"
                   "$= \\min_{d \\in \\{D_1,D_2,D_3\\}}\\ TP_d / TP_\\mathrm{nom}$",
                   fontsize=9)
    ax.set_xlabel("WIP (parts, nominal)")
    ax.set_ylabel("Throughput (parts/h, nominal)")
    ax.set_title("Resilience-augmented Pareto front (S1: TP / WIP)")
    ax.grid(alpha=0.3, zorder=1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scenario_matrix(resil, highlights, path):
    dfp = resil[["config_id"] + [f"tp_drop_pct_{d}" for d in DISRUPTIONS]].copy()
    dfp = dfp.set_index("config_id")
    dfp.columns = [SCENARIO_LABEL[d] for d in DISRUPTIONS]

    # Order rows by worst-case resilience (best at top).
    order = resil.sort_values("R_worst", ascending=False)["config_id"].tolist()
    dfp = dfp.loc[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(dfp.values, cmap="Reds", aspect="auto", vmin=0)
    ax.set_xticks(range(len(dfp.columns)))
    ax.set_xticklabels(dfp.columns, rotation=20, ha="right", fontsize=8)
    labels = []
    highlight_ids = set(highlights.values())
    highlight_lookup = {v: k for k, v in highlights.items()}
    for cid in dfp.index:
        base = f"cfg {cid}"
        if cid in highlight_ids:
            base = f"{base}  ({highlight_lookup[cid]})"
        labels.append(base)
    ax.set_yticks(range(len(dfp.index)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, cid in enumerate(dfp.index):
        for j, col in enumerate(dfp.columns):
            v = dfp.iloc[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > dfp.values.max()*0.55 else "black",
                    fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Throughput drop vs nominal (%)", fontsize=9)
    ax.set_title("Throughput degradation per configuration and disruption\n"
                 "(configs sorted by worst-case resilience, best at top)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_selected_bars(resil, highlights, path):
    rows = []
    for lbl, cid in highlights.items():
        r = resil[resil["config_id"] == cid].iloc[0]
        rows.append({
            "point": lbl,
            "D1": r["tp_drop_pct_D1_press1_aging"],
            "D2": r["tp_drop_pct_D2_systemic_degradation"],
            "D3": r["tp_drop_pct_D3_feeder_failure"],
        })
    df = pd.DataFrame(rows).set_index("point")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(df.index))
    w = 0.25
    for i, sc in enumerate(["D1", "D2", "D3"]):
        ax.bar(x + (i - 1) * w, df[sc].values, w, label=sc)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Throughput drop vs nominal (%)")
    ax.set_title("Throughput degradation of LLM-selected Pareto points")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Disruption", ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    runs = pd.read_csv(RUNS_CSV)
    agg = aggregate(runs)
    agg.to_csv(CONFIG_SC_CSV, index=False)

    resil = resilience_metrics(agg)
    resil.to_csv(RESIL_CSV, index=False)

    nom_agg = agg[agg["scenario"] == "nominal"].copy()
    highlights = pick_highlight_configs(nom_agg, runs)
    print("Highlighted configs:")
    for lbl, cid in highlights.items():
        row = nom_agg[nom_agg["config_id"] == cid].iloc[0]
        r_row = resil[resil["config_id"] == cid].iloc[0]
        buf = _bufstr(row)
        print(f"  {lbl:10s}  cfg {cid:2d}  bufs={buf}  "
              f"TP={row['tp_mean']:.3f}  WIP={row['wip_mean']:.3f}  "
              f"R_worst={r_row['R_worst']:.3f}  R_mean={r_row['R_mean']:.3f}")

    plot_pareto_resilience(agg, resil, highlights, HERE / "r1_pareto_resilience.png")
    plot_scenario_matrix(resil, highlights, HERE / "r1_scenario_kpi_matrix.png")
    plot_selected_bars(resil, highlights, HERE / "r1_selected_bars.png")

    print("\nTop-5 most resilient configs (by R_worst):")
    top = resil.sort_values("R_worst", ascending=False).head(5)
    for _, r in top.iterrows():
        row = nom_agg[nom_agg["config_id"] == r["config_id"]].iloc[0]
        print(f"  cfg {int(r['config_id']):2d}  bufs={_bufstr(row)}  "
              f"TP_nom={r['tp_nominal']:.3f}  R_worst={r['R_worst']:.3f}")

    print("\nBottom-5 least resilient configs (by R_worst):")
    bot = resil.sort_values("R_worst", ascending=True).head(5)
    for _, r in bot.iterrows():
        row = nom_agg[nom_agg["config_id"] == r["config_id"]].iloc[0]
        print(f"  cfg {int(r['config_id']):2d}  bufs={_bufstr(row)}  "
              f"TP_nom={r['tp_nominal']:.3f}  R_worst={r['R_worst']:.3f}")

    print(f"\nMean TP drop by scenario (over all {len(resil)} unique configs):")
    for d in DISRUPTIONS:
        print(f"  {SCENARIO_LABEL[d]:50s}  "
              f"mean drop = {resil[f'tp_drop_pct_{d}'].mean():5.2f}%  "
              f"max drop = {resil[f'tp_drop_pct_{d}'].max():5.2f}%")


if __name__ == "__main__":
    main()
