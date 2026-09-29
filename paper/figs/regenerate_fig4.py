"""Regenerate Fig 4 (Scenario S1: TH vs WIP).

Data sources (all committed to the repo, no re-simulation required):

    Framework Pareto front  : paper/figs/moo_pareto_solutions.csv
                              (19 non-dominated points; 15 unique buffer configs)
                              -- SAME source as Table 6 (tab:paretofront).
    Framework scatter       : data/moo_simulation_results.csv
                              (MOEA candidates from one representative seed).
    FACTS Pareto front      : Artikeldata/ALL_PARETO_for_HV/facts experiments/ParetoFacts1.csv
                              (representative single-seed FACTS Pareto, seed index 1).
    FACTS scatter           : Artikeldata/ALL_PARETO_for_HV/facts experiments/Baseline_buffer_config_1.csv
                              (FACTS optimization candidates from the same seed).
Both Pareto fronts are drawn from single representative seeds so that each
side's scatter and front come from the same underlying run. The prose HV
comparison averages HV over seeds; this figure is illustrative rather than
the source of the HV numbers reported in the text.

V1/V2/V3 are NOT highlighted on the plot: their ten-seed mean coordinates
(Tables 5 and 7) differ slightly from the single-seed Pareto values used
to draw the Framework front, so placing rings at either coordinate would
misrepresent the relationship. Table 7 lists the three points explicitly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_SCAT_PATH = ROOT / "data" / "moo_simulation_results.csv"
FRAMEWORK_FRONT_PATH = ROOT / "paper" / "figs" / "moo_pareto_solutions.csv"
FACTS_DIR = ROOT / "Artikeldata" / "ALL_PARETO_for_HV" / "facts experiments"
FACTS_SCAT_PATH = FACTS_DIR / "Baseline_buffer_config_1.csv"
FACTS_FRONT_PATH = FACTS_DIR / "ParetoFacts1.csv"
OUT_PNG = ROOT / "paper" / "figs" / "Fig4_TH_vs_WIP.png"

FW_COLOR    = "#3b6fdb"
FACTS_COLOR = "#e15b4a"


def load_framework_front():
    return pd.read_csv(FRAMEWORK_FRONT_PATH).sort_values("wip").reset_index(drop=True)


def load_framework_scatter():
    df = pd.read_csv(FRAMEWORK_SCAT_PATH)
    return df[["wip", "throughput"]].dropna()


def load_facts_scatter():
    raw = pd.read_csv(FACTS_SCAT_PATH, sep=";", dtype=str)
    return pd.DataFrame({
        "wip": pd.to_numeric(raw["WIP"].str.replace(",", "."), errors="coerce"),
        "throughput": pd.to_numeric(
            raw["Sink1_Throughput"].str.replace(",", "."), errors="coerce"
        ),
    }).dropna()


def load_facts_front():
    return pd.read_csv(FACTS_FRONT_PATH)[["wip", "throughput"]] \
             .sort_values("wip").reset_index(drop=True)


def stepped_front(df, x_col="wip", y_col="throughput"):
    """Return WIP/TH arrays that draw a minimize-WIP / maximize-TH step front."""
    d = df.sort_values(x_col).reset_index(drop=True)
    xs, ys = [], []
    best_y = -float("inf")
    for _, r in d.iterrows():
        x, y = r[x_col], r[y_col]
        if y > best_y:
            if xs:
                xs.append(x); ys.append(best_y)
            xs.append(x); ys.append(y)
            best_y = y
    return xs, ys


def main():
    fw_scat = load_framework_scatter()
    fw_front = load_framework_front()
    fa_scat = load_facts_scatter()
    fa_front = load_facts_front()

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    ax.scatter(fw_scat["wip"], fw_scat["throughput"],
               s=10, alpha=0.28, color=FW_COLOR, edgecolor="none",
               label="Framework candidates (one seed)", zorder=1)
    ax.scatter(fa_scat["wip"], fa_scat["throughput"],
               s=10, alpha=0.28, color=FACTS_COLOR, edgecolor="none",
               label="FACTS candidates (one seed)", zorder=1)

    xs, ys = stepped_front(fa_front)
    ax.plot(xs, ys, color=FACTS_COLOR, linewidth=1.6, alpha=0.9, zorder=3)
    ax.scatter(fa_front["wip"], fa_front["throughput"],
               s=42, color=FACTS_COLOR, edgecolor="white", linewidth=0.6,
               label=f"FACTS Pareto front ({len(fa_front)} pts)", zorder=4)

    xs, ys = stepped_front(fw_front)
    ax.plot(xs, ys, color=FW_COLOR, linewidth=1.6, alpha=0.9, zorder=3)
    ax.scatter(fw_front["wip"], fw_front["throughput"],
               s=42, color=FW_COLOR, edgecolor="white", linewidth=0.6,
               label=f"Framework Pareto front ({len(fw_front)} pts)", zorder=4)

    ax.set_xlim(10, 40)
    ax.set_ylim(25.9, 27.7)
    ax.set_xlabel("WIP (parts)")
    ax.set_ylabel("Throughput (parts/h)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(loc="lower right", framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")

    print(f"Framework Pareto: {len(fw_front)} points from {FRAMEWORK_FRONT_PATH.name}")
    print(f"FACTS Pareto    : {len(fa_front)} points from {FACTS_FRONT_PATH.name}")


if __name__ == "__main__":
    main()
