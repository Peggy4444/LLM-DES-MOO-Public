"""R2 plots: bottleneck-ranking matrix and TP-gain heatmap."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
RANKING = HERE / "r2_score_ranking.csv"

SCENARIO_LABEL = {
    "nominal":                     "Nominal",
    "D1_press1_aging":             "D1: Press 1 aging",
    "D2_systemic_degradation":     "D2: Systemic degradation",
    "D3_feeder_failure":           "D3: Feeder failure",
}
SCEN_ORDER = ["nominal", "D1_press1_aging", "D2_systemic_degradation", "D3_feeder_failure"]

MACHINE_ORDER = [
    "Loading robot", "Conveyor belt", "Washing machine", "Hantering cell",
    "Presses cell 1", "Presses cell 2", "Quality station cell",
]


def plot_gain_heatmap(df, path):
    # Row = machine, col = scenario, value = TP gain %.
    mat = np.zeros((len(MACHINE_ORDER), len(SCEN_ORDER)))
    for i, m in enumerate(MACHINE_ORDER):
        for j, sc in enumerate(SCEN_ORDER):
            row = df[(df.scenario == sc) & (df.machine == m)].iloc[0]
            mat[i, j] = row["tp_gain"] / row["tp_baseline"] * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    vmax = max(abs(mat.min()), mat.max())
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SCEN_ORDER)))
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCEN_ORDER], rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(MACHINE_ORDER)))
    ax.set_yticklabels(MACHINE_ORDER, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Throughput gain from SCORE relaxation (%)", fontsize=9)
    ax.set_title("R2: Scenario-conditioned SCORE bottleneck sensitivity\n"
                 "Relaxation per machine: avail $+5$ pp, MTTR $\\times$0.7, PT $\\times$0.9")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ranking_matrix(df, path):
    # Row = machine, col = scenario, value = rank (1 = biggest bottleneck).
    mat = np.zeros((len(MACHINE_ORDER), len(SCEN_ORDER)), dtype=int)
    for i, m in enumerate(MACHINE_ORDER):
        for j, sc in enumerate(SCEN_ORDER):
            row = df[(df.scenario == sc) & (df.machine == m)].iloc[0]
            mat[i, j] = int(row["rank"])

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    im = ax.imshow(mat, cmap="viridis_r", vmin=1, vmax=len(MACHINE_ORDER),
                   aspect="auto")
    ax.set_xticks(range(len(SCEN_ORDER)))
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCEN_ORDER], rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(len(MACHINE_ORDER)))
    ax.set_yticklabels(MACHINE_ORDER, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v <= 3 else "black"
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bottleneck rank (1 = most improvable)", fontsize=9)
    ax.set_title("R2: Bottleneck ranking under each scenario\n"
                 "Ranks 1--2 stable  =>  Press 1 & Press 2 are robust bottlenecks")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(RANKING)
    plot_gain_heatmap(df, HERE / "r2_gain_heatmap.png")
    plot_ranking_matrix(df, HERE / "r2_ranking_matrix.png")
    print("R2 plots written:")
    print("  r2_gain_heatmap.png")
    print("  r2_ranking_matrix.png")

    # Compact summary table for the manuscript.
    print("\nTop-3 bottlenecks per scenario:")
    for sc in SCEN_ORDER:
        top = df[df.scenario == sc].sort_values("rank").head(3)
        row = " > ".join(
            f"{r.machine} ({r.tp_gain/r.tp_baseline*100:+.2f}%)" for _, r in top.iterrows()
        )
        print(f"  {SCENARIO_LABEL[sc]:35s} -> {row}")


if __name__ == "__main__":
    main()
