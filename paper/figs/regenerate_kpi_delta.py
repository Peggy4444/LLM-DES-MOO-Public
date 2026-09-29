"""
Regenerate fig:kpi-delta (model_comparison_kpis.png) from the freezed CSV.

Reads paper/figs/suggested_improvements_freezed.csv and draws three
side-by-side bar plots (TH, WIP, SEC) so the different KPI scales never
share an axis. Bars are annotated with their mean value; the FACTS
reference for each configuration is drawn as a dashed horizontal marker
inside each group, and the delta relative to the Initial (unoptimized)
Framework mean is written above each bar.

Run:
    python paper/figs/regenerate_kpi_delta.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "suggested_improvements_freezed.csv"
OUT_PATH = HERE / "model_comparison_kpis.png"

ORDER = ["Initial", "V1", "V2", "V3"]
DISPLAY = {
    "Initial": "Initial\n(unoptimized)",
    "V1": "V1\n(lowest-WIP)",
    "V2": "V2\n(knee-point)",
    "V3": "V3\n(highest-TH)",
}

FW_COLOR = "#3b6ea8"
FACTS_COLOR = "#c9522b"
DELTA_COLOR = "#4a4a4a"


def read_rows():
    with CSV_PATH.open() as f:
        rows = {r["variant"]: r for r in csv.DictReader(f)}
    return rows


def _pop_std(vals):
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _load_per_seed():
    per_seed_path = HERE / "suggested_improvements_freezed_perseed.csv"
    kpis = {v: {"th": [], "wip": [], "sec": []} for v in ORDER}
    with per_seed_path.open() as f:
        for r in csv.DictReader(f):
            v = r["variant"]
            kpis[v]["th"].append(float(r["th"]))
            kpis[v]["wip"].append(float(r["wip"]))
            kpis[v]["sec"].append(float(r["sec"]))
    return kpis


def make_figure():
    rows = read_rows()
    per_seed = _load_per_seed()

    means = {
        "th":  [sum(per_seed[v]["th"])/len(per_seed[v]["th"]) for v in ORDER],
        "wip": [sum(per_seed[v]["wip"])/len(per_seed[v]["wip"]) for v in ORDER],
        "sec": [sum(per_seed[v]["sec"])/len(per_seed[v]["sec"]) for v in ORDER],
    }
    stds = {
        "th":  [_pop_std(per_seed[v]["th"])  for v in ORDER],
        "wip": [_pop_std(per_seed[v]["wip"]) for v in ORDER],
        "sec": [_pop_std(per_seed[v]["sec"]) for v in ORDER],
    }
    facts = {
        "th":  [float(rows[v]["facts_th"])  for v in ORDER],
        "wip": [float(rows[v]["facts_wip"]) for v in ORDER],
        "sec": [float(rows[v]["facts_sec"]) for v in ORDER],
    }

    initial = {k: means[k][0] for k in ("th", "wip", "sec")}

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    kpi_specs = [
        ("th",  "Throughput (parts/h)",       axes[0], 0.02, 2),
        ("wip", "Work-in-process (parts)",    axes[1], 0.02, 2),
        ("sec", "Specific energy (kWh/part)", axes[2], 0.02, 4),
    ]
    x = np.arange(len(ORDER))
    bar_w = 0.55

    for key, ylabel, ax, headroom, decimals in kpi_specs:
        m = means[key]; s = stds[key]; f = facts[key]

        bars = ax.bar(
            x, m, width=bar_w,
            yerr=s, capsize=4,
            color=FW_COLOR, edgecolor="#1f4573", linewidth=0.8,
            label="Framework (10-seed mean)",
            error_kw=dict(ecolor="#1f4573", elinewidth=1.0),
        )

        # FACTS reference as a horizontal tick centred on each bar
        half = bar_w / 2 * 0.85
        for xi, fi in zip(x, f):
            ax.hlines(fi, xi - half, xi + half,
                      colors=FACTS_COLOR, linewidth=2.2,
                      linestyles=(0, (2, 1)),
                      label="FACTS reference" if xi == 0 else None)

        # y-range with generous top headroom so annotations never touch tick labels
        all_vals = list(m) + list(f) + [mi + si for mi, si in zip(m, s)]
        vmin = min(0 if key != "sec" else min(all_vals) * 0.9, min(all_vals))
        vmax = max(all_vals)
        span = vmax - vmin
        pad_bottom = 0.05 * span if key == "sec" else 0
        pad_top = 0.32 * span
        ax.set_ylim(vmin - pad_bottom, vmax + pad_top)

        # Value labels above bar top + std AND above FACTS line, whichever is higher,
        # so they never collide with the dashed FACTS marker.
        fmt = f"{{:.{decimals}f}}"
        for xi, mi, si, fi in zip(x, m, s, f):
            top_of_bar = max(mi + si, fi)
            label_y = top_of_bar + 0.05 * span
            ax.text(xi, label_y, fmt.format(mi),
                    ha="center", va="bottom",
                    fontsize=9, color="#1a1a1a")
            if xi > 0:  # relative delta to Initial
                delta_pct = (mi - initial[key]) / initial[key] * 100
                sign = "+" if delta_pct >= 0 else ""
                ax.text(xi, label_y + 0.11 * span,
                        f"({sign}{delta_pct:.1f}%)",
                        ha="center", va="bottom",
                        fontsize=8.5, color=DELTA_COLOR, style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[v] for v in ORDER], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Single figure-level legend below the axes so nothing overlaps the plots
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=FW_COLOR, edgecolor="#1f4573"),
        plt.Line2D([0], [0], color=FACTS_COLOR, linewidth=2.2, linestyle=(0, (2, 1))),
    ]
    labels = [
        "Framework 10-seed mean (± pop. std)",
        "FACTS reference (same buffer config)",
    ]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    make_figure()
