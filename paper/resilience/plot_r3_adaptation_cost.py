"""
R3 -- Adaptation-cost analysis for what-if resilience queries.

Compares the marginal cost (USD) and wall-clock time (minutes) of answering
a resilience what-if with the LLM-agent framework against a
FACTS-Analyzer-plus-expert baseline. The LLM numbers come from the paper's
Table 5 (baseline per-run averages); the FACTS numbers use conservative
industry estimates for expert setup and per-scenario rework time.

This is a comparative *plot* — no additional simulations required.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent

# LLM-agent framework (paper Table 5 means across 10 baseline seeds).
LLM = {
    "initial_setup_usd":    0.316,
    "initial_setup_min":    992 / 60,        # ~16.5 min
    "per_query_usd":        0.15,             # adapter + re-optim/SCORE, conservative
    "per_query_min":        8.0,              # adapter + re-run
}

# FACTS-Analyzer + expert baseline (industry estimates, conservative).
# Expert loaded rate at ~USD 100/hour to keep the comparison defensible.
EXPERT_HOURLY_USD = 100.0
FACTS = {
    "initial_setup_hours":   80,              # ~2 weeks of specialist time (paper's own claim)
    "per_query_hours":       2.0,             # scenario reconfiguration + rerun
    "license_annual_usd":    None,            # not amortized here for fairness
}
FACTS_INITIAL_USD = FACTS["initial_setup_hours"] * EXPERT_HOURLY_USD
FACTS_PER_QUERY_USD = FACTS["per_query_hours"] * EXPERT_HOURLY_USD
FACTS_INITIAL_MIN   = FACTS["initial_setup_hours"] * 60
FACTS_PER_QUERY_MIN = FACTS["per_query_hours"] * 60

QUERY_COUNTS = [0, 1, 3, 5, 10]


def plot_cumulative_cost(path):
    llm_cost   = [LLM["initial_setup_usd"] + n * LLM["per_query_usd"] for n in QUERY_COUNTS]
    facts_cost = [FACTS_INITIAL_USD + n * FACTS_PER_QUERY_USD for n in QUERY_COUNTS]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(QUERY_COUNTS, facts_cost, marker="s", linewidth=2, label="FACTS + expert", color="firebrick")
    ax.plot(QUERY_COUNTS, llm_cost, marker="o", linewidth=2, label="LLM-agent framework", color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("Number of what-if resilience queries answered")
    ax.set_ylabel("Cumulative cost (USD, log scale)")
    ax.set_title("R3: Cumulative cost of resilience what-if queries")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center right")

    # Annotate the 5-query point.
    n5 = QUERY_COUNTS.index(5)
    ax.annotate(f"${llm_cost[n5]:.2f}", (QUERY_COUNTS[n5], llm_cost[n5]),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=9, color="steelblue")
    ax.annotate(f"${facts_cost[n5]:,.0f}", (QUERY_COUNTS[n5], facts_cost[n5]),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=9, color="firebrick")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_time_to_answer(path):
    labels = ["Initial\nsetup", "Per what-if\nquery"]
    llm_times   = [LLM["initial_setup_min"],   LLM["per_query_min"]]
    facts_times = [FACTS_INITIAL_MIN,           FACTS_PER_QUERY_MIN]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(x - w/2, llm_times,   w, label="LLM-agent framework", color="steelblue")
    ax.bar(x + w/2, facts_times, w, label="FACTS + expert",       color="firebrick")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time to answer (minutes, log scale)")
    ax.set_title("R3: Time-to-answer for resilience queries")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend()

    # Annotate.
    for xi, (a, b) in enumerate(zip(llm_times, facts_times)):
        ax.text(xi - w/2, a * 1.15, f"{a:.0f} min", ha="center", fontsize=8, color="steelblue")
        if b >= 60:
            lbl = f"{b/60:.0f} h" if b < 60*24 else f"{b/60/24:.0f} d"
        else:
            lbl = f"{b:.0f} min"
        ax.text(xi + w/2, b * 1.15, lbl, ha="center", fontsize=8, color="firebrick")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p1 = HERE / "r3_cumulative_cost.png"
    p2 = HERE / "r3_time_to_answer.png"
    plot_cumulative_cost(p1)
    plot_time_to_answer(p2)

    print("R3 plots written:")
    print(f"  {p1.name}")
    print(f"  {p2.name}")
    print()
    print("Numbers used in R3:")
    print(f"  LLM initial:    ${LLM['initial_setup_usd']:.3f}   {LLM['initial_setup_min']:.1f} min")
    print(f"  LLM per query:  ${LLM['per_query_usd']:.3f}   {LLM['per_query_min']:.1f} min")
    print(f"  FACTS initial:  ${FACTS_INITIAL_USD:,.0f}   {FACTS['initial_setup_hours']:.0f} h")
    print(f"  FACTS per q:    ${FACTS_PER_QUERY_USD:,.0f}   {FACTS['per_query_hours']:.1f} h")
    print()
    print("5-query resilience study:")
    llm_5   = LLM['initial_setup_usd'] + 5 * LLM['per_query_usd']
    facts_5 = FACTS_INITIAL_USD + 5 * FACTS_PER_QUERY_USD
    print(f"  LLM:    ${llm_5:.2f}")
    print(f"  FACTS:  ${facts_5:,.0f}")
    print(f"  Ratio:  {facts_5 / llm_5:.0f}x cheaper with LLM framework")


if __name__ == "__main__":
    main()
