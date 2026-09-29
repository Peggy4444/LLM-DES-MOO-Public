# Study goal — external practitioner evaluation

## One-sentence goal

To evaluate whether the LLM-generated human-facing artifacts of the framework
(topology diagrams, Pareto-front interpretations, SCORE alleviation rankings,
natural-language rationales, adaptation reports) are **accurate to what a
domain expert would independently conclude from the same underlying data**
and **trustworthy enough for practitioners to act on** — as judged by two
domain experts external to the host company.

## What this study validates

1. **Interpretive fidelity.** Does the LLM's natural-language interpretation
   of a Pareto front, a SCORE frequency table, or a topology parameter set
   agree with what an experienced simulation practitioner would say about
   the same raw data?
2. **Trust and actionability.** Would practitioners at other OEMs act on
   the outputs the framework produces? Which outputs, and under what
   preconditions?
3. **Usability of the human-in-the-loop design.** Are the six HITL
   checkpoints (H1–H6) at the right points, and are the artifacts
   inspectable enough to audit?
4. **Adoption barriers.** What organizational, IT, or data-governance
   conditions must hold before the framework could be used on a real line?

## What this study does NOT validate

- KPI numerical accuracy — already covered by the FACTS Analyzer comparison
  in §5.1.
- LLM code-generation reliability — already covered by the ablation in §5.4.
- Generalizability across many industrial sites — N=2 is a formative study,
  not a statistical one.
- The LLM itself as a language model — that is not this paper's contribution.

## Fit into the manuscript

- Replaces or complements §7.4 "Industrial reflections" (currently informal,
  host-company only, N=3) with an external, structured evaluation (N=2,
  Scania + Volvo Group).
- Directly addresses §7.6 Limitation 3 (no structured usability study) and
  §8 future work ("structured studies with industrial practitioners are
  needed").
- Provides material for a new short subsection, e.g. §7.4b *External
  practitioner evaluation*, or a rewrite of §7.4.

## How to frame it in reviews

- N=2 is small; frame as a **formative expert evaluation** (Nielsen 1994
  precedent: 3–5 experts detect most usability issues), not a user study.
- Participants are **external** to the host OEM: this is what makes N=2
  defensible — it addresses a distinct threat to validity (host-company
  bias) rather than trying to make statistical claims.
- Report medians and ranges, not means; quote verbatim; do not
  hypothesis-test.

## The single most publishable output

The **before/after interpretation comparison** (Protocol §04): each
participant first interprets the raw data (Pareto scatter, SCORE frequency
table, machine parameter list) without seeing the LLM output; then sees the
LLM's interpretation; then judges agreement, omissions, and errors. This
gives a concrete, defensible answer to a question no other paper in the
adjacent literature has answered: *"When the LLM narrates a simulation
result, does the narration match what an expert would say?"*
