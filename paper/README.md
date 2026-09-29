# C&IE manuscript -- draft

Draft of the Computers & Industrial Engineering paper based on the
Krusten/Norberg MSc thesis, descoped per the June 2026 meeting:

- formal user study dropped; informal industrial reflections moved
  into Section~6;
- ablation reduced to 7 configurations (C0, C1, C2, C3, C4, C6, C7).

The manuscript follows the Elsevier CAS template (`cas-sc` document
class, single-column layout), as required by the
**Computers & Industrial Engineering** Guide for Authors.

## Files

- `main.tex` -- manuscript draft using `cas-sc` + `natbib` (author-year).
- `references.bib` -- bibliography (author-year compatible).

## Opening it in Overleaf

The Elsevier CAS template is available in the Overleaf gallery.
Two ways to set it up:

**Option A -- start from the template in the gallery (recommended).**
1. Open Overleaf and search for "Elsevier CAS template" in the
   gallery; click *Open as Template*. This gives you a project that
   already contains the `cas-sc.cls` class file, bibliography styles
   (`cas-model2-names.bst`), and supporting CAS macro files.
2. Delete the boilerplate `main.tex` from that gallery project and
   upload our `main.tex` and `references.bib` in its place.
3. Set the main document to `main.tex` and the compiler to
   **pdfLaTeX**.
4. Compile twice (BibTeX runs on the second pass).

**Option B -- upload a zip.**
1. Download `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst`
   from the Elsevier CAS bundle (CTAN: `els-cas-templates`).
2. Zip them together with `main.tex` and `references.bib` and upload
   to Overleaf as a new project.

## What's still TODO before submission

Search the source for `\TODO{...}` markers -- they highlight every
placeholder. Categories:

- **Figures**: port from the thesis PDF (architecture, Case~A layout,
  Pareto fronts, KPI bars, SCORE rankings, WIP/SEC scatter).
- **Tables**: related-work table (Section~2.6), KPI fidelity table
  (5.1), Pareto-point table (5.2), ablation matrix (5.4), cost
  summary (5.5).
- **Declarations**: competing interest, generative AI use, data
  availability link. (CRediT roles are already populated via
  `\credit{}` per author and rendered by `\printcredits`.)
- **Numerical placeholders** (marked in blue with `\NUM{...}`) should
  be regenerated once the reduced ablation is finished. The numbers
  currently shown are copied from the thesis baseline run.

## CRediT contributor mapping (placeholder)

Currently set in `main.tex`. Verify and adjust before submission.

| Author | CRediT roles |
|---|---|
| Krusten, Norberg | Conceptualization, Methodology, Software, Validation, Investigation, Writing -- original draft, Visualization |
| Schmitt, Gegenmantel | Conceptualization, Methodology, Software, Writing -- review & editing |
| Rahimian (corr.) | Conceptualization, Methodology, Writing -- review & editing, Supervision, Project administration |
| Urenda Moris | Conceptualization, Resources, Writing -- review & editing, Supervision |
| M\aa rtensson | Resources, Writing -- review & editing |
| Amouzgar | Conceptualization, Methodology, Supervision, Writing -- review & editing |

## Special-issue target

Recommendation: submit to the C&IE special issue *"AI, Analytics and
Decision-Support for Human-Centric, Resilient and Sustainable
Industrial Systems aligned with Industry 5.0"* if the deadline is
still open at submission time. Otherwise: regular track.

## Switching to longer frontmatter

If the title block + author list + abstract overflows one page,
swap the document class options to enable the longer frontmatter
layout:

```latex
\documentclass[a4paper,fleqn,longmktitle]{cas-sc}
```
