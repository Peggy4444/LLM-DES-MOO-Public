# Participant packet

**An external evaluation of an LLM-based decision-support framework for
simulation-driven production-line optimization.**

Facilitator: Pegah Rahimian, Uppsala University
Contact: rahimian.pegah@gmail.com
Date: __________________  Participant ID (assigned): __________

---

## Consent

You are invited to take part in an approximately 90-minute session in
which you will use a research prototype and give your professional
opinion on the outputs it produces. This session is part of the
evaluation for an academic paper.

**What happens.** You will be shown the tool briefly, and then asked to
perform three short tasks and answer questions about the outputs.

**Recording.** With your permission, we will record the screen and your
audio to capture your comments accurately. Recordings will not be shared
publicly and will be deleted after the paper is published.

**Use of your input.** Your comments may be quoted anonymously in the
paper, referenced only by your role and industry sector (e.g., "a
simulation engineer at a European truck OEM"). Your name and employer
will not appear.

**Your rights.** You may stop the session, skip any question, or withdraw
your data at any time up to submission of the paper, without giving a
reason.

**Data.** No production data from your company is used in this session.
You will interact only with a public academic case study.

- [ ] I consent to participate.
- [ ] I consent to screen and audio recording.
- [ ] I consent to being quoted anonymously by role and sector.

Signature: ____________________________  Date: __________

---

## Background form

Please answer as briefly as you like. Skip anything you prefer not to
answer.

1. Your current role:
   ______________________________________________________________

2. Employer sector (truck / passenger cars / components / other):
   ______________________________________________________________

3. Years of experience in industrial simulation or production
   engineering: ________

4. Simulation tools you have used in the past 3 years (circle all):
   FACTS Analyzer   ·   Plant Simulation   ·   Arena   ·   AnyLogic
   ·   SimPy   ·   ExtendSim   ·   Simio   ·   Other: __________

5. Multi-objective optimization tools you have used (circle all):
   pymoo   ·   MATLAB gamultiobj   ·   ModeFRONTIER   ·   HEEDS
   ·   FACTS built-in optimizer   ·   Custom / in-house
   ·   Other: __________   ·   None

6. How often do you personally build or modify DES models?
   Weekly   ·   Monthly   ·   A few times a year   ·   Rarely   ·   Never

7. How often do you personally interpret optimization results (Pareto
   fronts, sensitivity analyses)?
   Weekly   ·   Monthly   ·   A few times a year   ·   Rarely   ·   Never

8. Have you used ChatGPT / Claude / Copilot at work in the past 6 months?
   Yes   ·   No
   If yes, briefly for what: ________________________________________

9. Is your team currently allowed to send production data (event logs,
   parameters) to a cloud LLM provider (OpenAI, Anthropic, Google)?
   Yes   ·   No   ·   Unsure   ·   Only with anonymization

10. Any comments on what would need to be true before your company
    would try a tool like this? ______________________________________
    _____________________________________________________________

---

## Rating sheet — per-artifact ratings

For each artifact the framework produced, rate five dimensions on a
1–5 scale. Use the shorthand: 1 = very poor, 2 = poor, 3 = acceptable,
4 = good, 5 = excellent. Add one sentence of justification where you
feel it matters.

**Dimension definitions:**
- **Correctness** — Does the output correctly reflect the underlying data
  or system?
- **Clarity** — Is it understandable without extra explanation?
- **Trust** — Would you rely on it to make a decision?
- **Actionability** — Could you or your team act on this output?
- **Verification effort** — How easy is it to check whether it is right?
  (1 = very hard to verify, 5 = easy to verify)

| Artifact | Correct | Clear | Trust | Action | Verify | Comment |
|---|---|---|---|---|---|---|
| Mermaid topology diagram (S2 visualizer) |  |  |  |  |  |  |
| Generated SimPy code (S2 builder) |  |  |  |  |  |  |
| Pareto front & knee/low-WIP/high-TH labels (S3 optimizer) |  |  |  |  |  |  |
| SCORE alleviation ranking (S4 bottleneck) |  |  |  |  |  |  |
| SCORE natural-language rationale (S4 bottleneck) |  |  |  |  |  |  |
| Adapter agent KPI report (S5 evaluator) |  |  |  |  |  |  |

---

## Rating sheet — comparison to your current workflow

For each artifact, compare how you would obtain the same insight today
(e.g., with FACTS Analyzer, Plant Simulation, manual analysis, or an
Excel model). Rate on −2 to +2:

- −2 = this tool is much worse than my current approach
- −1 = somewhat worse
-  0 = about the same
- +1 = somewhat better
- +2 = much better

Also note **what you currently use** for that step.

| Artifact | What I use today | Rating (−2…+2) | One-line why |
|---|---|---|---|
| Topology diagram |  |  |  |
| Pareto interpretation |  |  |  |
| Bottleneck ranking |  |  |  |
| Bottleneck rationale (natural language) |  |  |  |
| Model adaptation |  |  |  |

---

## Paired interpretation tasks (T1, T2, T3)

The facilitator will guide you through three short tasks. For each, you
will first see raw data and give your own interpretation aloud, then see
the LLM's interpretation of the same data. After each task, answer the
four questions below.

### Task T1 — DES topology

**Q1.** Did the LLM's diagram/interpretation match your mental model of
the line? (Yes / Mostly / Partially / No)

**Q2.** Did the LLM include anything you consider wrong? If yes, what?
______________________________________________________________________

**Q3.** Did the LLM omit anything important you would have said?
______________________________________________________________________

**Q4.** If you were sending this to a production-floor colleague, would
you send your interpretation, the LLM's, or a merged version? Why?
______________________________________________________________________

Agreement rating (circle one): Full · Partial · None

---

### Task T2 — Pareto front interpretation

**Q1.** Did the LLM's recommended Pareto point match the one you would
have chosen? (Same point / Adjacent point / Different / Not comparable)

**Q2.** Did the LLM include anything you consider wrong or misleading?
______________________________________________________________________

**Q3.** Did the LLM omit any concern you raised? (e.g., about robustness,
about a specific machine, about feasibility)
______________________________________________________________________

**Q4.** Would you show the LLM's summary to a plant manager as-is,
edit it first, or write your own? Why?
______________________________________________________________________

Agreement rating (circle one): Full · Partial · None

---

### Task T3 — Bottleneck ranking and rationale

**Q1.** Did the LLM's top-ranked alleviation match the one you would
have picked? (Same / Adjacent / Different / Not comparable)

**Q2.** Did the LLM's natural-language rationale correctly reflect what
the SCORE frequency numbers say?
______________________________________________________________________

**Q3.** Did the LLM omit any bottleneck you consider important?
______________________________________________________________________

**Q4.** If your maintenance team read only the LLM's rationale (not the
underlying numbers), would they take the right action?
______________________________________________________________________

Agreement rating (circle one): Full · Partial · None

---

## Overall usability — System Usability Scale (SUS)

Rate each statement 1 (strongly disagree) to 5 (strongly agree).

| # | Statement | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | I think that I would like to use this system frequently. |  |  |  |  |  |
| 2 | I found the system unnecessarily complex. |  |  |  |  |  |
| 3 | I thought the system was easy to use. |  |  |  |  |  |
| 4 | I think that I would need the support of a technical person to use this system. |  |  |  |  |  |
| 5 | I found the various functions in this system well integrated. |  |  |  |  |  |
| 6 | I thought there was too much inconsistency in this system. |  |  |  |  |  |
| 7 | I would imagine that most people would learn to use this system very quickly. |  |  |  |  |  |
| 8 | I found the system very cumbersome to use. |  |  |  |  |  |
| 9 | I felt very confident using the system. |  |  |  |  |  |
| 10 | I needed to learn a lot of things before I could get going with this system. |  |  |  |  |  |

(SUS scoring is standard; the facilitator will compute the 0–100 score
after the session.)

---

## Trust in the tool (Jian, Bisantz & Drury 2000, short form)

Rate 1 (not at all) to 5 (very much).

| # | Statement | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | The system is deceptive. |  |  |  |  |  |
| 2 | The system behaves in an underhanded manner. |  |  |  |  |  |
| 3 | I am suspicious of the system's intent, action, or outputs. |  |  |  |  |  |
| 4 | I am wary of the system. |  |  |  |  |  |
| 5 | The system's actions will have a harmful or injurious outcome. |  |  |  |  |  |
| 6 | I am confident in the system. |  |  |  |  |  |
| 7 | The system provides security. |  |  |  |  |  |
| 8 | The system has integrity. |  |  |  |  |  |
| 9 | The system is dependable. |  |  |  |  |  |
| 10 | The system is reliable. |  |  |  |  |  |
| 11 | I can trust the system. |  |  |  |  |  |
| 12 | I am familiar with the system. |  |  |  |  |  |

---

## Expertise barrier item

Rate 1 (strongly disagree) to 5 (strongly agree):

*"Compared to my current workflow, this tool would allow a colleague
without simulation-specialist training to reach a comparable analysis
of a production line."*

Rating: ___   One-sentence why: ______________________________________
______________________________________________________________________

---

## Final open comments

Anything else you would like the researchers to know?
______________________________________________________________________
______________________________________________________________________
______________________________________________________________________

**Thank you.** The researcher will send you a 1-page summary of what
they heard from you within 48 hours, and ask you to confirm that it
reflects your view correctly before it is used in the paper.

Preferred email for the summary: ____________________________________
