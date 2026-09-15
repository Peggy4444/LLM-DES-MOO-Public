"""Interactive Streamlit wizard that mirrors main.py end-to-end.

Run with:
    streamlit run interactive_app.py

Same private gating as streamlit_app.py: enter an OpenAI API key. Optional
STREAMLIT_APP_PASSCODE env var adds a passphrase.

The wizard walks the user through every stage of main.py:
  1. Configuration
  2. Event-log analysis (process mining)
  3. Build initial DES model
  4. Inspect + run initial model to retrieve KPIs
  5. Optional manual edit
  6. Visualize model (mermaid flow chart)
  7. Choose optimization path (bottleneck / MOO / skip)
  8. MOO or SCORE stage with sub-steps
  9. Optional human adaptation input
 10. CPD safety check
 11. Adapt-loop: apply each suggestion and retrieve KPIs
 12. LLM evaluation + KPI comparison chart
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Framework imports (rooted at repo root).
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from processmining import eventlog, metrics
from agents.builder import ModelBuilder
from agents.optimizer import Blueprintoptimizer
from agents.bottleneck import BottleneckOptimizer
from agents.adapter import Modeladaptor
from agents.evaluator import Evaluater
from agents.cpdagent import CPD
from agents.visualizer import Modelvisualizer
from helpers.other_helpers import (
    save_model,
    remove_code_wrappers,
    retrieve_KPIs,
    visualize_results,
    inspect_code,
)


PASSCODE = os.getenv("STREAMLIT_APP_PASSCODE", "").strip()
RESULTS_DIR = APP_ROOT / "results"
DATA_DIR = APP_ROOT / "data"
BLUEPRINT_DIR = APP_ROOT / "blueprint"

st.set_page_config(
    page_title="LLM-DES-MOO Interactive",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Defaults (from main.py)
# ---------------------------------------------------------------------------

DEFAULT_BUFFERS = (
    "PostLoadingBuffer(Capacity = 2, processtime = 10), "
    "PostConveyorBuffer(Capacity = 2, processtime = 10), "
    "PostWashingBuffer(Capacity = 2, processtime = 10), "
    "PrePress1Buffer(Capacity = 3, processtime = 32), "
    "PrePress2Buffer(Capacity = 3, processtime = 32), "
    "PostPress1&Press2Buffer(Capacity = 3, processtime = 32)"
)
DEFAULT_DEFECTS = "Defect rate = 0.089, defect sink = defect, initiated at Qualitystation"
DEFAULT_CPD = (
    "1. The presses need to have a processtime of at least 60s. "
    "All buffer capacities must be kept at the same original level. "
    "The original buffer capacities were: PostLoadingBuffer(Capacity = 2), "
    "PostConveyorBuffer(Capacity = 2), PostWashingBuffer(Capacity = 2), "
    "PrePress1Buffer(Capacity = 3), PrePress2Buffer(Capacity = 3), "
    "PostPress1&Press2Buffer(Capacity = 3)"
)
DEFAULT_NOTE = (
    "Production stops from friday 17.00 till saturday 07:00 and from "
    "saturday 17:00 till sunday 07:00."
)
DEFAULT_EVENT_LOG = str((DATA_DIR / "workingtest.csv").relative_to(APP_ROOT))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _validate_openai_key(api_key: str) -> tuple[bool, str]:
    try:
        from openai import OpenAI, AuthenticationError, APIError
    except Exception as exc:  # pragma: no cover
        return False, f"openai import failed: {exc}"

    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, "Key accepted."
    except AuthenticationError:
        return False, "OpenAI rejected the key (authentication failed)."
    except APIError as exc:
        return False, f"OpenAI API error: {exc}"
    except Exception as exc:
        return False, f"Could not validate key: {exc}"


def _render_login() -> None:
    st.title("🔒 LLM-DES-MOO Interactive Wizard")
    st.write(
        "This wizard walks you through the full framework — event-log "
        "analysis, LLM model building, MOO / bottleneck optimisation, "
        "adaptation, and evaluation. Enter an OpenAI API key to unlock."
    )
    with st.form("login_form"):
        if PASSCODE:
            passcode_in = st.text_input("Passphrase", type="password")
        else:
            passcode_in = None
        api_key = st.text_input(
            "OpenAI API key", type="password", placeholder="sk-..."
        )
        submitted = st.form_submit_button("Unlock", type="primary")
    if not submitted:
        return
    if PASSCODE and (passcode_in or "").strip() != PASSCODE:
        st.error("Wrong passphrase.")
        return
    if not api_key.strip():
        st.error("API key is required.")
        return
    with st.spinner("Validating key with OpenAI..."):
        ok, msg = _validate_openai_key(api_key.strip())
    if not ok:
        st.error(msg)
        return
    st.session_state["api_key"] = api_key.strip()
    st.session_state["authed"] = True
    st.rerun()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("stage", 0)
    ss.setdefault("config", {
        "event_log": DEFAULT_EVENT_LOG,
        "buffers": DEFAULT_BUFFERS,
        "defects": DEFAULT_DEFECTS,
        "cpd_info": DEFAULT_CPD,
        "manual_note": DEFAULT_NOTE,
        "sim_time": 8 * 24 * 3600,
        "warmup_seconds": 24 * 3600,
        "replications": 10,
    })
    ss.setdefault("kpi_results", [])


def _advance() -> None:
    st.session_state["stage"] += 1
    st.rerun()


def _reset() -> None:
    keep = {"authed", "api_key"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.rerun()


def _client():
    from openai import OpenAI
    return OpenAI(api_key=st.session_state["api_key"])


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


_MERMAID_HEADERS = (
    "flowchart",
    "graph ",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "gantt",
    "pie",
    "journey",
    "gitGraph",
    "mindmap",
)


def _clean_mermaid_source(src: str) -> str:
    """Strip fences and any preamble the LLM added before the diagram."""
    s = (src or "").strip()
    # Strip a leading ```mermaid / ```md / ``` fence.
    s = re.sub(r"^```[a-zA-Z]*\s*\n", "", s)
    if s.endswith("```"):
        s = s[:-3].rstrip()
    # Find the first line that looks like a mermaid diagram header.
    lines = s.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(h) for h in _MERMAID_HEADERS):
            return "\n".join(lines[i:]).strip()
    return s


def _render_mermaid_inline(mmd_source: str, height: int = 500) -> None:
    import html as _html

    cleaned = _clean_mermaid_source(mmd_source)
    # HTML-escape so `<`, `>`, `&` inside labels don't break the parent
    # HTML. Mermaid reads `textContent`, which the browser un-escapes.
    escaped = _html.escape(cleaned)
    page = f"""
    <div id="mmd" class="mermaid">{escaped}</div>
    <pre id="mmd-error" style="color:#b00020;white-space:pre-wrap;font-family:monospace;font-size:12px;display:none"></pre>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: false, theme: 'default', securityLevel: 'loose' }});
      try {{
        await mermaid.run({{ querySelector: '.mermaid' }});
      }} catch (err) {{
        const el = document.getElementById('mmd-error');
        el.style.display = 'block';
        el.textContent = 'Mermaid render failed:\\n' + (err && err.message ? err.message : err);
      }}
    </script>
    """
    components.html(page, height=height, scrolling=True)


# ---------------------------------------------------------------------------
# Stage 0: Configuration
# ---------------------------------------------------------------------------


def _stage_config(active: bool) -> None:
    cfg = st.session_state["config"]
    if not active:
        with st.expander("✅ 1. Configuration", expanded=False):
            st.write(f"**Event log:** `{cfg['event_log']}`")
            st.write(f"**Sim time / warmup:** {cfg['sim_time']} s / {cfg['warmup_seconds']} s, replications = {cfg['replications']}")
            with st.expander("Buffers / defects / CPD / note", expanded=False):
                st.text(cfg["buffers"])
                st.text(cfg["defects"])
                st.text(cfg["cpd_info"])
                st.text(cfg["manual_note"])
        return

    st.header("1. Configuration")
    st.caption(
        "These correspond to the constants at the top of `main.py`. "
        "The defaults are pre-filled. Adjust before starting the run."
    )
    with st.form("config_form"):
        c1, c2 = st.columns(2)
        with c1:
            event_log = st.text_input("Event log path", value=cfg["event_log"])
            sim_time = st.number_input(
                "SIM_TIME (seconds)", value=int(cfg["sim_time"]), step=3600
            )
            warmup = st.number_input(
                "WARMUP_SECONDS", value=int(cfg["warmup_seconds"]), step=3600
            )
            reps = st.number_input(
                "Replications", value=int(cfg["replications"]), min_value=1, step=1
            )
        with c2:
            note = st.text_area("Manual note (passed to builder)", value=cfg["manual_note"], height=100)
        buffers = st.text_area("Buffer definitions", value=cfg["buffers"], height=100)
        defects = st.text_area("Defect info", value=cfg["defects"], height=68)
        cpd = st.text_area("Critical Process Definition (CPD)", value=cfg["cpd_info"], height=120)
        submitted = st.form_submit_button("Save & continue", type="primary")
    if submitted:
        st.session_state["config"] = {
            "event_log": event_log.strip(),
            "buffers": buffers.strip(),
            "defects": defects.strip(),
            "cpd_info": cpd.strip(),
            "manual_note": note.strip(),
            "sim_time": int(sim_time),
            "warmup_seconds": int(warmup),
            "replications": int(reps),
        }
        _advance()


# ---------------------------------------------------------------------------
# Stage 1: Process mining
# ---------------------------------------------------------------------------


def _stage_process_mining(active: bool) -> None:
    ss = st.session_state
    if not active and "stations_md" not in ss:
        return
    title = "2. Event-log analysis"

    def _show():
        st.markdown("**Station metrics**")
        st.dataframe(ss["stations_df"], use_container_width=True, hide_index=True)
        st.markdown("**Directly-follows sequence**")
        st.code(ss["sequence_text"], language="text")

    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    cfg = ss["config"]
    st.caption(f"Reading `{cfg['event_log']}` and running pm4py footprint discovery.")
    if st.button("Run event-log analysis", type="primary"):
        path = APP_ROOT / cfg["event_log"] if not os.path.isabs(cfg["event_log"]) else Path(cfg["event_log"])
        if not path.exists():
            st.error(f"Event log not found: {path}")
            return
        with st.spinner("Loading and preprocessing event log..."):
            df_raw = eventlog.load(path)
            df_clean = eventlog.preprocess(df_raw)
            stations_df = metrics.compute(df_clean)
            stations_md = stations_df.to_string(index=False)
            sequence_text = eventlog.to_sequence_text(df_clean)
        ss["df_clean"] = df_clean
        ss["stations_df"] = stations_df
        ss["stations_md"] = stations_md
        ss["sequence_text"] = sequence_text
        _advance()


# ---------------------------------------------------------------------------
# Stage 2: Build initial model
# ---------------------------------------------------------------------------


def _read_blueprints() -> dict[str, str]:
    return {
        "util": (BLUEPRINT_DIR / "blueprint_util.py").read_text(encoding="utf-8"),
        "moo_buffer": (BLUEPRINT_DIR / "blueprint_MOO_buffer.py").read_text(encoding="utf-8"),
        "moo_availability": (BLUEPRINT_DIR / "blueprint_MOO_availability.py").read_text(encoding="utf-8"),
        "score": (BLUEPRINT_DIR / "blueprint_SCORE.py").read_text(encoding="utf-8"),
    }


def _stage_build(active: bool) -> None:
    ss = st.session_state
    title = "3. Build initial DES model (LLM)"

    def _show():
        st.caption(f"Saved to `{ss['init_model_path']}`")
        st.code(ss["clean_initial_model"], language="python")

    if not active and "clean_initial_model" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Calls `ModelBuilder.build(...)` — a builder + inspector LLM pair "
        "that produces the initial DES model from the event-log summary "
        "and the blueprint. Takes ~30–90 seconds."
    )
    if st.button("Build initial model", type="primary"):
        cfg = ss["config"]
        bp = _read_blueprints()
        ss["blueprints"] = bp
        with st.spinner("Running builder + inspector LLM chain..."):
            builder = ModelBuilder(_client())
            model_code = builder.build(
                blueprint_code=bp["util"],
                stations_table_md=ss["stations_md"],
                sequence_text=ss["sequence_text"],
                buffers=cfg["buffers"],
                defects=cfg["defects"],
                manual_note=cfg["manual_note"],
                sim_time=cfg["sim_time"],
                warmup_seconds=cfg["warmup_seconds"],
                replications=cfg["replications"],
            )
        clean = remove_code_wrappers(model_code)
        _ensure_results_dir()
        save_model(clean, str(RESULTS_DIR), "initial_model.py")
        ss["clean_initial_model"] = clean
        ss["init_model_path"] = str(RESULTS_DIR / "initial_model.py")
        _advance()


# ---------------------------------------------------------------------------
# Stage 3: Inspect (iterative repair) + retrieve initial KPIs
# ---------------------------------------------------------------------------


def _stage_inspect(active: bool) -> None:
    ss = st.session_state
    title = "4. Inspect and evaluate the initial model"

    def _show():
        st.caption(f"Working file: `{ss['init_model_path']}`")
        st.markdown("**Initial-model KPIs**")
        st.code("\n".join(ss["kpi_results"][0]), language="text")

    if not active and "kpi_results" not in ss:
        return
    if not active and not ss.get("kpi_results"):
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Runs `inspect_code(...)` — up to 6 execution attempts, each "
        "repaired by the inspector LLM if the previous attempt fails. "
        "On success, the model is executed once more to extract KPIs."
    )
    if st.button("Inspect + run initial model", type="primary"):
        with st.spinner("Inspecting and executing..."):
            inspected = inspect_code(
                ss["clean_initial_model"], _client(), "initial_model.py"
            )
            if inspected is None:
                st.error("Inspector reached max attempts without a runnable model.")
                return
            clean_inspected = remove_code_wrappers(inspected)
            ss["clean_inspected_initial_model"] = clean_inspected
            kpi_original = retrieve_KPIs(clean_inspected, "Original model")
        ss["kpi_results"] = [kpi_original]
        _advance()


# ---------------------------------------------------------------------------
# Stage 4: Optional manual edit
# ---------------------------------------------------------------------------


def _stage_manual_edit(active: bool) -> None:
    ss = st.session_state
    title = "5. Optional: manually edit the initial model"

    def _show(readonly: bool = True):
        if ss.get("manual_edited"):
            st.success("Model was manually edited.")
        else:
            st.info("No manual edit applied.")

    if not active and "manual_edit_done" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Mirrors main.py's `Do you want to manually edit...` prompt. "
        "You can tweak the model here before the visualization and "
        "optimization stages run against it."
    )
    edit = st.radio("Edit model?", ["Skip", "Edit inline"], horizontal=True)
    if edit == "Edit inline":
        new_code = st.text_area(
            "Model code",
            value=ss["clean_inspected_initial_model"],
            height=400,
            key="manual_edit_text",
        )
        if st.button("Save edit and continue", type="primary"):
            _ensure_results_dir()
            save_model(new_code, str(RESULTS_DIR), "initial_model.py")
            ss["clean_inspected_initial_model"] = new_code
            ss["manual_edited"] = True
            ss["manual_edit_done"] = True
            _advance()
    else:
        if st.button("Continue without editing", type="primary"):
            ss["manual_edited"] = False
            ss["manual_edit_done"] = True
            _advance()


# ---------------------------------------------------------------------------
# Stage 5: Visualize (mermaid)
# ---------------------------------------------------------------------------


def _stage_visualize(active: bool) -> None:
    ss = st.session_state
    title = "6. Visualize the model (mermaid)"

    def _show():
        st.markdown("**Rendered flow chart**")
        _render_mermaid_inline(ss["mermaid_code"], height=600)
        st.download_button(
            "Download model_visualization.mmd",
            ss["mermaid_code"],
            file_name="model_visualization.mmd",
            mime="text/plain",
            key="dl_mmd_model",
        )
        with st.expander("Mermaid source (cleaned)"):
            st.code(ss["mermaid_code"], language="text")
        raw = ss.get("mermaid_code_raw")
        if raw and raw.strip() != ss["mermaid_code"].strip():
            with st.expander("Raw LLM output (before cleanup)"):
                st.code(raw, language="text")
        if st.button("Regenerate flow chart", key="regen_flow"):
            ss.pop("mermaid_code", None)
            ss.pop("mermaid_code_raw", None)
            ss["stage"] = 5  # index of visualize stage
            st.rerun()

    if not active and "mermaid_code" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Runs `Modelvisualizer.visualize_agent(...)` — asks the LLM to "
        "produce a mermaid flow chart of the model's material flow."
    )
    if st.button("Generate flow chart", type="primary"):
        with st.spinner("Asking the visualizer LLM..."):
            visualizer = Modelvisualizer(_client())
            mermaid_code = visualizer.visualize_agent(
                ss["clean_inspected_initial_model"]
            )
        cleaned = _clean_mermaid_source(mermaid_code)
        _ensure_results_dir()
        # Save the cleaned form (fence-free) so downstream tools like
        # mermaid-cli see valid input.
        (RESULTS_DIR / "model_visualization.mmd").write_text(
            cleaned, encoding="utf-8"
        )
        ss["mermaid_code"] = cleaned
        ss["mermaid_code_raw"] = mermaid_code
        _advance()


# ---------------------------------------------------------------------------
# Stage 6: Optimisation choice
# ---------------------------------------------------------------------------


def _stage_opt_choice(active: bool) -> None:
    ss = st.session_state
    title = "7. Choose optimisation path"

    def _show():
        st.write(f"Selected: **{ss.get('opt_choice_label', '(none)')}**")

    if not active and "opt_choice" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption("Same three options as `main.py`.")
    choice_labels = {
        "1": "1 — Bottleneck (SCORE) analysis",
        "2": "2 — MOO on buffers / process time / availability / MTTR",
        "3": "3 — Skip optimisation (continue to manual adaptation)",
    }
    picked = st.radio(
        "Choice",
        options=list(choice_labels.keys()),
        format_func=lambda k: choice_labels[k],
        index=1,
    )
    if picked == "1":
        st.warning(
            "The bottleneck path executes the generated SCORE code as a "
            "subprocess and — in the CLI framework — pauses for manual "
            "review with `input(...)`. In this UI those pauses are "
            "skipped; the run may still take several minutes and can "
            "fail if the generated code has issues."
        )
    if st.button("Confirm", type="primary"):
        ss["opt_choice"] = picked
        ss["opt_choice_label"] = choice_labels[picked]
        _advance()


# ---------------------------------------------------------------------------
# Stage 7: MOO / SCORE / Skip
# ---------------------------------------------------------------------------


_MOO_OBJ_SCENARIOS = {
    "1": (["wip", "throughput"], ["min", "max"], "WIP (min) vs Throughput (max)"),
    "2": (["wip", "energy consumption per part"], ["min", "min"], "WIP (min) vs Specific Energy (min)"),
    "3": (["throughput", "energy consumption per part"], ["max", "min"], "Throughput (max) vs Specific Energy (min)"),
}

_MOO_DECISION_SCENARIOS = {
    "1": ("All buffer capacities. Discrete values on the range ", "Buffer configuration"),
    "2": ("All machine process times", "Process time"),
    "3": ("All machine availabilities ", "Availability"),
    "4": ("All machine MTTR", "MTTR"),
}

# Presets from "Settings for demo tests.docx". Each preset maps 1:1 to one
# of the pre-computed moo_simulation_results{N}.csv files under data/.
_MOO_PRESETS = {
    "1": {
        "label": "Demo 1 — Baseline (WIP vs TH, buffer 1–10)",
        "obj": "1", "dec": "1",
        "rng": "1-10", "constraint": "", "pct": "10",
        "pop": 50, "gens": 50, "sim": 691_200, "warmup": 86_400,
        "csv": "moo_simulation_results1.csv",
        "user_prompt": "I want data points around the knee-point of the pareto front",
    },
    "2": {
        "label": "Demo 2 — Process time (WIP vs SEC, ±10%)",
        "obj": "2", "dec": "2",
        "rng": "1-10", "constraint": "", "pct": "10",
        "pop": 10, "gens": 30, "sim": 691_200, "warmup": 86_400,
        "csv": "moo_simulation_results2.csv",
        "user_prompt": "I want data points around the knee-point of the pareto front",
    },
    "3": {
        "label": "Demo 3 — Constrained (TH vs SEC, buffer 1–10, total ≤ 40)",
        "obj": "3", "dec": "1",
        "rng": "1-10", "constraint": "40", "pct": "10",
        "pop": 50, "gens": 50, "sim": 691_200, "warmup": 86_400,
        "csv": "moo_simulation_results3.csv",
        "user_prompt": "I want data points around the knee-point of the pareto front",
    },
}


def _match_demo(obj_key: str, dec_key: str, constraint_str: str) -> Optional[str]:
    """Return the demo id ("1"/"2"/"3") that matches the current MOO
    settings, or None. Matching keys are the two radio choices plus
    the presence/absence of a total-buffer constraint."""
    has_constraint = bool((constraint_str or "").strip())
    if obj_key == "1" and dec_key == "1" and not has_constraint:
        return "1"
    if obj_key == "2" and dec_key == "2":
        return "2"
    if obj_key == "3" and dec_key == "1" and has_constraint:
        return "3"
    return None


def _find_pareto_from_csv(csv_path: Path, selected_objectives, directions) -> pd.DataFrame:
    from paretoset import paretoset

    df = pd.read_csv(csv_path)
    mask = paretoset(
        df[[selected_objectives[0], selected_objectives[1]]],
        sense=[directions[0], directions[1]],
    )
    return df[mask]


def _moo_scatter(csv_path: Path, pareto_df: pd.DataFrame, suggestions_df: pd.DataFrame, selected_objectives: list[str]):
    df = pd.read_csv(csv_path)
    df["Type"] = "Standard Solution"
    p = pareto_df.copy()
    p["Type"] = "Pareto Optimal"
    s = suggestions_df.copy()
    s["Type"] = "LLM chosen points"
    combined = pd.concat([df, p, s], ignore_index=True, sort=False)
    hover_cols = [
        c for c in combined.columns
        if c not in selected_objectives and c != "Type"
    ]
    fig = px.scatter(
        combined,
        x=selected_objectives[0],
        y=selected_objectives[1],
        color="Type",
        hover_data=hover_cols,
        title="MOO Simulation Results",
        labels={c: c.replace("_", " ").title() for c in selected_objectives},
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="DarkGrey")))
    return fig


def _stage_moo_inputs(active: bool) -> None:
    ss = st.session_state
    title = "8a. MOO — configure the run"
    if ss.get("opt_choice") != "2":
        return

    def _show():
        st.write(f"**Objectives:** {ss['moo_obj_label']}")
        st.write(f"**Decision variable:** {ss['moo_dec_label']} — `{ss['moo_decision_vars']}`")
        st.write(
            f"**Population / generations:** {ss['moo_pop']} / {ss['moo_gens']}; "
            f"SIM_TIME = {ss['moo_sim_time']} s, warmup = {ss['moo_warmup']} s"
        )
        st.write(
            f"**Matched preset:** {ss.get('moo_demo_label', '(none)')} "
            f"→ `{ss['moo_demo_csv']}`"
        )

    if not active and "moo_selected_objectives" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Because this is the demo variant of the framework, actual MOO "
        "execution is skipped. Your answers are matched against the "
        "scenarios in **Settings for demo tests.docx**, and the "
        "corresponding `data/moo_simulation_results{N}.csv` is loaded."
    )

    preset_options = ["Custom"] + [f"{k}: {v['label']}" for k, v in _MOO_PRESETS.items()]
    picked = st.selectbox(
        "Load one of the three preset demo scenarios (fills the form below)",
        options=preset_options,
        index=1,
        help="Selecting a preset only fills the defaults; you can still edit fields before submitting.",
    )
    if picked.startswith(("1:", "2:", "3:")):
        d = _MOO_PRESETS[picked[0]]
    else:
        d = {"obj": "1", "dec": "1", "rng": "1-10", "constraint": "",
             "pct": "10", "pop": 50, "gens": 50, "sim": 691_200, "warmup": 86_400}

    with st.form("moo_form"):
        c1, c2 = st.columns(2)
        with c1:
            obj_key = st.radio(
                "Objectives",
                options=list(_MOO_OBJ_SCENARIOS.keys()),
                format_func=lambda k: _MOO_OBJ_SCENARIOS[k][2],
                index=list(_MOO_OBJ_SCENARIOS.keys()).index(d["obj"]),
            )
            dec_key = st.radio(
                "Decision variables",
                options=list(_MOO_DECISION_SCENARIOS.keys()),
                format_func=lambda k: _MOO_DECISION_SCENARIOS[k][1],
                index=list(_MOO_DECISION_SCENARIOS.keys()).index(d["dec"]),
            )
        with c2:
            pop = st.number_input("Population size", min_value=2, value=int(d["pop"]), step=1)
            gens = st.number_input("Generations", min_value=1, value=int(d["gens"]), step=1)
            sim_time = st.number_input("SIM_TIME (s)", min_value=1000, value=int(d["sim"]), step=3600)
            warmup = st.number_input("WARMUP_SECONDS", min_value=0, value=int(d["warmup"]), step=3600)

        st.markdown("**Decision-variable parameters**")
        if dec_key == "1":
            rng = st.text_input("Buffer range (e.g. 1-10)", value=d.get("rng", "1-10"))
            constraint = st.text_input(
                "OPTIONAL total-capacity constraint",
                value=d.get("constraint", ""),
            )
            pct = "10"
        else:
            rng = ""
            pct = st.text_input("Percentage", value=d.get("pct", "10"))
            constraint = st.text_input(
                "OPTIONAL constraint (max # of machines affected)",
                value=d.get("constraint", ""),
            )
        submitted = st.form_submit_button("Save MOO configuration", type="primary")

    # Live preview of which demo the current picks match — outside the form.
    preview_match = _match_demo(d["obj"], d["dec"], d.get("constraint", ""))
    if preview_match:
        st.success(f"Current picks match **{_MOO_PRESETS[preview_match]['label']}** — will load `data/{_MOO_PRESETS[preview_match]['csv']}`.")
    else:
        st.info(
            "Current picks don't match any preset. Only Demos 1, 2, and 3 "
            "have pre-computed results; unmatched settings will be rejected."
        )

    if submitted:
        matched = _match_demo(obj_key, dec_key, constraint)
        if matched is None:
            st.error(
                "This configuration doesn't match any of the pre-computed "
                "demo scenarios (Demo 1: WIP/TH + buffer 1–10 no constraint; "
                "Demo 2: WIP/SEC + process time; Demo 3: TH/SEC + buffer + "
                "total-capacity constraint). Pick a preset above or adjust "
                "your answers."
            )
            return

        sel, dirs, obj_label = _MOO_OBJ_SCENARIOS[obj_key]
        base_desc, dec_label = _MOO_DECISION_SCENARIOS[dec_key]
        if dec_key == "1":
            dv = base_desc + rng + (f" The total buffer capacity of all buffers is constrained to: {constraint}" if constraint.strip() else "")
        else:
            direction_word = "increase" if dec_key == "3" else "decrease"
            noun = {
                "2": "process times",
                "3": "availability",
                "4": "MTTR",
            }[dec_key]
            dv = base_desc + f"can {direction_word} with {pct} percent or stay at the same level"
            if constraint.strip():
                dv += f" At most {constraint} machines can have {direction_word}d {noun}."

        preset = _MOO_PRESETS[matched]
        csv_path = DATA_DIR / preset["csv"]
        ss["moo_selected_objectives"] = sel
        ss["moo_directions"] = dirs
        ss["moo_obj_label"] = obj_label
        ss["moo_choice"] = dec_key
        ss["moo_decision_vars"] = dv
        ss["moo_dec_label"] = dec_label
        ss["moo_pop"] = int(pop)
        ss["moo_gens"] = int(gens)
        ss["moo_sim_time"] = int(sim_time)
        ss["moo_warmup"] = int(warmup)
        ss["moo_demo_nr"] = matched
        ss["moo_demo_csv"] = str(csv_path)
        ss["moo_demo_label"] = preset["label"]
        _advance()


def _stage_moo_generate(active: bool) -> None:
    ss = st.session_state
    if ss.get("opt_choice") != "2":
        return
    title = "8b. MOO — generate code, combine, find Pareto front"

    def _show():
        st.markdown("**Pareto solutions**")
        st.dataframe(ss["pareto_df"], use_container_width=True, hide_index=True)
        st.markdown("**MOO structure**")
        _render_mermaid_inline(ss["moo_uml"], height=500)
        with st.expander("Generated MOO code"):
            st.code(ss["moo_code"], language="python")
        with st.expander("Combined simulation + MOO code"):
            st.code(ss["combined_code"], language="python")

    if not active and "pareto_df" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Calls `Blueprintoptimizer._generate_code`, `_combiner`, "
        "`_generate_MOO_UML`, then loads the pre-computed CSV and "
        "extracts the Pareto front (demo mode; the generated code is "
        "not executed)."
    )
    if st.button("Generate MOO artefacts", type="primary"):
        cfg = ss["config"]
        bp = ss.get("blueprints") or _read_blueprints()
        ss["blueprints"] = bp
        moo_blueprint = bp["moo_buffer"] if ss["moo_choice"] == "1" else bp["moo_availability"]

        opt = Blueprintoptimizer(_client())
        obj_str = (
            f" {ss['moo_selected_objectives'][0]} ({ss['moo_directions'][0]}) "
            f"and {ss['moo_selected_objectives'][1]} ({ss['moo_directions'][1]})"
        )
        with st.spinner("Generating MOO code..."):
            moo_raw = opt._generate_code(
                ss["clean_inspected_initial_model"],
                moo_blueprint,
                obj_str,
                ss["moo_decision_vars"],
                ss["moo_pop"],
                ss["moo_gens"],
            )
            moo_code = remove_code_wrappers(moo_raw)
            _ensure_results_dir()
            save_model(moo_code, str(RESULTS_DIR), "MOO_initial_code.py")

        with st.spinner("Combining MOO + simulation code..."):
            combined_raw = opt._combiner(
                ss["clean_inspected_initial_model"],
                moo_code,
                ss["moo_selected_objectives"],
                ss["moo_sim_time"],
                ss["moo_warmup"],
            )
            combined_code = remove_code_wrappers(combined_raw)
            save_model(combined_code, str(RESULTS_DIR), "initial_combined_code.py")

        with st.spinner("Generating MOO UML diagram..."):
            viz = Modelvisualizer(_client())
            moo_uml_raw = viz._generate_MOO_UML(
                combined_code, obj_str, ss["moo_decision_vars"]
            )
            moo_uml = _clean_mermaid_source(moo_uml_raw)
            (RESULTS_DIR / "MOO_visualization.mmd").write_text(moo_uml, encoding="utf-8")

        csv_path = Path(ss["moo_demo_csv"])
        if not csv_path.exists():
            st.error(f"Demo CSV missing: {csv_path}")
            return
        pareto_df = _find_pareto_from_csv(
            csv_path, ss["moo_selected_objectives"], ss["moo_directions"]
        )
        pareto_csv = RESULTS_DIR / "moo_pareto_solutions.csv"
        pareto_df.to_csv(pareto_csv, index=False)
        with open(pareto_csv) as fh:
            pareto_str = fh.read()

        ss["moo_code"] = moo_code
        ss["combined_code"] = combined_code
        ss["moo_uml"] = moo_uml
        ss["pareto_df"] = pareto_df
        ss["pareto_str"] = pareto_str
        _advance()


def _stage_moo_suggestions(active: bool) -> None:
    ss = st.session_state
    if ss.get("opt_choice") != "2":
        return
    title = "8c. MOO — priorities, LLM suggestions, and explanation"

    def _show():
        st.write(f"**Your priorities:** {ss['moo_user_input']}")
        st.markdown("**Suggestions (JSON)**")
        st.json(ss["moo_suggestions"])
        st.markdown("**Explanation**")
        st.markdown(ss["moo_explanation"])
        st.markdown("**Scatter plot**")
        st.plotly_chart(ss["moo_scatter_fig"], use_container_width=True)

    if not active and "moo_suggestions" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Same prompt as `main.py`: describe your production-manager "
        "priorities. The LLM picks three Pareto points aligned with "
        "them and explains why."
    )
    with st.form("priorities_form"):
        user_input = st.text_area(
            "Your priorities",
            value="I want data points around the knee-point of the pareto front",
            height=100,
        )
        submitted = st.form_submit_button("Get suggestions", type="primary")
    if submitted:
        opt = Blueprintoptimizer(_client())
        with st.spinner("LLM is choosing Pareto points..."):
            suggestions = opt._suggest_improvements(
                ss["clean_inspected_initial_model"],
                user_input,
                ss["pareto_str"],
                ss["moo_selected_objectives"],
            )
        # Save suggestions as CSV, mirroring the framework.
        try:
            root_key = list(suggestions.keys())[0]
            sugg_df = pd.DataFrame(suggestions[root_key])
        except Exception:
            sugg_df = pd.DataFrame()
        sugg_df.to_csv(RESULTS_DIR / "suggested_improvements.csv", index=False)
        # For visualising, place a copy at the CWD too — `visualize_MOO_results`
        # reads from CWD; we build our own scatter so this is not required,
        # but keeping the artefact for downstream compatibility.

        with st.spinner("Explaining the suggestions..."):
            explanation = opt._explain_suggestions(
                suggestions, ss["clean_inspected_initial_model"]
            )

        fig = _moo_scatter(
            Path(ss["moo_demo_csv"]),
            ss["pareto_df"],
            sugg_df,
            ss["moo_selected_objectives"],
        )

        ss["moo_user_input"] = user_input
        ss["moo_suggestions"] = suggestions
        ss["moo_suggestions_df"] = sugg_df
        ss["moo_explanation"] = explanation
        ss["moo_scatter_fig"] = fig

        # Extract step_list for adaptation stage.
        if isinstance(suggestions, dict):
            step_list = suggestions.get("instructions", [])
        elif isinstance(suggestions, list):
            step_list = suggestions
        else:
            step_list = []
        ss["step_list"] = step_list
        _advance()


# ---------------------------------------------------------------------------
# SCORE (bottleneck) — condensed
# ---------------------------------------------------------------------------


def _stage_score(active: bool) -> None:
    ss = st.session_state
    if ss.get("opt_choice") != "1":
        return
    title = "8. Bottleneck (SCORE) analysis"

    def _show():
        st.markdown("**Frequency analysis**")
        st.json(ss.get("score_frequencies", {}))
        st.markdown("**Suggestions**")
        st.json(ss.get("score_suggestions", {}))
        st.markdown("**Explanation**")
        st.markdown(ss.get("score_explanation", ""))

    if not active and "score_suggestions" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.warning(
        "The SCORE run executes the generated code as a subprocess. In "
        "the CLI it pauses for manual review inside the repair loop; the "
        "UI skips those pauses and runs whatever the LLM produced. If it "
        "fails, click **Skip SCORE** to fall back to manual adaptation."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        run_it = st.button("Run SCORE analysis", type="primary")
    with c2:
        skip = st.button("Skip SCORE", type="secondary")

    if skip:
        ss["step_list"] = []
        ss["score_skipped"] = True
        _advance()
        return

    if not run_it:
        return

    bp = ss.get("blueprints") or _read_blueprints()
    ss["blueprints"] = bp
    optimizer = BottleneckOptimizer(_client())

    try:
        with st.spinner("Generating SCORE code..."):
            raw = optimizer._generate_code(
                ss["clean_inspected_initial_model"], bp["score"]
            )
            score_code = remove_code_wrappers(raw)
            _ensure_results_dir()
            save_model(score_code, str(RESULTS_DIR), "SCORE_initial_code.py")

        with st.spinner("Combining SCORE + simulation code..."):
            combined_raw = optimizer._combiner(
                ss["clean_inspected_initial_model"], score_code, 86_400, 86_400 * 2
            )
            score_combined = remove_code_wrappers(combined_raw)
            save_model(score_combined, str(RESULTS_DIR), "SCORE_initial_combined_code.py")

        with st.spinner("Executing SCORE analysis (may take minutes)..."):
            from helpers.runner import run_python_code

            _ = run_python_code(score_combined)

        with st.spinner("Extracting Pareto ranks and frequencies..."):
            _ = optimizer._find_pareto_front(5)
            frequencies = optimizer.get_flag_frequencies()

        with st.spinner("Asking LLM for bottleneck suggestions..."):
            suggestions = optimizer._suggest_improvements(score_combined, frequencies)
            explanation = optimizer._explain_suggestions(
                suggestions, ss["clean_inspected_initial_model"]
            )
    except Exception as exc:
        st.error(f"SCORE run failed: {exc}")
        return

    ss["score_code"] = score_code
    ss["score_combined"] = score_combined
    ss["score_frequencies"] = frequencies
    ss["score_suggestions"] = suggestions
    ss["score_explanation"] = explanation
    if isinstance(suggestions, dict):
        step_list = suggestions.get("instructions", [])
    elif isinstance(suggestions, list):
        step_list = suggestions
    else:
        step_list = []
    ss["step_list"] = step_list
    _advance()


# ---------------------------------------------------------------------------
# Stage 9: Human adaptation input
# ---------------------------------------------------------------------------


def _stage_human_input(active: bool) -> None:
    ss = st.session_state
    title = "9. Optional: add a manual change to the adaptation list"

    def _show():
        v = ss.get("human_input", "")
        if v:
            st.success(f"Added manual change: {v}")
        else:
            st.info("No manual change added.")

    if not active and "human_input_done" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    step_list = ss.get("step_list", [])
    st.write(f"Current suggestion list has **{len(step_list)}** step(s).")
    if step_list:
        st.json(step_list)
    with st.form("human_input_form"):
        v = st.text_input("Optional change (leave blank to skip)", value="")
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        if v.strip():
            ss.setdefault("step_list", []).append(v.strip())
        ss["human_input"] = v.strip()
        ss["human_input_done"] = True
        _advance()


# ---------------------------------------------------------------------------
# Stage 10: CPD safety check
# ---------------------------------------------------------------------------


def _stage_cpd(active: bool) -> None:
    ss = st.session_state
    title = "10. CPD safety check"

    def _show():
        st.markdown(ss.get("cpd_result", ""))

    if not active and "cpd_result" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    st.caption(
        "Runs the CPD agent on the suggestion list to flag any change "
        "that violates the critical-process rules from the config."
    )
    if st.button("Run CPD check", type="primary"):
        cfg = ss["config"]
        expert = CPD(_client())
        with st.spinner("Checking suggestions..."):
            result = expert.evaluatecpd(ss.get("step_list", []), cfg["cpd_info"])
        ss["cpd_result"] = result
        _advance()


# ---------------------------------------------------------------------------
# Stage 11: Adaptation loop
# ---------------------------------------------------------------------------


def _stage_adapt(active: bool) -> None:
    ss = st.session_state
    title = "11. Apply adaptations and retrieve KPIs"

    def _show():
        st.markdown(f"Applied **{len(ss['kpi_results']) - 1}** adaptation(s).")
        for block in ss["kpi_results"][1:]:
            st.code("\n".join(block), language="text")

    if not active and "adapt_done" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    step_list = ss.get("step_list", [])
    if not step_list:
        st.info("No adaptation steps to apply.")
        if st.button("Continue", type="primary"):
            ss["adapt_done"] = True
            _advance()
        return

    st.write(f"Will apply **{len(step_list)}** adaptation step(s).")
    if st.button("Run adaptations", type="primary"):
        _ensure_results_dir()
        client = _client()
        progress = st.progress(0.0, text="Starting adaptations...")
        for idx, step in enumerate(step_list, start=1):
            progress.progress(
                (idx - 1) / len(step_list),
                text=f"Adaptation {idx}/{len(step_list)}: {str(step)[:80]}",
            )
            adaptor = Modeladaptor(client)
            kpi_adapted = adaptor.adapter(
                original_code=ss["clean_inspected_initial_model"],
                instruction=step,
                final_path=str(RESULTS_DIR),
                name="Adapted_model_version",
                multi_agent_setting=False,
                index_model=idx,
            )
            ss["kpi_results"].append(kpi_adapted)
        progress.progress(1.0, text="Done.")
        ss["adapt_done"] = True
        _advance()


# ---------------------------------------------------------------------------
# Stage 12: Evaluate + compare
# ---------------------------------------------------------------------------


def _stage_evaluate(active: bool) -> None:
    ss = st.session_state
    title = "12. Evaluate and compare all model versions"

    def _show():
        st.markdown("**LLM evaluation summary**")
        st.markdown(ss["final_evaluation"])
        st.markdown("**KPI comparison chart**")
        st.image(str(RESULTS_DIR / "model_comparison_kpis.png"))

    if not active and "final_evaluation" not in ss:
        return
    if not active:
        st.markdown(f"### ✅ {title}")
        _show()
        st.divider()
        return

    st.header(title)
    if st.button("Run evaluator + build chart", type="primary"):
        evaluator = Evaluater(_client())
        with st.spinner("LLM is summarising..."):
            summary = evaluator.evaluate(ss["kpi_results"])
        try:
            visualize_results(
                ss["kpi_results"],
                file_name="model_comparison_kpis.png",
                save_path=str(RESULTS_DIR),
            )
        except Exception as exc:
            st.warning(f"Could not build KPI chart: {exc}")
        ss["final_evaluation"] = summary
        _advance()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _is_moo(ss) -> bool:
    return ss.get("opt_choice") == "2"


def _is_score(ss) -> bool:
    return ss.get("opt_choice") == "1"


def _is_adapt_path(ss) -> bool:
    # Adaptation runs unless SCORE ran and we chose to exit; here we always
    # continue to adaptation whichever path was chosen.
    return True


STAGES: list[dict] = [
    {"fn": _stage_config, "applicable": lambda ss: True},
    {"fn": _stage_process_mining, "applicable": lambda ss: True},
    {"fn": _stage_build, "applicable": lambda ss: True},
    {"fn": _stage_inspect, "applicable": lambda ss: True},
    {"fn": _stage_manual_edit, "applicable": lambda ss: True},
    {"fn": _stage_visualize, "applicable": lambda ss: True},
    {"fn": _stage_opt_choice, "applicable": lambda ss: True},
    {"fn": _stage_moo_inputs, "applicable": _is_moo},
    {"fn": _stage_moo_generate, "applicable": _is_moo},
    {"fn": _stage_moo_suggestions, "applicable": _is_moo},
    {"fn": _stage_score, "applicable": _is_score},
    {"fn": _stage_human_input, "applicable": _is_adapt_path},
    {"fn": _stage_cpd, "applicable": _is_adapt_path},
    {"fn": _stage_adapt, "applicable": _is_adapt_path},
    {"fn": _stage_evaluate, "applicable": _is_adapt_path},
]


def _skip_inapplicable() -> int:
    """Advance the stage counter past any stage whose applicability
    predicate is False in the current session state, and return the new
    current stage index. Called before rendering."""
    ss = st.session_state
    cur = ss.get("stage", 0)
    while cur < len(STAGES) and not STAGES[cur]["applicable"](ss):
        cur += 1
    ss["stage"] = cur
    return cur


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


_SIDEBAR_CSS = """
<style>
  section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.35rem; }
  .fm-brand {
    text-align: center; padding: 10px 0 6px 0; border-radius: 10px;
    background: linear-gradient(135deg, #0f2c4d 0%, #1a4d8f 100%);
    color: #fff; margin-bottom: 10px;
  }
  .fm-brand .fm-icon { font-size: 30px; line-height: 1; }
  .fm-brand .fm-title { font-weight: 700; font-size: 15px; letter-spacing: 0.4px; }
  .fm-brand .fm-sub { font-size: 11px; opacity: 0.75; margin-top: 2px; }
  .fm-section {
    font-size: 12px; font-weight: 700; letter-spacing: 0.6px;
    text-transform: uppercase; opacity: 0.7; margin: 12px 0 4px 0;
  }
  .fm-chip {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; margin-right: 4px;
  }
  .fm-chip.ok { background: #e6f4ea; color: #1e6c34; }
  .fm-chip.wait { background: #fff4e5; color: #a76800; }
  .fm-chip.off { background: #eee; color: #555; }
  .fm-card {
    border: 1px solid rgba(120,120,120,0.15);
    border-radius: 8px; padding: 8px 10px; margin-bottom: 8px;
  }
  .fm-kv { display: flex; justify-content: space-between; font-size: 12px; margin: 2px 0; }
  .fm-kv .k { opacity: 0.7; }
  .fm-kv .v { font-weight: 600; }
</style>
"""


def _sidebar_brand() -> None:
    st.sidebar.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="fm-brand">
          <div class="fm-icon">🏭</div>
          <div class="fm-title">LLM-DES-MOO</div>
          <div class="fm-sub">Automated DES · MOO · Bottleneck</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_run_status() -> None:
    ss = st.session_state
    cur = ss.get("stage", 0)
    total = len(STAGES)
    completed = cur >= total

    if not ss.get("authed"):
        status_chip = '<span class="fm-chip off">🔒 Locked</span>'
    elif completed:
        status_chip = '<span class="fm-chip ok">✔ Complete</span>'
    elif cur == 0:
        status_chip = '<span class="fm-chip wait">⏸ Idle</span>'
    else:
        status_chip = '<span class="fm-chip ok">▶ Running</span>'

    st.sidebar.markdown('<div class="fm-section">Run status</div>', unsafe_allow_html=True)
    st.sidebar.markdown(status_chip, unsafe_allow_html=True)
    st.sidebar.progress(min(cur / max(total, 1), 1.0), text=f"Stage {cur} / {total}")

    # Compact metrics grid (industrial look)
    kpi_count = max(len(ss.get("kpi_results", [])) - 1, 0)
    steps = len(ss.get("step_list", []) or [])
    opt = ss.get("opt_choice_label", "—")
    st.sidebar.markdown(
        f"""
        <div class="fm-card">
          <div class="fm-kv"><span class="k">Optimisation</span><span class="v">{opt}</span></div>
          <div class="fm-kv"><span class="k">Adaptations</span><span class="v">{kpi_count} / {steps or '—'}</span></div>
          <div class="fm-kv"><span class="k">Sim time</span><span class="v">{ss.get('config', {}).get('sim_time', '—')} s</span></div>
          <div class="fm-kv"><span class="k">Warmup</span><span class="v">{ss.get('config', {}).get('warmup_seconds', '—')} s</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_controls() -> None:
    st.sidebar.markdown('<div class="fm-section">Controls</div>', unsafe_allow_html=True)
    if st.sidebar.button("↩ Reset run", use_container_width=True, key="sb_reset"):
        _reset()
    if st.sidebar.button("🔒 Log out", use_container_width=True, key="sb_logout"):
        for k in ("api_key", "authed"):
            st.session_state.pop(k, None)
        st.rerun()


def _sidebar_footer() -> None:
    st.sidebar.markdown('<div class="fm-section">Institution</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="fm-card" style="line-height: 1.4;">
          <div style="font-weight: 600; font-size: 12px;">Uppsala University</div>
          <div style="font-size: 11px; opacity: 0.75;">Dept. of Civil &amp; Industrial Engineering</div>
          <div style="font-size: 11px; opacity: 0.75; margin-top: 4px;">In collaboration with Scania CV AB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(current_page: str = "wizard") -> None:
    _sidebar_brand()
    if current_page == "wizard":
        _sidebar_run_status()
        if st.session_state.get("authed"):
            _sidebar_controls()
    _sidebar_footer()


# ---------------------------------------------------------------------------
# Login (rendered at the top of the wizard page)
# ---------------------------------------------------------------------------


def _render_login_top() -> None:
    st.markdown("### 🔒 Unlock the framework")
    st.caption(
        "Enter an OpenAI API key. It is validated once against OpenAI "
        "and kept only in the current browser session."
    )
    with st.form("login_form_top"):
        c1, c2 = st.columns([3, 1])
        with c1:
            if PASSCODE:
                passcode_in = st.text_input("Passphrase", type="password")
            else:
                passcode_in = None
            api_key = st.text_input(
                "OpenAI API key", type="password", placeholder="sk-...",
            )
        with c2:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Unlock", type="primary", use_container_width=True)

    if not submitted:
        return
    if PASSCODE and (passcode_in or "").strip() != PASSCODE:
        st.error("Wrong passphrase.")
        return
    if not api_key.strip():
        st.error("API key is required.")
        return
    with st.spinner("Validating key with OpenAI..."):
        ok, msg = _validate_openai_key(api_key.strip())
    if not ok:
        st.error(msg)
        return
    st.session_state["api_key"] = api_key.strip()
    st.session_state["authed"] = True
    st.rerun()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


_AUTHORS = [
    ("Arvid Krusten", "Uppsala University",
     "Conceptualization · Methodology · Software · Investigation"),
    ("Viktor Norberg", "Uppsala University",
     "Conceptualization · Methodology · Software · Investigation"),
    ("Thomas Schmitt", "Scania CV AB &nbsp;·&nbsp; Uppsala University",
     "Conceptualization · Methodology · Software"),
    ("Pegah Rahimian  ✉", "Uppsala University",
     "Corresponding · Supervision · pegah.rahimian@angstrom.uu.se"),
    ("Matías Urenda Moris", "Uppsala University",
     "Supervision"),
    ("Kaveh Amouzgar", "Uppsala University",
     "Resources · Supervision"),
]


def welcome_page() -> None:
    _sidebar("welcome")

    # Hero
    st.markdown(
        """
        <div style="padding: 18px 22px; border-radius: 12px;
                    background: linear-gradient(135deg, #0f2c4d 0%, #1a4d8f 100%);
                    color: #fff; margin-bottom: 16px;">
          <div style="font-size: 12px; letter-spacing: 1px; opacity: 0.8;">
            🏭 &nbsp; LLM-DES-MOO FRAMEWORK
          </div>
          <div style="font-size: 26px; font-weight: 700; margin-top: 4px;">
            Automated discrete-event simulation, multi-objective optimization,
            and bottleneck analysis of manufacturing production lines
          </div>
          <div style="font-size: 14px; opacity: 0.85; margin-top: 6px;">
            An LLM-agent framework — Uppsala University · Scania CV AB
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Short introduction (paraphrase of abstract)
    st.markdown(
        """
Wrapping a discrete-event simulation (DES) of a production line in a tailored
optimizer and translating the resulting Pareto front into concrete process
improvements is a sequence of expert tasks that few manufacturers can
dedicate to routine decision support.

**This framework closes that gap.** A sequence of LLM agents — each
constrained by a small executable blueprint — jointly generates:

- 🧱 **A SimPy DES model** of the production line, straight from a raw event log
- 🎯 **A tailored NSGA-II MOEA** wrapped around that DES, driven by user-chosen KPIs
- 🔎 **An automated SCORE bottleneck analysis** (previously an expert-only task in commercial software)

A dedicated inspector agent repairs syntactically or structurally broken
code, and light human-in-the-loop checkpoints let a practitioner set
objectives, hyperparameters, and Pareto-point preferences **without writing
any code**.
        """
    )

    # Result banners (industrial dashboard style)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Hypervolume", "192.29", delta="+5.8% vs FACTS")
    with c2:
        st.metric("KPI agreement", "±few %", delta="vs commercial")
    with c3:
        st.metric("Median run cost", "$0.30", delta="OpenAI API")
    with c4:
        st.metric("Human effort", "Minimal", delta="Prompts only")

    st.divider()

    # Authors
    st.markdown("### 👥 Authors")
    cols = st.columns(3)
    for i, (name, aff, role) in enumerate(_AUTHORS):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="border: 1px solid rgba(120,120,120,0.18);
                            border-radius: 10px; padding: 12px 14px;
                            margin-bottom: 10px; height: 108px;">
                  <div style="font-weight: 700; font-size: 14px;">{name}</div>
                  <div style="font-size: 12px; opacity: 0.75; margin-top: 2px;">{aff}</div>
                  <div style="font-size: 11px; opacity: 0.65; margin-top: 6px;">{role}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # CTA
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### ➡️ Get started")
            st.write(
                "Head to **🚀 Start the Run** in the sidebar to launch the "
                "interactive framework. You'll be asked for an OpenAI API "
                "key at the top of that page — it is kept only in the "
                "current browser session."
            )
        with c2:
            st.write("")
            st.write("")
            if st.button("Start the run →", type="primary", use_container_width=True):
                st.switch_page(_wizard_page_obj)  # type: ignore[arg-type]

    st.caption(
        "Master's-thesis project · Uppsala University · Department of Civil "
        "and Industrial Engineering · 2026"
    )


def wizard_page() -> None:
    _init_state()
    _sidebar("wizard")

    st.title("🧭 Interactive Framework")
    st.caption(
        "Same pipeline as `main.py`, rendered as a Streamlit wizard. Every "
        "prompt in the CLI becomes a widget here."
    )

    if not st.session_state.get("authed"):
        _render_login_top()
        return

    current = _skip_inapplicable()
    ss = st.session_state
    for idx, stage in enumerate(STAGES):
        if not stage["applicable"](ss):
            continue
        stage["fn"](active=(idx == current))

    if current >= len(STAGES):
        st.success("🎉 Pipeline complete. Use **↩ Reset run** in the sidebar to start over.")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# `st.Page` returns a page object; we hold onto the wizard object so
# `welcome_page`'s CTA can `st.switch_page` to it.
_wizard_page_obj = st.Page(wizard_page, title="Start the Run", icon="🚀", url_path="run")
_welcome_page_obj = st.Page(welcome_page, title="Home", icon="🏠", default=True, url_path="home")


def main() -> None:
    pg = st.navigation(
        {"🏭 LLM-DES-MOO Framework": [_welcome_page_obj, _wizard_page_obj]}
    )
    pg.run()


if __name__ == "__main__":
    main()
