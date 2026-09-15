"""Private Streamlit UI for the LLM-DES-MOO demo runs.

Run locally with:
    streamlit run streamlit_app.py

Access is gated by an OpenAI API key entered in the sidebar (kept in session
state only). Optionally set STREAMLIT_APP_PASSCODE in the environment for a
second-factor passphrase.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parent
DEMO_DIR = APP_ROOT / "Demo Work"
PASSCODE = os.getenv("STREAMLIT_APP_PASSCODE", "").strip()

st.set_page_config(
    page_title="LLM-DES-MOO Demo Viewer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------


@dataclass
class Demo:
    key: str
    title: str
    subtitle: str
    csv: Path
    objectives: tuple[str, str]
    obj_labels: tuple[str, str]
    obj_directions: tuple[str, str]  # "min" or "max"
    decision_var_cols: list[str]
    settings: dict = field(default_factory=dict)
    cluster_dir: Optional[Path] = None
    description: str = ""


DEMOS: dict[str, Demo] = {
    "demo1": Demo(
        key="demo1",
        title="Demo 1 — Baseline",
        subtitle="Buffer configuration MOO, WIP vs Throughput",
        csv=DEMO_DIR / "moo_simulation_results1.csv",
        objectives=("wip", "throughput"),
        obj_labels=("WIP (parts)", "Throughput (parts)"),
        obj_directions=("min", "max"),
        decision_var_cols=[
            "cap_post_loading",
            "cap_post_conveyor",
            "cap_post_washing",
            "cap_pre_press1",
            "cap_pre_press2",
            "cap_post_press12",
        ],
        settings={
            "MOO target": "Buffer configuration",
            "Objectives": "WIP (min) vs Throughput (max)",
            "Buffer range": "1–10",
            "Total buffer constraint": "None",
            "Population size": 50,
            "Generations": 50,
            "SIM_TIME (s)": 691_200,
            "WARMUP_SECONDS": 86_400,
            "User prompt": "I want data points around the knee-point of the pareto front",
        },
        cluster_dir=None,
        description=(
            "Baseline demo run: the MOO optimiser searches the space of "
            "buffer capacities (1–10) across the six buffers in the line "
            "and reports the trade-off between WIP and Throughput."
        ),
    ),
    "demo2": Demo(
        key="demo2",
        title="Demo 2 — Process-time MOO",
        subtitle="±10% process-time flags, WIP vs Specific Energy",
        csv=DEMO_DIR / "moo_simulation_results2.csv",
        objectives=("wip", "energy consumption per part"),
        obj_labels=("WIP (parts)", "Energy per part (kWh)"),
        obj_directions=("min", "min"),
        decision_var_cols=[
            "loading_robot_process_time_reduction10percent_flag",
            "conveyor_belt_process_time_reduction10percent_flag",
            "washing_machine_process_time_reduction10percent_flag",
            "hantering_cell_process_time_reduction10percent_flag",
            "presses_cell1_process_time_reduction10percent_flag",
            "presses_cell2_process_time_reduction10percent_flag",
            "quality_station_cell_process_time_reduction10percent_flag",
        ],
        settings={
            "MOO target": "Process-time reduction flags (±10%)",
            "Objectives": "WIP (min) vs Specific Energy (min)",
            "Percentage": "10%",
            "Total buffer constraint": "None",
            "Population size": 10,
            "Generations": 30,
            "SIM_TIME (s)": 691_200,
            "WARMUP_SECONDS": 86_400,
            "User prompt": "I want data points around the knee-point of the pareto front",
        },
        cluster_dir=DEMO_DIR / "cluster_results_demo2",
        description=(
            "The MOO decides for each station whether to shave 10% off its "
            "process time. Objectives are WIP and Specific Energy "
            "Consumption per part."
        ),
    ),
    "demo3": Demo(
        key="demo3",
        title="Demo 3 — Constrained buffer MOO",
        subtitle="Total buffer capacity ≤ 40, Throughput vs Specific Energy",
        csv=DEMO_DIR / "moo_simulation_results3.csv",
        objectives=("throughput", "energy consumption per part"),
        obj_labels=("Throughput (parts)", "Energy per part (kWh)"),
        obj_directions=("max", "min"),
        decision_var_cols=[
            "post_loading_buffer_capacity",
            "post_conveyor_buffer_capacity",
            "post_washing_buffer_capacity",
            "pre_press1_buffer_capacity",
            "pre_press2_buffer_capacity",
            "post_press12_buffer_capacity",
        ],
        settings={
            "MOO target": "Buffer configuration (constrained)",
            "Objectives": "Throughput (max) vs Specific Energy (min)",
            "Buffer range": "1–10",
            "Total buffer constraint": 40,
            "Population size": 50,
            "Generations": 50,
            "SIM_TIME (s)": 691_200,
            "WARMUP_SECONDS": 86_400,
            "User prompt": "I want data points around the knee-point of the pareto front",
        },
        cluster_dir=DEMO_DIR / "cluster_results_demo3",
        description=(
            "Buffer-configuration MOO with a hard constraint that the total "
            "buffer capacity across the line cannot exceed 40. Objectives "
            "are Throughput and Specific Energy per part."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Auth / gating
# ---------------------------------------------------------------------------


def _validate_openai_key(api_key: str) -> tuple[bool, str]:
    """Ping OpenAI to confirm the key works. Returns (ok, message)."""
    try:
        from openai import OpenAI, AuthenticationError, APIError
    except Exception as exc:  # pragma: no cover
        return False, f"openai package not installed: {exc}"

    try:
        client = OpenAI(api_key=api_key)
        # Cheapest possible call — just list a couple of models.
        client.models.list()
        return True, "Key accepted."
    except AuthenticationError:
        return False, "OpenAI rejected the key (authentication failed)."
    except APIError as exc:
        return False, f"OpenAI API error: {exc}"
    except Exception as exc:
        return False, f"Could not validate key: {exc}"


def _render_login() -> None:
    st.title("🔒 LLM-DES-MOO Demo Viewer")
    st.write(
        "This viewer is private. Enter an OpenAI API key to unlock the demo "
        "results. The key is kept only in this browser session."
    )

    with st.form("login_form", clear_on_submit=False):
        if PASSCODE:
            passcode_in = st.text_input(
                "Passphrase",
                type="password",
                help="Set via the STREAMLIT_APP_PASSCODE env var.",
            )
        else:
            passcode_in = None
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Only stored in-memory for this session.",
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
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _generation_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("generation", "generation_index"):
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def _pareto_mask_cached(
    key: str, x: tuple, y: tuple, x_dir: str, y_dir: str
) -> list[bool]:
    import numpy as np

    xs = np.array(x, dtype=float) * (1 if x_dir == "min" else -1)
    ys = np.array(y, dtype=float) * (1 if y_dir == "min" else -1)
    order = np.lexsort((ys, xs))
    mask = np.zeros(len(xs), dtype=bool)
    best_y = np.inf
    for idx in order:
        if ys[idx] < best_y:
            mask[idx] = True
            best_y = ys[idx]
    return mask.tolist()


def _pareto_mask(df: pd.DataFrame, demo: Demo) -> pd.Series:
    x_col, y_col = demo.objectives
    x_dir, y_dir = demo.obj_directions
    mask = _pareto_mask_cached(
        demo.key,
        tuple(df[x_col].tolist()),
        tuple(df[y_col].tolist()),
        x_dir,
        y_dir,
    )
    return pd.Series(mask, index=df.index)


def _pareto_plot(df: pd.DataFrame, demo: Demo):
    df = df.copy()
    df["Pareto"] = _pareto_mask(df, demo).map(
        {True: "Pareto front", False: "Dominated"}
    )
    x_col, y_col = demo.objectives
    x_lbl, y_lbl = demo.obj_labels
    hover_cols = [c for c in demo.decision_var_cols if c in df.columns]
    gen_col = _generation_col(df)
    if gen_col:
        hover_cols.append(gen_col)
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="Pareto",
        color_discrete_map={"Pareto front": "#e45756", "Dominated": "#4c78a8"},
        hover_data=hover_cols,
        labels={x_col: x_lbl, y_col: y_lbl},
        title=f"{x_lbl} vs {y_lbl}",
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color="white"))
    )
    fig.update_layout(legend_title_text="", height=520)
    return fig


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_settings(demo: Demo) -> None:
    st.markdown(f"### {demo.title}")
    st.caption(demo.subtitle)
    st.write(demo.description)
    settings_df = pd.DataFrame(
        [(k, str(v)) for k, v in demo.settings.items()],
        columns=["Setting", "Value"],
    )
    st.dataframe(settings_df, hide_index=True, use_container_width=True)


def _render_pareto(demo: Demo) -> Optional[pd.DataFrame]:
    if not demo.csv.exists():
        st.warning(f"Missing CSV: {demo.csv.name}")
        return None
    df = _load_csv(str(demo.csv))
    fig = _pareto_plot(df, demo)
    st.plotly_chart(fig, use_container_width=True)

    n_total = len(df)
    n_pareto = int(_pareto_mask(df, demo).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Evaluated points", n_total)
    c2.metric("Pareto front", n_pareto)
    gen_col = _generation_col(df)
    if gen_col is not None:
        c3.metric("Generations", int(df[gen_col].max()) + 1)
    return df


def _render_pareto_solutions(demo: Demo) -> None:
    if demo.cluster_dir is None:
        st.info("No archived Pareto-solutions file for this demo.")
        return
    p = demo.cluster_dir / "moo_pareto_solutions.csv"
    s = demo.cluster_dir / "suggested_improvements.csv"
    if p.exists():
        st.markdown("**Pareto solutions** (from MOO run)")
        st.dataframe(pd.read_csv(p), use_container_width=True)
    if s.exists():
        st.markdown("**Suggested improvements** (chosen by the LLM adapter)")
        st.dataframe(pd.read_csv(s), use_container_width=True)


def _render_all_points(df: Optional[pd.DataFrame], demo: Demo) -> None:
    if df is None:
        return
    st.markdown("**All evaluated designs**")
    st.dataframe(df, use_container_width=True, height=360)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=demo.csv.name,
        mime="text/csv",
    )


def _render_models(demo: Demo) -> None:
    if demo.cluster_dir is None:
        st.info("No model artefacts archived for this demo.")
        return
    files = {
        "Initial model": demo.cluster_dir / "initial_model.py",
        "Combined initial + MOO (checked)": demo.cluster_dir / "checked_initial_combined_code.py",
        "Adapted v1": demo.cluster_dir / "Adapted_model_version_1.py",
        "Adapted v2": demo.cluster_dir / "Adapted_model_version_2.py",
        "Adapted v3": demo.cluster_dir / "Adapted_model_version_3.py",
    }
    available = {k: v for k, v in files.items() if v.exists()}
    if not available:
        st.info("No generated Python files found.")
        return
    tabs = st.tabs(list(available.keys()))
    for tab, (label, path) in zip(tabs, available.items()):
        with tab:
            code = path.read_text(encoding="utf-8", errors="replace")
            st.caption(f"{path.name} — {len(code):,} chars")
            st.download_button(
                f"Download {path.name}",
                code,
                file_name=path.name,
                mime="text/x-python",
                key=f"dl-{demo.key}-{path.name}",
            )
            st.code(code, language="python")


def _render_kpi_image(demo: Demo) -> None:
    if demo.cluster_dir is None:
        st.info("No KPI comparison chart archived for this demo.")
        return
    p = demo.cluster_dir / "model_comparison_kpis.png"
    if not p.exists():
        st.info("KPI comparison PNG not found.")
        return
    st.image(str(p), caption="Original vs adapted models — KPI comparison", use_container_width=True)


_MERMAID_HEADERS = (
    "flowchart", "graph ", "sequenceDiagram", "classDiagram",
    "stateDiagram", "erDiagram", "gantt", "pie", "journey",
    "gitGraph", "mindmap",
)


def _clean_mermaid(src: str) -> str:
    import re as _re
    s = (src or "").strip()
    s = _re.sub(r"^```[a-zA-Z]*\s*\n", "", s)
    if s.endswith("```"):
        s = s[:-3].rstrip()
    for i, line in enumerate(s.splitlines()):
        if any(line.lstrip().startswith(h) for h in _MERMAID_HEADERS):
            return "\n".join(s.splitlines()[i:]).strip()
    return s


def _render_mermaid(path: Path, title: str) -> None:
    if not path.exists():
        return
    import html as _html
    src = _clean_mermaid(path.read_text(encoding="utf-8", errors="replace"))
    escaped = _html.escape(src)
    with st.expander(f"{title} — {path.name}"):
        page = f"""
        <div class="mermaid">{escaped}</div>
        <pre style="color:#b00020;white-space:pre-wrap;font-family:monospace;font-size:12px;display:none" id="mmd-err-{path.stem}"></pre>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: false, theme: 'default', securityLevel: 'loose' }});
          try {{
            await mermaid.run({{ querySelector: '.mermaid' }});
          }} catch (err) {{
            const el = document.getElementById('mmd-err-{path.stem}');
            el.style.display = 'block';
            el.textContent = 'Mermaid render failed:\\n' + (err && err.message ? err.message : err);
          }}
        </script>
        """
        components.html(page, height=600, scrolling=True)
        st.download_button(
            f"Download {path.name}",
            src,
            file_name=path.name,
            mime="text/plain",
            key=f"dl-mmd-{path.name}",
        )


def _render_diagrams(demo: Demo) -> None:
    if demo.cluster_dir is None:
        st.info("No diagrams archived for this demo.")
        return
    _render_mermaid(demo.cluster_dir / "model_visualization.mmd", "Model flow chart")
    _render_mermaid(demo.cluster_dir / "MOO_visualization.mmd", "MOO structure")


def _render_moo_html(demo: Demo) -> None:
    if demo.cluster_dir is None:
        st.info("No interactive MOO plot archived for this demo.")
        return
    p = demo.cluster_dir / "moo_results.html"
    if not p.exists():
        st.info("moo_results.html not found.")
        return
    size_mb = p.stat().st_size / (1024 * 1024)
    st.caption(f"Full interactive Pareto plot from the pipeline ({size_mb:.1f} MB).")

    with open(p, "rb") as fh:
        b = fh.read()
    st.download_button(
        "Download moo_results.html",
        b,
        file_name=f"{demo.key}_moo_results.html",
        mime="text/html",
    )
    with st.expander("Show inline (may take a moment to render)"):
        components.html(b.decode("utf-8", errors="replace"), height=700, scrolling=True)


def _render_ask_ai(demo: Demo, df: Optional[pd.DataFrame]) -> None:
    st.markdown(
        "Ask GPT-4o-mini about this demo. The Pareto solutions and demo "
        "settings are sent along as context."
    )
    with st.form(f"ask-{demo.key}"):
        q = st.text_area(
            "Your question",
            placeholder="e.g. Which buffer sizes drive the knee point?",
            height=100,
        )
        submitted = st.form_submit_button("Ask", type="primary")
    if not submitted:
        return
    if not q.strip():
        st.warning("Type a question first.")
        return

    api_key = st.session_state.get("api_key")
    if not api_key:
        st.error("Session missing API key — log out and back in.")
        return

    try:
        from openai import OpenAI
    except Exception as exc:
        st.error(f"openai import failed: {exc}")
        return

    pareto_snippet = ""
    if df is not None:
        pmask = _pareto_mask(df, demo)
        pdf = df[pmask].head(50)
        pareto_snippet = pdf.to_csv(index=False)

    context = (
        f"Demo: {demo.title}\n"
        f"Subtitle: {demo.subtitle}\n"
        f"Description: {demo.description}\n\n"
        f"Settings:\n"
        + "\n".join(f"- {k}: {v}" for k, v in demo.settings.items())
        + "\n\nPareto solutions (up to 50 rows, CSV):\n"
        + pareto_snippet
    )

    client = OpenAI(api_key=api_key)
    with st.spinner("Thinking..."):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an analyst reviewing a discrete-event "
                            "simulation MOO run. Be precise, quantitative, "
                            "and cite specific rows or columns from the "
                            "Pareto CSV when relevant."
                        ),
                    },
                    {"role": "user", "content": context},
                    {"role": "user", "content": q.strip()},
                ],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
        except Exception as exc:
            st.error(f"OpenAI call failed: {exc}")
            return
    st.markdown(answer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _render_app() -> None:
    with st.sidebar:
        st.markdown("### LLM-DES-MOO Demo")
        st.caption("Private viewer for the three demo runs.")
        selected_label = st.radio(
            "Demo",
            options=list(DEMOS.keys()),
            format_func=lambda k: DEMOS[k].title,
            index=0,
        )
        st.divider()
        st.caption("Session")
        if st.button("Log out", use_container_width=True):
            for k in ("api_key", "authed"):
                st.session_state.pop(k, None)
            st.rerun()

    demo = DEMOS[selected_label]

    top_left, top_right = st.columns([3, 1])
    with top_left:
        _render_settings(demo)
    with top_right:
        st.markdown("#### Files")
        if demo.cluster_dir:
            st.write(f"`{demo.cluster_dir.relative_to(APP_ROOT)}`")
        st.write(f"`{demo.csv.relative_to(APP_ROOT)}`")

    df: Optional[pd.DataFrame] = None
    tabs = st.tabs(
        [
            "Pareto front",
            "Pareto solutions",
            "All evaluations",
            "Generated models",
            "KPI comparison",
            "Diagrams",
            "Interactive MOO",
            "Ask AI",
        ]
    )

    with tabs[0]:
        df = _render_pareto(demo)
    with tabs[1]:
        _render_pareto_solutions(demo)
    with tabs[2]:
        _render_all_points(df, demo)
    with tabs[3]:
        _render_models(demo)
    with tabs[4]:
        _render_kpi_image(demo)
    with tabs[5]:
        _render_diagrams(demo)
    with tabs[6]:
        _render_moo_html(demo)
    with tabs[7]:
        _render_ask_ai(demo, df)


def main() -> None:
    if not st.session_state.get("authed"):
        _render_login()
        return
    _render_app()


if __name__ == "__main__":
    main()
