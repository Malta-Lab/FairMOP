"""
FairMOP Interactive Dashboard – Streamlit application.

Provides an interactive web interface for:
    - Loading and visualizing FairMOP evaluation results
    - Exploring Pareto frontiers with customizable metrics
    - Comparing multiple models/methods side by side
    - Exporting charts (SVG, HTML) and data (JSON, CSV)

Launch with:
    python -m fairmop dashboard
    # or
    streamlit run fairmop/dashboard/app.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fairmop.output.export import (
    export_pareto_csv,
    export_results_csv,
    load_results_json,
)
from fairmop.output.pareto import (
    ConfigurationPoint,
    ParetoResult,
    build_pareto_from_results,
    find_pareto_frontier,
)

# ── Pre-load path from CLI (python -m fairmop dashboard --results path.json) ─


def _preload_path() -> str | None:
    """Return the path passed via --results / -r, or None."""
    argv = sys.argv[1:]
    for flag in ("--results", "-r"):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                return argv[idx + 1]
    return None


_PRELOAD_PATH = _preload_path()

# ── Constants ────────────────────────────────────────────────────────────────

COLOR_OPTIONS = [
    ("Orange", "#FF8C00"),
    ("Red", "#FF4444"),
    ("Green", "#28A745"),
    ("Purple", "#8B5CF6"),
    ("Yellow", "#FFC107"),
    ("Light Blue", "#17A2B8"),
    ("Brown", "#8B4513"),
    ("Pink", "#FF69B4"),
    ("Dark Gray", "#6C757D"),
]

SYMBOL_OPTIONS = [
    ("Circle", "circle"),
    ("Square", "square"),
    ("Triangle Up", "triangle-up"),
    ("Diamond", "diamond"),
    ("Cross", "cross"),
    ("Star", "star"),
    ("X", "x"),
]

METRIC_DISPLAY = {
    "avg_clip_score": "CLIP Score",
    "gender_entropy": "Gender Entropy (Fairness)",
    "ethnicity_entropy": "Ethnicity Entropy",
    "age_entropy": "Age Entropy",
    "entropy_fairness": "Overall Entropy (Fairness)",
    "gender_kl": "Gender KL Divergence",
    "kl_fairness": "Overall KL Divergence",
    "fid_score": "FID Score",
    "inverse_fid": "1/FID (Quality)",
    "prdc_score": "PRDC Precision",
    "precision": "PRDC Precision",
}


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FairMOP Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def detect_metrics(results: dict) -> list[str]:
    """Detect available metrics from a results dictionary."""
    if not results.get("configurations"):
        return []
    first_config = list(results["configurations"].values())[0]
    return list(first_config.get("aggregates", {}).keys())


def build_summary_table(results: dict, selected_metrics: list[str]) -> pd.DataFrame:
    """Build a summary DataFrame from results."""
    rows = []
    for config_name, config_data in sorted(results["configurations"].items()):
        row = {"Configuration": config_name, "Images": config_data.get("images", 0)}
        for metric in selected_metrics:
            val = config_data.get("aggregates", {}).get(metric, "—")
            if isinstance(val, float) and val != float("inf"):
                row[METRIC_DISPLAY.get(metric, metric)] = f"{val:.4f}"
            elif val == float("inf"):
                row[METRIC_DISPLAY.get(metric, metric)] = "∞"
            else:
                row[METRIC_DISPLAY.get(metric, metric)] = val
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main App ─────────────────────────────────────────────────────────────────

st.title("⚖️ FairMOP – Fairness-Utility Trade-off Dashboard")
st.markdown(
    "Analyze and visualize Pareto frontiers for text-to-image model benchmarking."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("📂 Data Input")

uploaded_main = st.sidebar.file_uploader(
    "Main Results JSON", type=["json"], key="main_json"
)

num_comparisons = st.sidebar.number_input(
    "Number of comparison files", min_value=0, max_value=5, value=0
)

comparison_uploads = []
for i in range(num_comparisons):
    col_file, col_name = st.sidebar.columns([2, 1])
    with col_file:
        f = st.file_uploader(f"Comparison {i + 1}", type=["json"], key=f"comp_{i}")
    with col_name:
        name = st.text_input(f"Label {i + 1}", key=f"comp_name_{i}")
    color_idx = st.sidebar.selectbox(
        f"Color {i + 1}",
        range(len(COLOR_OPTIONS)),
        format_func=lambda x: COLOR_OPTIONS[x][0],
        index=i % len(COLOR_OPTIONS),
        key=f"comp_color_{i}",
    )
    symbol_idx = st.sidebar.selectbox(
        f"Symbol {i + 1}",
        range(len(SYMBOL_OPTIONS)),
        format_func=lambda x: SYMBOL_OPTIONS[x][0],
        index=(i + 1) % len(SYMBOL_OPTIONS),
        key=f"comp_symbol_{i}",
    )
    comparison_uploads.append(
        {
            "file": f,
            "name": name,
            "color": COLOR_OPTIONS[color_idx][1],
            "symbol": SYMBOL_OPTIONS[symbol_idx][1],
        }
    )

# ── Load Data ────────────────────────────────────────────────────────────────

results = None

# Auto-load from --results path (passed via CLI)
if _PRELOAD_PATH and os.path.isfile(_PRELOAD_PATH):
    try:
        with open(_PRELOAD_PATH, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        st.sidebar.success(
            f"Auto-loaded: {os.path.basename(_PRELOAD_PATH)}\n"
            f"Topic: {results.get('topic', '?')} — "
            f"{results.get('summary', {}).get('total_configs', len(results.get('configurations', {})))} configs"
        )
    except Exception as e:
        st.sidebar.error(f"Error auto-loading {_PRELOAD_PATH}: {e}")

if uploaded_main:
    try:
        raw = json.loads(uploaded_main.read().decode("utf-8"))
        results = raw  # uploaded file overrides CLI pre-load
        st.sidebar.success(
            f"Loaded: {raw.get('topic', '?')} — "
            f"{raw.get('summary', {}).get('total_configs', '?')} configs"
        )
    except Exception as e:
        st.sidebar.error(f"Error loading JSON: {e}")

if results is None:
    st.info(
        "Upload a FairMOP results JSON in the sidebar to begin, "
        "or launch with:\n\n"
        "```\npython -m fairmop dashboard --results path/to/results.json\n```"
    )
    st.stop()

# ── Metric Selection ─────────────────────────────────────────────────────────

available_metrics = detect_metrics(results)

st.sidebar.header("📊 Metrics")

x_metric = st.sidebar.selectbox(
    "X-axis (Utility)",
    available_metrics,
    format_func=lambda m: METRIC_DISPLAY.get(m, m),
)
y_metric = st.sidebar.selectbox(
    "Y-axis (Fairness)",
    [m for m in available_metrics if m != x_metric],
    format_func=lambda m: METRIC_DISPLAY.get(m, m),
)

# ── Summary Table ────────────────────────────────────────────────────────────

st.header("📋 Configuration Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Topic", results.get("topic", "—"))
with col2:
    st.metric("Configurations", results.get("summary", {}).get("total_configs", "—"))
with col3:
    st.metric("Total Images", results.get("summary", {}).get("total_images", "—"))

df = build_summary_table(results, available_metrics)
st.dataframe(df, use_container_width=True)

# ── Pareto Chart ─────────────────────────────────────────────────────────────

st.header("📈 Pareto Frontier Analysis")

# Determine if metrics need inversion
fid_like = {"fid_score", "fid"}
utility_invert = x_metric in fid_like
fairness_invert = y_metric in fid_like

pareto_result = build_pareto_from_results(
    results,
    utility_metric=x_metric,
    fairness_metric=y_metric,
    utility_invert=utility_invert,
    fairness_invert=fairness_invert,
)

# Build scatter data
main_x = [p.utility for p in pareto_result.all_points]
main_y = [p.fairness for p in pareto_result.all_points]
main_labels = [p.config_name for p in pareto_result.all_points]

# Chart appearance
st.sidebar.header("🎨 Appearance")
main_color_idx = st.sidebar.selectbox(
    "Main color",
    range(len(COLOR_OPTIONS)),
    format_func=lambda x: COLOR_OPTIONS[x][0],
    index=5,
)
main_symbol_idx = st.sidebar.selectbox(
    "Main symbol",
    range(len(SYMBOL_OPTIONS)),
    format_func=lambda x: SYMBOL_OPTIONS[x][0],
    index=0,
)
pareto_color_idx = st.sidebar.selectbox(
    "Pareto color",
    range(len(COLOR_OPTIONS)),
    format_func=lambda x: COLOR_OPTIONS[x][0],
    index=1,
)
pareto_symbol_idx = st.sidebar.selectbox(
    "Pareto symbol",
    range(len(SYMBOL_OPTIONS)),
    format_func=lambda x: SYMBOL_OPTIONS[x][0],
    index=5,
)

main_label = st.sidebar.text_input("Main Legend Label", "Evaluated Configurations")
pareto_label = st.sidebar.text_input("Pareto Legend Label", "Pareto-Optimal")

fig = go.Figure()

# Main points
fig.add_trace(
    go.Scatter(
        x=main_x,
        y=main_y,
        mode="markers",
        marker=dict(
            size=14,
            color=COLOR_OPTIONS[main_color_idx][1],
            symbol=SYMBOL_OPTIONS[main_symbol_idx][1],
            line=dict(width=1, color="white"),
        ),
        text=main_labels,
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{METRIC_DISPLAY.get(x_metric, x_metric)}: %{{x:.4f}}<br>"
            f"{METRIC_DISPLAY.get(y_metric, y_metric)}: %{{y:.4f}}<br>"
            "<extra></extra>"
        ),
        name=main_label,
    )
)

# Pareto frontier
if pareto_result.pareto_points:
    px_pts = [p.utility for p in pareto_result.pareto_points]
    py_pts = [p.fairness for p in pareto_result.pareto_points]
    p_labels = [p.config_name for p in pareto_result.pareto_points]

    fig.add_trace(
        go.Scatter(
            x=px_pts,
            y=py_pts,
            mode="lines+markers",
            line=dict(color=COLOR_OPTIONS[pareto_color_idx][1], width=2, dash="dash"),
            marker=dict(
                size=16,
                color=COLOR_OPTIONS[pareto_color_idx][1],
                symbol=SYMBOL_OPTIONS[pareto_symbol_idx][1],
                line=dict(width=2, color="white"),
            ),
            text=p_labels,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{METRIC_DISPLAY.get(x_metric, x_metric)}: %{{x:.4f}}<br>"
                f"{METRIC_DISPLAY.get(y_metric, y_metric)}: %{{y:.4f}}<br>"
                "<extra></extra>"
            ),
            name=pareto_label,
        )
    )

# Comparison data
for comp in comparison_uploads:
    if comp["file"] is not None:
        try:
            comp_data = json.loads(comp["file"].read().decode("utf-8"))
            comp_pareto = build_pareto_from_results(
                comp_data,
                utility_metric=x_metric,
                fairness_metric=y_metric,
                utility_invert=utility_invert,
                fairness_invert=fairness_invert,
            )
            cx = [p.utility for p in comp_pareto.all_points]
            cy = [p.fairness for p in comp_pareto.all_points]
            cl = [p.config_name for p in comp_pareto.all_points]

            fig.add_trace(
                go.Scatter(
                    x=cx,
                    y=cy,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=comp["color"],
                        symbol=comp["symbol"],
                        line=dict(width=1, color="white"),
                    ),
                    text=cl,
                    name=comp["name"] or f"Comparison",
                )
            )
        except Exception as e:
            st.warning(f"Error loading comparison: {e}")

# Layout
fig.update_layout(
    xaxis_title=METRIC_DISPLAY.get(x_metric, x_metric),
    yaxis_title=METRIC_DISPLAY.get(y_metric, y_metric),
    height=700,
    legend=dict(
        orientation="h",
        x=0.5,
        y=1.1,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=13),
    ),
    margin=dict(l=80, r=40, t=100, b=80),
)

st.plotly_chart(fig, use_container_width=True)

# ── Pareto Summary ───────────────────────────────────────────────────────────

st.subheader("🏆 Pareto-Optimal Configurations")

if pareto_result.pareto_points:
    for p in pareto_result.pareto_points:
        st.success(
            f"**{p.config_name}** — "
            f"{METRIC_DISPLAY.get(x_metric, x_metric)}: {p.utility:.4f}, "
            f"{METRIC_DISPLAY.get(y_metric, y_metric)}: {p.fairness:.4f}"
        )
else:
    st.info("No Pareto-optimal configurations found.")

# ── Exports ──────────────────────────────────────────────────────────────────

st.header("💾 Export")

col_json, col_csv, col_chart = st.columns(3)

with col_json:
    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Results (JSON)",
        json_str,
        file_name=f"{results.get('topic', 'results')}_fairmop.json",
        mime="application/json",
    )

with col_csv:
    csv_str = df.to_csv(index=False)
    st.download_button(
        "Download Summary (CSV)",
        csv_str,
        file_name=f"{results.get('topic', 'results')}_summary.csv",
        mime="text/csv",
    )

with col_chart:
    try:
        html_data = fig.to_html(include_plotlyjs=True)
        st.download_button(
            "Download Chart (HTML)",
            html_data,
            file_name="pareto_frontier.html",
            mime="text/html",
        )
    except Exception:
        st.info("Chart export unavailable.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "FairMOP – Benchmarking Fairness-Utility Trade-offs in "
    "Text-to-Image Models via Pareto Frontiers"
)
