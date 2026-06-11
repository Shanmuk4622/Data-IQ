"""pages/01_📊_Overview.py — Dataset overview and health score."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils.chart_factory import make_health_gauge, make_radar_chart, make_bar_chart
from utils.formatters import DTYPE_COLORS, fmt_number, fmt_pct


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("📊 Dataset Overview")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")
health = st.session_state.get("health_score") or {}

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar to get started.")
    st.stop()

meta = profile.get("meta", {})
columns_profile = profile.get("columns", {})

# ── Health Score ──────────────────────────────────────────────────────────────
st.markdown("## 🏥 Dataset Health Score")
col_gauge, col_radar = st.columns([1, 1])

with col_gauge:
    fig = make_health_gauge(health.get("total", 0), health.get("grade", "?"))
    st.plotly_chart(fig, use_container_width=True)
    
    grade = health.get("grade", "?")
    fix = health.get("one_line_fix", "")
    worst = health.get("worst_dimension", "")
    
    st.markdown(
        f'<div class="warning-box">⚡ <strong>Priority fix:</strong> {fix}<br>'
        f'<small>Weakest dimension: <strong>{worst}</strong> '
        f'({health.get("dimensions", {}).get(worst, 0):.0f}/100)</small></div>',
        unsafe_allow_html=True,
    )

with col_radar:
    dims = health.get("dimensions", {})
    if dims:
        fig = make_radar_chart(
            list(dims.keys()),
            list(dims.values()),
            title="Health Dimensions",
        )
        st.plotly_chart(fig, use_container_width=True)

# Score breakdown table
st.markdown("#### Score Breakdown")
breakdown_data = {
    "Dimension": ["Missing Values", "Outliers", "Duplicates", "Class Balance", "Correlations"],
    "Score": [
        health.get("missing_score", 0),
        health.get("outlier_score", 0),
        health.get("duplicate_score", 0),
        health.get("class_balance_score", 0),
        health.get("correlation_score", 0),
    ],
    "Weight": ["30%", "20%", "15%", "20%", "15%"],
}
bdf = pd.DataFrame(breakdown_data)
bdf["Score"] = bdf["Score"].apply(lambda x: f"{x:.1f}")
st.dataframe(bdf, use_container_width=True, hide_index=True)

st.divider()

# ── Column Type Distribution ──────────────────────────────────────────────────
st.markdown("## 📋 Column Overview")

from collections import Counter
dtype_counts = Counter(v.get("dtype_class") for v in columns_profile.values())

cols_bar_col, cols_table_col = st.columns([1, 2])

with cols_bar_col:
    if dtype_counts:
        colors = [DTYPE_COLORS.get(k, "#94A3B8") for k in dtype_counts.keys()]
        fig = make_bar_chart(
            list(dtype_counts.keys()),
            list(dtype_counts.values()),
            title="Column Types",
            horizontal=True,
            color=colors,
        )
        st.plotly_chart(fig, use_container_width=True)

with cols_table_col:
    # Column summary table
    rows = []
    for col, info in columns_profile.items():
        rows.append({
            "Column": col,
            "Type": info.get("dtype_class", "?"),
            "Non-Null": fmt_number(info.get("non_null_count", 0)),
            "Nulls": f"{info.get('null_pct', 0)*100:.1f}%",
            "Unique": fmt_number(info.get("unique_count", 0)),
            "Target Score": f"{info.get('target_candidate_score', 0):.2f}",
        })
    col_df = pd.DataFrame(rows)

    # Style the type column
    st.dataframe(col_df, use_container_width=True, hide_index=True, height=420)

st.divider()

# ── Target Column Selection ───────────────────────────────────────────────────
st.markdown("## 🎯 Target Column")

from modules.target_detector import get_top_candidates
candidates = get_top_candidates(columns_profile, top_n=3)

st.markdown("**Top 3 target column candidates (auto-detected):**")
c1, c2, c3 = st.columns(3)
for i, (cand, col) in enumerate(zip(candidates, [c1, c2, c3])):
    score_pct = int(cand["score"] * 100)
    col.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">Rank #{i+1}</div>
        <div class="metric-value" style="font-size:1.1rem;">{cand['col']}</div>
        <div class="metric-sub">{cand['dtype']} • Score: {score_pct}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

all_cols = ["None (unsupervised)"] + list(df.columns)
current_target = st.session_state.get("target_col")
default_idx = all_cols.index(current_target) if current_target in all_cols else 0

selected = st.selectbox(
    "Override target column:",
    options=all_cols,
    index=default_idx,
    help="DataIQ auto-detects the most likely target. You can override it here.",
)

if st.button("✅ Apply Target Selection", type="primary"):
    new_target = None if selected == "None (unsupervised)" else selected
    st.session_state["target_col"] = new_target
    # Clear cached analysis to re-run
    st.session_state["analysis_complete"] = False
    st.session_state["profile"] = None
    from app import run_full_analysis
    run_full_analysis.clear()
    st.success(f"Target set to: **{new_target or 'None'}** — re-running analysis…")
    st.rerun()

st.divider()

# ── Sample Data Preview ───────────────────────────────────────────────────────
st.markdown("## 🗂 Data Preview")
n_rows = st.slider("Rows to preview", 5, min(100, len(df)), 10)
st.dataframe(df.head(n_rows), use_container_width=True)
