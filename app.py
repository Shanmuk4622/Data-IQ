"""app.py — DataIQ Main Entry Point."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

# ── Page config must be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="DataIQ — Dataset Intelligence Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "df_raw": None,
    "df_clean": None,
    "file_name": None,
    "file_size_mb": None,
    "profile": None,
    "target_col": None,
    "problem_type": None,
    "groq_report": None,
    "pipeline": None,
    "pipeline_params": {},
    "report_html": None,
    "health_score": None,
    "analysis_complete": False,
    "demo_mode": False,
    "groq_api_key": "",
    "cleaning_choices": {},
}

for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helper: run full analysis ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_full_analysis(df: pd.DataFrame, target_col: str | None) -> dict:
    """Master analysis — runs all modules. Cached by df hash + target_col."""
    from modules.profiler import profile_columns
    from modules.quality import analyze_quality
    from modules.statistics import compute_statistics
    from modules.outliers import detect_outliers
    from modules.correlations import compute_correlations
    from modules.datetime_analysis import analyze_datetime
    from modules.text_analysis import analyze_text
    from modules.ml_advisor import assess_ml_readiness
    from modules.preprocessor import recommend_preprocessing_all
    from modules.feature_engineer import suggest_features
    from modules.health_score import compute_health_score

    profile: dict = {}

    with st.spinner("Running full dataset analysis…"):
        pb = st.progress(0, text="Profiling columns…")

        # Meta
        profile["meta"] = {
            "file_name": st.session_state.get("file_name", "unknown"),
            "rows": len(df),
            "columns": df.shape[1],
            "total_cells": df.size,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        }

        # Columns
        profile["columns"] = profile_columns(df)
        pb.progress(15, text="Analysing data quality…")

        profile["quality"] = analyze_quality(df)
        pb.progress(30, text="Computing statistics…")

        profile["statistics"] = compute_statistics(df, profile["columns"])
        pb.progress(45, text="Detecting outliers…")

        profile["outliers"] = detect_outliers(df, profile["columns"])
        pb.progress(55, text="Computing correlations…")

        profile["correlations"] = compute_correlations(df)
        pb.progress(65, text="Analysing datetime columns…")

        profile["datetime_analysis"] = analyze_datetime(df)
        profile["text_analysis"] = analyze_text(df, profile["columns"])
        pb.progress(72, text="Assessing ML readiness…")

        profile["ml_readiness"] = assess_ml_readiness(df, target_col, profile["columns"], profile)
        if target_col:
            profile["ml_readiness"]["target_column"] = target_col
        pb.progress(82, text="Building preprocessing recommendations…")

        profile["preprocessing_recommendations"] = recommend_preprocessing_all(df, profile)
        pb.progress(90, text="Generating feature engineering suggestions…")

        profile["feature_engineering"] = suggest_features(df, profile)
        pb.progress(96, text="Computing health score…")

        profile["health_score"] = compute_health_score(profile)
        pb.progress(100, text="Analysis complete!")

    return profile


def load_demo(name: str):
    """Load a built-in sample dataset into session state."""
    from modules.loader import load_sample_dataset
    try:
        df, meta = load_sample_dataset(name)
        st.session_state["df_raw"] = df
        st.session_state["file_name"] = meta["file_name"]
        st.session_state["file_size_mb"] = meta["file_size_mb"]
        st.session_state["demo_mode"] = True
        st.session_state["analysis_complete"] = False
        st.session_state["profile"] = None
        st.session_state["groq_report"] = None
        st.session_state["df_clean"] = None
        st.session_state["target_col"] = None
        run_analysis_trigger()
    except Exception as e:
        st.error(f"Failed to load sample dataset: {e}")


def run_analysis_trigger():
    """Trigger (or re-trigger) full analysis and store results."""
    df = st.session_state.get("df_raw")
    if df is None:
        return
    target_col = st.session_state.get("target_col")

    # Large dataset warning
    if len(df) > 50_000:
        st.warning(
            f"Dataset has {len(df):,} rows (>50K). "
            "DataIQ will sample down to 50,000 rows for analysis performance."
        )
        df = df.sample(50_000, random_state=42)

    profile = run_full_analysis(df, target_col)
    st.session_state["profile"] = profile
    st.session_state["health_score"] = profile.get("health_score", {})

    # Auto-detect target if not set
    if not st.session_state.get("target_col"):
        from modules.target_detector import detect_target_column
        best_col, conf = detect_target_column(df, profile["columns"])
        if best_col:
            st.session_state["target_col"] = best_col
        st.session_state["problem_type"] = profile.get("ml_readiness", {}).get("problem_type")

    st.session_state["analysis_complete"] = True


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="gradient-title" style="font-size:1.6rem;margin-bottom:0.3rem;">📊 DataIQ</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-Powered Dataset Intelligence")
    st.divider()

    # Demo datasets
    st.markdown("**Try a sample dataset**")
    c1, c2, c3 = st.columns(3)
    if c1.button("Titanic", use_container_width=True, help="891 passengers, survival classification"):
        load_demo("titanic")
        st.rerun()
    if c2.button("Housing", use_container_width=True, help="2,000 houses, price regression"):
        load_demo("housing")
        st.rerun()
    if c3.button("E-comm", use_container_width=True, help="1,500 orders, messy real-world data"):
        load_demo("ecommerce")
        st.rerun()

    st.divider()

    # File upload
    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
        help="Max 200 MB. Supported: CSV, Excel, JSON, Parquet, TSV",
    )
    if uploaded is not None:
        # Only reload if file changes
        if uploaded.name != st.session_state.get("file_name"):
            from modules.loader import load_file
            try:
                df, meta = load_file(uploaded)
                st.session_state["df_raw"] = df
                st.session_state["file_name"] = meta["file_name"]
                st.session_state["file_size_mb"] = meta["file_size_mb"]
                st.session_state["demo_mode"] = False
                st.session_state["analysis_complete"] = False
                st.session_state["profile"] = None
                st.session_state["groq_report"] = None
                st.session_state["df_clean"] = None
                st.session_state["target_col"] = None
                run_analysis_trigger()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

    # Groq API key
    key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        placeholder="gsk_...",
        help="Free at console.groq.com — enables AI explanations",
    )
    if key_input:
        st.session_state["groq_api_key"] = key_input

    st.divider()

    # Analysis status
    if st.session_state.get("df_raw") is not None:
        st.markdown("**Analysis Status**")
        status = st.session_state.get("analysis_complete", False)
        profile = st.session_state.get("profile", {}) or {}

        def _check(key):
            return "✅" if profile.get(key) else "⬜"

        items = [
            ("✅", "Dataset loaded"),
            (_check("columns"), "Profiling"),
            (_check("quality"), "Quality analysis"),
            (_check("statistics"), "Statistics"),
            (_check("outliers"), "Outlier detection"),
            (_check("correlations"), "Correlations"),
            (_check("ml_readiness"), "ML readiness"),
            (_check("preprocessing_recommendations"), "Preprocessing"),
            ("✅" if st.session_state.get("groq_report") else "⬜", "AI explanation"),
        ]
        for icon, label in items:
            st.markdown(f"{icon} {label}")

        if st.button("🔄 Re-run Analysis", use_container_width=True):
            st.session_state["analysis_complete"] = False
            st.session_state["profile"] = None
            run_full_analysis.clear()
            run_analysis_trigger()
            st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
df = st.session_state.get("df_raw")

if df is None:
    # Welcome screen
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon">📊</div>
            <h1 class="gradient-title">DataIQ</h1>
            <p style="font-size:1.1rem;color:#94A3B8;max-width:600px;margin:0 auto 2rem;">
                AI-powered Exploratory Data Analysis, preprocessing, and dataset intelligence.
                Upload any structured dataset or try a sample to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🔍 Automated EDA</div>
            <div style="color:#C4B5FD;font-size:0.9rem;margin-top:0.5rem;">
                Full statistical profiling, outlier detection, correlation analysis — zero config required.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">⚙️ Smart Preprocessing</div>
            <div style="color:#C4B5FD;font-size:0.9rem;margin-top:0.5rem;">
                Auto-builds sklearn pipelines with imputation, encoding, scaling — export ready to use.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🧠 AI Explanations</div>
            <div style="color:#C4B5FD;font-size:0.9rem;margin-top:0.5rem;">
                Groq AI models write a professional data science report explaining every finding.
            </div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Use the sidebar to upload a dataset or try Titanic, Housing, or E-commerce demo data.")

else:
    # Run analysis if not done yet
    if not st.session_state.get("analysis_complete"):
        run_analysis_trigger()

    profile = st.session_state.get("profile") or {}
    health = st.session_state.get("health_score") or {}
    meta = profile.get("meta", {})

    # Header
    file_name = st.session_state.get("file_name", "")
    if st.session_state.get("demo_mode"):
        st.caption("🎭 Demo Mode — using sample dataset")
    st.markdown(
        f'<h1 class="gradient-title" style="font-size:1.8rem;">📊 {file_name}</h1>',
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "Rows",    f"{meta.get('rows', 0):,}",       ""),
        (c2, "Columns", str(meta.get("columns", 0)),       ""),
        (c3, "Memory",  f"{meta.get('memory_mb', 0):.1f} MB", ""),
        (c4, "Missing", f"{len(profile.get('quality', {}).get('missing_values', {}))} cols", ""),
        (c5, "Outlier Cols", str(len(profile.get("outliers", {}))), ""),
        (c6, "Health",  f"{health.get('total', 0):.0f}/100", f"Grade {health.get('grade', '?')}"),
    ]
    for col, label, val, sub in metrics:
        sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>{sub_html}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "**Navigate using the sidebar pages** to explore EDA, statistics, visualisations, "
        "correlations, preprocessing, ML recommendations, AI explanations, and downloads.",
    )
    st.markdown(
        "Start with **01 Overview** → **02 Data Quality** → **08 AI Explanation** → **09 Downloads**."
    )
