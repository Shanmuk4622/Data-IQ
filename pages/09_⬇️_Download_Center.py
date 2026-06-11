"""pages/09_⬇️_Download_Center.py — All exports."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import json


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("⬇️ Download Center")
st.caption("Export everything — cleaned data, reports, notebooks, pipelines")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

health = st.session_state.get("health_score") or profile.get("health_score", {})
groq_report = st.session_state.get("groq_report", "")
df_clean = st.session_state.get("df_clean")

# ── Data Files ────────────────────────────────────────────────────────────────
st.markdown("## 📁 Data Files")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""<div class="metric-card">
    <div class="metric-label">Original Dataset</div>
    <div class="metric-value" style="font-size:1rem;">Raw CSV</div>
    </div>""", unsafe_allow_html=True)
    csv_raw = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Raw CSV",
        data=csv_raw,
        file_name=f"{profile.get('meta', {}).get('file_name', 'dataset')}_raw.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    if df_clean is not None:
        st.markdown("""<div class="metric-card" style="border-color:rgba(16,185,129,0.4);">
        <div class="metric-label">Cleaned Dataset</div>
        <div class="metric-value" style="font-size:1rem;color:#10B981;">Ready for ML</div>
        </div>""", unsafe_allow_html=True)
        csv_clean = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned CSV",
            data=csv_clean,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.markdown("""<div class="metric-card" style="opacity:0.5;">
        <div class="metric-label">Cleaned Dataset</div>
        <div class="metric-value" style="font-size:0.9rem;">Not yet — run preprocessing first</div>
        </div>""", unsafe_allow_html=True)
        st.button("📥 Download Cleaned CSV", disabled=True, use_container_width=True)

with c3:
    st.markdown("""<div class="metric-card">
    <div class="metric-label">Excel Export</div>
    <div class="metric-value" style="font-size:1rem;">.xlsx</div>
    </div>""", unsafe_allow_html=True)
    try:
        import io
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "📥 Download as Excel",
            data=buf.getvalue(),
            file_name="dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.caption(f"Excel export unavailable: {e}")

st.divider()

# ── Reports ───────────────────────────────────────────────────────────────────
st.markdown("## 📄 Reports")
r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("""<div class="metric-card">
    <div class="metric-label">JSON Report</div>
    <div class="metric-value" style="font-size:1rem;">Full Profile</div>
    </div>""", unsafe_allow_html=True)
    from modules.reporter import generate_json_report
    json_report = generate_json_report(profile)
    st.download_button(
        "📥 Download JSON Report",
        data=json_report.encode("utf-8"),
        file_name="dataiq_report.json",
        mime="application/json",
        use_container_width=True,
    )

with r2:
    st.markdown("""<div class="metric-card">
    <div class="metric-label">Markdown Report</div>
    <div class="metric-value" style="font-size:1rem;">GitHub-ready</div>
    </div>""", unsafe_allow_html=True)
    from modules.reporter import generate_markdown_report
    md_report = generate_markdown_report(profile, health, groq_report)
    st.download_button(
        "📥 Download Markdown Report",
        data=md_report.encode("utf-8"),
        file_name="dataiq_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

with r3:
    st.markdown("""<div class="metric-card">
    <div class="metric-label">HTML Report</div>
    <div class="metric-value" style="font-size:1rem;">Self-contained</div>
    </div>""", unsafe_allow_html=True)
    from modules.reporter import generate_html_report
    html_report = generate_html_report(profile, health, groq_report)
    st.download_button(
        "📥 Download HTML Report",
        data=html_report.encode("utf-8"),
        file_name="dataiq_report.html",
        mime="text/html",
        use_container_width=True,
    )

st.divider()

# ── Jupyter Notebook ──────────────────────────────────────────────────────────
st.markdown("## 📓 Jupyter Notebook")
st.caption("Auto-generated ML pipeline notebook — open in Jupyter or Google Colab")

nb_col, nb_info = st.columns([1, 2])
with nb_col:
    try:
        from modules.notebook_generator import generate_notebook, notebook_to_bytes
        pipeline_params = st.session_state.get("cleaning_choices", {})
        nb = generate_notebook(profile, pipeline_params)
        nb_bytes = notebook_to_bytes(nb)
        st.download_button(
            "📥 Download Jupyter Notebook",
            data=nb_bytes,
            file_name="dataiq_ml_pipeline.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Notebook generation failed: {e}")

with nb_info:
    st.markdown("""<div class="info-box">
    <strong>What's in the notebook?</strong><br>
    • Data loading cell with your filename<br>
    • Quick EDA summary<br>
    • Auto-generated preprocessing pipeline<br>
    • Train/test split<br>
    • Top recommended model with starter code<br>
    • Evaluation metrics
    </div>""", unsafe_allow_html=True)

st.divider()

# ── AI Report ─────────────────────────────────────────────────────────────────
st.markdown("## 🧠 AI Analysis Report")
if groq_report:
    st.download_button(
        "📥 Download AI Report (.md)",
        data=groq_report.encode("utf-8"),
        file_name="ai_analysis_report.md",
        mime="text/markdown",
    )
    with st.expander("Preview AI Report"):
        st.markdown(groq_report)
else:
    st.markdown(
        '<div class="info-box">Generate the AI report on the <strong>AI Explanation</strong> page first.</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Dataset Comparator ────────────────────────────────────────────────────────
st.markdown("## 🔄 Dataset Comparator")
st.caption("Compare your dataset against another (e.g. train vs. test, raw vs. cleaned)")

comp_col1, comp_col2 = st.columns(2)
with comp_col1:
    comp_file = st.file_uploader(
        "Upload second dataset to compare:",
        type=["csv", "xlsx", "json", "parquet"],
        key="comparator_upload",
    )

if comp_file is not None and df is not None:
    try:
        import pandas as pd
        if comp_file.name.endswith(".csv"):
            df_b = pd.read_csv(comp_file)
        elif comp_file.name.endswith(".json"):
            df_b = pd.read_json(comp_file)
        else:
            df_b = pd.read_excel(comp_file)
        
        from modules.dataset_comparator import compare_datasets
        with st.spinner("Comparing datasets…"):
            comp_result = compare_datasets(
                df, df_b,
                name_a=st.session_state.get("file_name", "Dataset A"),
                name_b=comp_file.name,
            )
        
        summary = comp_result.get("summary", {})
        size_diff = comp_result.get("size_diff", {})
        
        st.markdown("### Comparison Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Schema Changes", summary.get("n_schema_changes", 0))
        c2.metric("Distribution Shifts", summary.get("n_distribution_shifts", 0))
        c3.metric("Row Delta", f"{size_diff.get('row_diff', 0):+,}")
        
        # Schema diff
        schema = comp_result.get("schema_diff", {})
        if schema.get("cols_only_in_a"):
            st.markdown(f"**Columns only in A:** {', '.join(f'`{c}`' for c in schema['cols_only_in_a'])}")
        if schema.get("cols_only_in_b"):
            st.markdown(f"**Columns only in B:** {', '.join(f'`{c}`' for c in schema['cols_only_in_b'])}")
        
        # Distribution shifts
        shifts = comp_result.get("distribution_shifts", {})
        shifted = [c for c, v in shifts.items() if v.get("shift_detected")]
        if shifted:
            st.markdown(f"**⚠️ Distribution shift detected in:** {', '.join(f'`{c}`' for c in shifted)}")
        
        # Download comparison
        comp_json = json.dumps(comp_result, indent=2, default=str)
        st.download_button(
            "📥 Download Comparison Report (JSON)",
            data=comp_json.encode("utf-8"),
            file_name="dataset_comparison.json",
            mime="application/json",
        )
    
    except Exception as e:
        st.error(f"Comparison failed: {e}")
