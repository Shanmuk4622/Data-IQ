"""pages/06_⚙️_Preprocessing.py — Pipeline builder and data cleaning."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import json


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("⚙️ Preprocessing")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

preprocessing_recs = profile.get("preprocessing_recommendations", {})
columns_profile = profile.get("columns", {})

# ── Step 1: Configure preprocessing per column ────────────────────────────────
st.markdown("## ⚙️ Configure Preprocessing Steps")
st.caption("DataIQ auto-recommends strategies — override any column below.")

user_choices = st.session_state.get("cleaning_choices", {})

# Table-style editor
with st.expander("📋 Column-by-Column Configuration", expanded=True):
    new_choices = {}
    header_cols = st.columns([3, 2, 2, 2, 1])
    for col_ui, label in zip(header_cols, ["Column", "Imputation", "Encoding", "Scaling", "Drop"]):
        col_ui.markdown(f"**{label}**")

    for col in df.columns:
        recs = preprocessing_recs.get(col, {})
        prev = user_choices.get(col, recs)
        dtype = columns_profile.get(col, {}).get("dtype_class", "?")
        
        c_name, c_imp, c_enc, c_scl, c_drop = st.columns([3, 2, 2, 2, 1])
        c_name.markdown(f"`{col}` <small style='color:#64748B;'>({dtype})</small>", unsafe_allow_html=True)
        
        imp_options = ["mean", "median", "mode", "none"]
        enc_options = ["none", "one_hot", "ordinal", "frequency"]
        scl_options = ["none", "standard", "robust", "minmax"]
        
        new_choices[col] = {
            "imputation": c_imp.selectbox(
                "imp", imp_options,
                index=imp_options.index(prev.get("imputation", "mean")) if prev.get("imputation") in imp_options else 0,
                key=f"imp_{col}", label_visibility="collapsed",
            ),
            "encoding": c_enc.selectbox(
                "enc", enc_options,
                index=enc_options.index(prev.get("encoding", "none")) if prev.get("encoding") in enc_options else 0,
                key=f"enc_{col}", label_visibility="collapsed",
            ),
            "scaling": c_scl.selectbox(
                "scl", scl_options,
                index=scl_options.index(prev.get("scaling", "none")) if prev.get("scaling") in scl_options else 0,
                key=f"scl_{col}", label_visibility="collapsed",
            ),
            "drop": c_drop.checkbox(
                "drop", value=prev.get("drop", False),
                key=f"drop_{col}", label_visibility="collapsed",
            ),
        }

    st.session_state["cleaning_choices"] = new_choices

st.divider()

# ── Step 2: Apply Cleaning ────────────────────────────────────────────────────
st.markdown("## 🧹 Apply Automated Cleaning")

if st.button("🚀 Clean Dataset", type="primary", use_container_width=False):
    from modules.preprocessor import apply_cleaning
    with st.spinner("Cleaning dataset…"):
        df_clean, report = apply_cleaning(df, profile, st.session_state["cleaning_choices"])
        st.session_state["df_clean"] = df_clean
        st.session_state["cleaning_report"] = report
    st.success("Dataset cleaned successfully!")

if st.session_state.get("df_clean") is not None:
    df_clean = st.session_state["df_clean"]
    report = st.session_state.get("cleaning_report", {})

    # Before/After comparison
    st.markdown("### Before / After Comparison")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Before:**")
        b_shape = report.get("shape_before", df.shape)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Shape</div>
        <div class="metric-value" style="font-size:1.1rem;">{b_shape[0]:,} × {b_shape[1]}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Missing Cells</div>
        <div class="metric-value" style="font-size:1.1rem;">{report.get('null_count_before', 0):,}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Memory</div>
        <div class="metric-value" style="font-size:1.1rem;">{report.get('memory_before_mb', 0):.2f} MB</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("**After:**")
        a_shape = report.get("shape_after", df_clean.shape)
        st.markdown(f"""<div class="metric-card" style="border-color:rgba(16,185,129,0.4);">
        <div class="metric-label">Shape</div>
        <div class="metric-value" style="font-size:1.1rem;color:#10B981;">{a_shape[0]:,} × {a_shape[1]}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card" style="border-color:rgba(16,185,129,0.4);">
        <div class="metric-label">Missing Cells</div>
        <div class="metric-value" style="font-size:1.1rem;color:#10B981;">{report.get('null_count_after', 0):,}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card" style="border-color:rgba(16,185,129,0.4);">
        <div class="metric-label">Memory</div>
        <div class="metric-value" style="font-size:1.1rem;color:#10B981;">{report.get('memory_after_mb', 0):.2f} MB</div>
        </div>""", unsafe_allow_html=True)

    # What was done
    dropped = report.get("dropped_columns", [])
    removed_dupes = report.get("removed_duplicates", 0)
    imputed = report.get("imputed_columns", {})
    
    if dropped:
        st.markdown(f"- 🗑 **Dropped {len(dropped)} columns:** {', '.join(f'`{c}`' for c in dropped)}")
    if removed_dupes:
        st.markdown(f"- 🗑 **Removed {removed_dupes:,} duplicate rows**")
    if imputed:
        st.markdown(f"- 🔧 **Imputed {len(imputed)} columns:** {', '.join(f'`{c}`' for c in list(imputed.keys())[:5])}")

    st.divider()
    
    # Preview
    st.markdown("### Cleaned Dataset Preview")
    st.dataframe(df_clean.head(20), use_container_width=True)

    # Download buttons
    st.markdown("### Downloads")
    dc1, dc2, dc3 = st.columns(3)
    
    with dc1:
        csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with dc2:
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            "📥 Download Cleaning Report (JSON)",
            data=report_json.encode("utf-8"),
            file_name="preprocessing_report.json",
            mime="application/json",
            use_container_width=True,
        )
    
    with dc3:
        # Build and download pipeline
        try:
            from modules.preprocessor import build_pipeline, pipeline_to_bytes
            pipeline = build_pipeline(df_clean, profile, st.session_state["cleaning_choices"])
            if pipeline:
                pkl_bytes = pipeline_to_bytes(pipeline)
                st.session_state["pipeline"] = pipeline
                st.download_button(
                    "📥 Download Pipeline (.pkl)",
                    data=pkl_bytes,
                    file_name="preprocessing_pipeline.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        except Exception as e:
            st.caption(f"Pipeline export unavailable: {e}")

    st.divider()

# ── Feature Engineering Suggestions ──────────────────────────────────────────
st.markdown("## 💡 Feature Engineering Suggestions")
fe_suggestions = profile.get("feature_engineering", [])

if not fe_suggestions:
    st.info("No automatic feature engineering suggestions for this dataset.")
else:
    st.caption(f"{len(fe_suggestions)} suggestions found")
    for s in fe_suggestions:
        with st.expander(f"💡 `{s['source_col']}` → {s['suggestion'][:60]}…"):
            st.markdown(f"**New column:** `{s['new_col_name']}`")
            st.markdown(f"**Rationale:** {s['rationale']}")
            if s.get("preview"):
                st.markdown(f"**Preview:** {s['preview']}")
            st.code(s["example_code"], language="python")
