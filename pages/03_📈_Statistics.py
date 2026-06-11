"""pages/03_📈_Statistics.py — Statistical summaries."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("📈 Statistical Analysis")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

stats = profile.get("statistics", {})
num_stats = stats.get("numerical", {})
cat_stats = stats.get("categorical", {})

tab_num, tab_cat = st.tabs(["🔢 Numerical", "🏷 Categorical"])

# ── Numerical Statistics ──────────────────────────────────────────────────────
with tab_num:
    if not num_stats:
        st.info("No numerical columns found in this dataset.")
    else:
        # Summary table
        st.markdown("### Summary Statistics")
        summary_rows = []
        for col, s in num_stats.items():
            summary_rows.append({
                "Column": col,
                "Count": s.get("count", 0),
                "Mean": f"{s.get('mean', 0):.4f}",
                "Median": f"{s.get('median', 0):.4f}",
                "Std": f"{s.get('std', 0):.4f}",
                "Min": f"{s.get('min', 0):.4f}",
                "Max": f"{s.get('max', 0):.4f}",
                "Skewness": f"{s.get('skewness', 0):.3f}",
                "Kurtosis": f"{s.get('kurtosis', 0):.3f}",
                "Distribution": s.get("distribution_shape", "unknown"),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("### Column Detail")
        selected_col = st.selectbox("Select numerical column:", list(num_stats.keys()))
        if selected_col:
            s = num_stats[selected_col]
            c1, c2, c3, c4 = st.columns(4)
            for col_ui, label, val in [
                (c1, "Mean",     f"{s.get('mean', 0):.4f}"),
                (c2, "Median",   f"{s.get('median', 0):.4f}"),
                (c3, "Std Dev",  f"{s.get('std', 0):.4f}"),
                (c4, "IQR",      f"{s.get('iqr', 0):.4f}"),
            ]:
                col_ui.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            c1, c2, c3, c4 = st.columns(4)
            for col_ui, label, val in [
                (c1, "Skewness",  f"{s.get('skewness', 0):.4f}"),
                (c2, "Kurtosis",  f"{s.get('kurtosis', 0):.4f}"),
                (c3, "CV",        f"{s.get('cv', 0):.4f}" if s.get("cv") else "N/A"),
                (c4, "Shape",     s.get("distribution_shape", "unknown").replace("_", " ").title()),
            ]:
                col_ui.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Distribution interpretation
            shape = s.get("distribution_shape", "unknown")
            interp_map = {
                "normal":       "✅ Approximately normally distributed — safe to use with parametric models.",
                "right_skewed": "⚠️ Right-skewed — consider log or sqrt transform for linear models.",
                "left_skewed":  "⚠️ Left-skewed — consider reflection + log transform.",
                "leptokurtic":  "ℹ️ Heavy-tailed (leptokurtic) — may have significant outliers.",
                "uniform":      "ℹ️ Approximately uniform distribution — low information density.",
                "unknown":      "ℹ️ Distribution shape unclear — inspect histogram visually.",
            }
            st.markdown(
                f'<div class="info-box">{interp_map.get(shape, "")}</div>',
                unsafe_allow_html=True,
            )

            # Quantile table
            st.markdown("#### Quantile Summary")
            quantile_df = pd.DataFrame({
                "Statistic": ["Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max"],
                "Value": [
                    s.get("min"), s.get("q1"), s.get("median"),
                    s.get("q3"), s.get("max"),
                ],
            })
            st.dataframe(quantile_df, use_container_width=True, hide_index=True)

# ── Categorical Statistics ────────────────────────────────────────────────────
with tab_cat:
    if not cat_stats:
        st.info("No categorical columns found in this dataset.")
    else:
        from utils.formatters import classify_imbalance

        st.markdown("### Categorical Summary")
        cat_rows = []
        for col, s in cat_stats.items():
            ratio = s.get("class_imbalance_ratio")
            cat_rows.append({
                "Column": col,
                "Unique Values": s.get("unique_count", 0),
                "Top Category": list(s.get("top_categories", {}).keys())[0] if s.get("top_categories") else "N/A",
                "Dominant %": f"{s.get('dominant_pct', 0)*100:.1f}%",
                "Imbalance Ratio": f"{ratio:.2f}" if ratio else "N/A",
                "Imbalance Label": classify_imbalance(ratio),
                "Rare Categories": len(s.get("rare_categories", [])),
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        st.markdown("### Column Detail")
        selected_cat = st.selectbox("Select categorical column:", list(cat_stats.keys()))
        if selected_cat:
            s = cat_stats[selected_cat]
            top_cats = s.get("top_categories", {})
            rare_cats = s.get("rare_categories", [])
            ratio = s.get("class_imbalance_ratio")

            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"""<div class="metric-card">
                <div class="metric-label">Unique Values</div>
                <div class="metric-value">{s.get("unique_count", 0)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"""<div class="metric-card">
                <div class="metric-label">Dominant Category</div>
                <div class="metric-value" style="font-size:1rem;">{s.get("dominant_category") or "None"}</div>
                <div class="metric-sub">{s.get("dominant_pct", 0)*100:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"""<div class="metric-card">
                <div class="metric-label">Class Imbalance</div>
                <div class="metric-value" style="font-size:1rem;">{classify_imbalance(ratio)}</div>
                <div class="metric-sub">ratio: {f"{ratio:.1f}:1" if ratio else "N/A"}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Top categories bar chart
            if top_cats:
                from utils.chart_factory import make_bar_chart
                fig = make_bar_chart(
                    labels=list(top_cats.keys()),
                    values=list(top_cats.values()),
                    title=f"Top Categories: {selected_cat}",
                    horizontal=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            if rare_cats:
                with st.expander(f"Rare categories ({len(rare_cats)}) — appearing in <1% of rows"):
                    st.write(", ".join(str(c) for c in rare_cats[:30]))
