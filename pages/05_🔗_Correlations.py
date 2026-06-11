"""pages/05_🔗_Correlations.py — Correlation heatmaps and pair plots."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils.chart_factory import make_heatmap, make_pair_plot, make_scatter


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🔗 Correlations")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

corr_data = profile.get("correlations", {})
columns_profile = profile.get("columns", {})
num_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") == "numerical"]

if corr_data.get("insufficient_columns"):
    st.warning("⚠️ Need at least 3 numerical columns for correlation analysis.")
    st.stop()

if corr_data.get("error"):
    st.error(f"Correlation computation failed: {corr_data['error']}")
    st.stop()

# ── High Correlation Pairs ────────────────────────────────────────────────────
high_pairs = corr_data.get("high_correlation_pairs", [])
if high_pairs:
    st.markdown("## ⚡ High Correlation Pairs (|r| > 0.8)")
    for pair in high_pairs[:10]:
        r = pair["pearson_r"]
        color = "#EF4444" if abs(r) > 0.95 else "#F59E0B"
        st.markdown(
            f"""<div class="metric-card" style="border-color:{color}33;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <strong style="color:{color};">{pair['col_a']}</strong>
                    <span style="color:#64748B;"> ↔ </span>
                    <strong style="color:{color};">{pair['col_b']}</strong>
                </div>
                <div style="font-size:1.5rem;font-weight:800;color:{color};">r = {r:.3f}</div>
            </div>
            <div style="color:#94A3B8;font-size:0.8rem;margin-top:0.3rem;">{pair['recommendation']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ── Correlation Method Selector ───────────────────────────────────────────────
method = st.radio(
    "Correlation method:",
    ["pearson", "spearman", "kendall"],
    horizontal=True,
    help="Pearson: linear; Spearman: rank-based; Kendall: robust to outliers",
)

corr_dict = corr_data.get(method, {})
if corr_dict:
    corr_df = pd.DataFrame(corr_dict)
    
    tab_heat, tab_pair, tab_target = st.tabs(["🗺 Heatmap", "🔵 Pair Plot", "🎯 Target Correlations"])
    
    with tab_heat:
        fig = make_heatmap(corr_df, title=f"{method.title()} Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_pair:
        st.markdown("### Scatter Matrix (Top 6 by variance)")
        cat_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") in ("categorical", "boolean")]
        
        # Top columns by variance
        num_df = df[num_cols].select_dtypes(include="number")
        top_var_cols = num_df.var().nlargest(min(6, len(num_cols))).index.tolist()
        
        color_opts = ["(none)"] + cat_cols
        color_col = st.selectbox("Color by:", color_opts, key="pair_color")
        gc = None if color_col == "(none)" else color_col
        
        if len(top_var_cols) >= 2:
            fig = make_pair_plot(df, top_var_cols, color_col=gc, title="Pair Plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical columns for pair plot.")
    
    with tab_target:
        target = st.session_state.get("target_col")
        if target and target in corr_df.columns:
            from modules.correlations import get_feature_importance_order
            pairs = get_feature_importance_order(corr_data.get("pearson", {}), target)
            
            if pairs:
                st.markdown(f"### Feature Correlations with Target: `{target}`")
                cols_list = [p[0] for p in pairs]
                vals_list = [p[1] for p in pairs]
                
                from utils.chart_factory import make_bar_chart
                fig = make_bar_chart(
                    labels=cols_list,
                    values=vals_list,
                    title=f"|Pearson r| with {target}",
                    horizontal=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                pairs_df = pd.DataFrame([
                    {
                        "Feature": col,
                        "|r|": f"{val:.4f}",
                        "r": f"{corr_data.get('pearson', {}).get(target, {}).get(col, 0):.4f}",
                        "Strength": (
                            "Strong" if val > 0.6 else
                            "Moderate" if val > 0.3 else
                            "Weak"
                        ),
                    }
                    for col, val in pairs
                ])
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
        else:
            st.info("Set a target column in the Overview page to see feature importance.")
