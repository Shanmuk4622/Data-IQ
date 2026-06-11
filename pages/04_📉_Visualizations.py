"""pages/04_📉_Visualizations.py — Distribution charts, box plots, violins."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils.chart_factory import (
    make_histogram, make_boxplot, make_violin,
    make_bar_chart, make_pie, make_treemap, make_qq_plot,
)


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("📉 Visualizations")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

columns_profile = profile.get("columns", {})
num_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") == "numerical"]
cat_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") in ("categorical", "boolean")]

tab_dist, tab_box, tab_cat, tab_custom = st.tabs(["📊 Distributions", "📦 Box / Violin", "🏷 Categorical", "🔧 Custom"])

# ── Distribution charts ───────────────────────────────────────────────────────
with tab_dist:
    if not num_cols:
        st.info("No numerical columns available.")
    else:
        st.markdown("### Histogram + KDE for Numerical Columns")
        n_per_row = 2
        for i in range(0, len(num_cols), n_per_row):
            cols_row = st.columns(n_per_row)
            for j, col_name in enumerate(num_cols[i:i + n_per_row]):
                with cols_row[j]:
                    fig = make_histogram(df[col_name], title=col_name)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.markdown("### Q-Q Plot (Normality Check)")
        selected_qq = st.selectbox("Select column for Q-Q plot:", num_cols, key="qq_col")
        if selected_qq:
            fig = make_qq_plot(df[selected_qq], title=f"Q-Q Plot: {selected_qq}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("If points follow the green dashed line, the column is approximately normally distributed.")

# ── Box + Violin ──────────────────────────────────────────────────────────────
with tab_box:
    if not num_cols:
        st.info("No numerical columns available.")
    else:
        st.markdown("### Box Plots (All Numerical)")
        # Show up to 8 at a time to avoid clutter
        show_cols = num_cols[:8]
        fig = make_boxplot(df, show_cols, title="Box Plots — Numerical Columns")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Violin Plot")
        c1, c2 = st.columns(2)
        vio_col = c1.selectbox("Numerical column:", num_cols, key="vio_num")
        group_opts = ["(none)"] + cat_cols
        group_col = c2.selectbox("Group by (optional):", group_opts, key="vio_cat")
        gc = None if group_col == "(none)" else group_col
        
        if vio_col:
            fig = make_violin(df, vio_col, group_col=gc, title=f"Violin: {vio_col}")
            st.plotly_chart(fig, use_container_width=True)

# ── Categorical charts ────────────────────────────────────────────────────────
with tab_cat:
    if not cat_cols:
        st.info("No categorical columns available.")
    else:
        st.markdown("### Categorical Column Charts")
        selected_cat = st.selectbox("Select categorical column:", cat_cols, key="cat_chart_col")
        if selected_cat:
            cat_stats = profile.get("statistics", {}).get("categorical", {}).get(selected_cat, {})
            top_cats = cat_stats.get("top_categories", {})
            
            if not top_cats:
                top_cats = dict(df[selected_cat].value_counts().head(15).items())

            n_unique = len(top_cats)
            c1, c2 = st.columns(2)
            
            with c1:
                fig = make_bar_chart(
                    labels=list(top_cats.keys()),
                    values=list(top_cats.values()),
                    title=f"Frequency: {selected_cat}",
                    horizontal=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                if n_unique <= 8:
                    fig = make_pie(
                        list(top_cats.keys()),
                        list(top_cats.values()),
                        title=f"Pie Chart: {selected_cat}",
                    )
                else:
                    fig = make_treemap(
                        list(top_cats.keys()),
                        list(top_cats.values()),
                        title=f"Treemap: {selected_cat}",
                    )
                st.plotly_chart(fig, use_container_width=True)

# ── Custom Scatter ────────────────────────────────────────────────────────────
with tab_custom:
    st.markdown("### Custom Scatter Plot")
    from utils.chart_factory import make_scatter
    
    if len(num_cols) < 2:
        st.info("Need at least 2 numerical columns for scatter plots.")
    else:
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis:", num_cols, key="sc_x")
        y_col = c2.selectbox("Y axis:", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        color_opts = ["(none)"] + cat_cols
        color_col = c3.selectbox("Color by:", color_opts, key="sc_c")
        gc = None if color_col == "(none)" else color_col
        
        if x_col and y_col and x_col != y_col:
            fig = make_scatter(df, x_col, y_col, color_col=gc, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        from modules.relationships import num_vs_num
        if x_col and y_col and x_col != y_col:
            result = num_vs_num(df, x_col, y_col)
            if result:
                interp = result.get("interpretation", "")
                box_class = "success-box" if result.get("significant") else "info-box"
                st.markdown(f'<div class="{box_class}">{interp}</div>', unsafe_allow_html=True)
