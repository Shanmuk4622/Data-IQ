"""pages/02_🔍_Data_Quality.py — Missing values, duplicates, consistency."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils.chart_factory import make_null_bar, make_missing_heatmap, make_bar_chart


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🔍 Data Quality")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

quality = profile.get("quality", {})
missing = quality.get("missing_values", {})
dupes = quality.get("duplicate_rows", {})
dup_cols = quality.get("duplicate_cols", [])
consistency = quality.get("consistency_issues", {})

# ── Summary Metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
total_missing_cells = sum(v["count"] for v in missing.values())
sev_critical = sum(1 for v in missing.values() if v["severity"] == "critical")
sev_high = sum(1 for v in missing.values() if v["severity"] == "high")

for col, label, val, sub in [
    (c1, "Missing Columns", str(len(missing)), f"{total_missing_cells:,} cells total"),
    (c2, "Duplicate Rows",  str(dupes.get("count", 0)), f"{dupes.get('pct', 0)*100:.2f}%"),
    (c3, "Critical Missing", str(sev_critical), "≥80% null"),
    (c4, "Consistency Issues", str(len(consistency)), "columns affected"),
]:
    col.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ── Missing Values ────────────────────────────────────────────────────────────
st.markdown("## 🕳 Missing Values")

if not missing:
    st.markdown('<div class="success-box">✅ No missing values detected — excellent data completeness!</div>', unsafe_allow_html=True)
else:
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🗺 Heatmap", "📋 Table"])
    
    with tab1:
        fig = make_null_bar(missing, title="Missing Values by Column (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.caption("Red = missing, Dark = present. Sampled to 200 rows for display.")
        fig = make_missing_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        rows = []
        for col, v in sorted(missing.items(), key=lambda x: x[1]["pct"], reverse=True):
            severity = v["severity"]
            severity_html = f'<span class="severity-{severity}">{severity.upper()}</span>'
            rows.append({
                "Column": col,
                "Missing Count": v["count"],
                "Missing %": f"{v['pct']*100:.2f}%",
                "Severity": severity.upper(),
                "Recommendation": {
                    "low": "Impute with mean/mode",
                    "medium": "Impute with median/mode or use predictive imputation",
                    "high": "Consider dropping or KNN imputation",
                    "critical": "Drop column — insufficient data",
                }.get(severity, "Review manually"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Row distribution of missingness ──────────────────────────────────────────
if missing:
    st.markdown("#### Missing Values Per Row Distribution")
    row_nulls = df.isnull().sum(axis=1)
    fig = make_bar_chart(
        labels=list(range(int(row_nulls.max()) + 1)),
        values=[(row_nulls == i).sum() for i in range(int(row_nulls.max()) + 1)],
        title="Number of Missing Values per Row",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Duplicate Rows ────────────────────────────────────────────────────────────
st.markdown("## 🗃 Duplicate Rows")
dup_count = dupes.get("count", 0)
if dup_count == 0:
    st.markdown('<div class="success-box">✅ No exact duplicate rows found.</div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="warning-box">⚠️ Found <strong>{dup_count:,}</strong> duplicate rows '
        f'({dupes.get("pct", 0)*100:.2f}% of dataset).<br>'
        f'Recommendation: Remove with <code>df.drop_duplicates()</code></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Preview duplicate rows"):
        st.dataframe(df[df.duplicated()].head(20), use_container_width=True)

if dup_cols:
    st.markdown(f"**Duplicate column pairs:** {len(dup_cols)}")
    for pair in dup_cols[:5]:
        st.markdown(f"- `{pair[0]}` is identical to `{pair[1]}`")

st.divider()

# ── Consistency Issues ────────────────────────────────────────────────────────
st.markdown("## ⚖️ Consistency Issues")
if not consistency:
    st.markdown('<div class="success-box">✅ No consistency issues detected.</div>', unsafe_allow_html=True)
else:
    for col, issues in consistency.items():
        with st.expander(f"🔴 **{col}** — {len(issues)} issue(s)"):
            for issue in issues:
                issue_type = issue.get("issue_type", "")
                detail = issue.get("detail", "")
                examples = issue.get("examples", [])
                
                if issue_type == "case_inconsistency":
                    st.markdown(f"**Case Inconsistency:** {detail}")
                    if examples:
                        st.markdown("Examples of case variants:")
                        for ex in examples[:3]:
                            st.code(str(ex))
                    st.code(f"# Fix:\ndf['{col}'] = df['{col}'].str.lower().str.strip()")
                
                elif issue_type == "mixed_types":
                    st.markdown(f"**Mixed Types:** {detail}")
                    st.code(
                        f"# Inspect mixed values:\ndf['{col}'].apply(lambda x: type(x).__name__).value_counts()"
                    )
                
                elif issue_type == "symbol_contamination":
                    st.markdown(f"**Symbol Contamination:** {detail}")
                    if examples:
                        st.code(str(examples[:5]))
                    st.code(
                        f"# Fix (strip currency symbols):\ndf['{col}'] = pd.to_numeric("
                        f"df['{col}'].str.replace(r'[\\$€£%,]', '', regex=True), errors='coerce')"
                    )
