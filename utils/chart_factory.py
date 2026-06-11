"""utils/chart_factory.py — All Plotly chart functions for DataIQ."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats


_TEMPLATE = "plotly_dark"
_LAYOUT_DEFAULTS = dict(
    height=420,
    margin=dict(l=30, r=20, t=50, b=30),
    paper_bgcolor="rgba(15,15,26,0.6)",
    plot_bgcolor="rgba(26,26,46,0.4)",
    font=dict(family="Inter, sans-serif", color="#E2E8F0"),
)

CAT_COLORS = [
    "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#F472B6",
    "#60A5FA", "#34D399", "#FBBF24", "#A78BFA", "#F87171",
]


def _apply_defaults(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    d = {**_LAYOUT_DEFAULTS, "height": height, "title": dict(text=title, font=dict(size=14, color="#C4B5FD"))}
    fig.update_layout(**d)
    return fig


def _error_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart unavailable: {msg}",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color="#F87171"),
    )
    _apply_defaults(fig)
    return fig


# ── Histogram with KDE ────────────────────────────────────────────────────────
def make_histogram(series: pd.Series, title: str = "") -> go.Figure:
    try:
        clean = series.dropna().astype(float)
        if len(clean) < 2:
            return _error_fig("Not enough data")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=clean, name="Count",
            marker_color="#8B5CF6", opacity=0.7,
            histnorm="probability density",
        ))
        # KDE
        kde_x = np.linspace(clean.min(), clean.max(), 200)
        kde = scipy_stats.gaussian_kde(clean)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde(kde_x),
            mode="lines", name="KDE",
            line=dict(color="#10B981", width=2),
        ))
        return _apply_defaults(fig, title or f"Distribution: {series.name}")
    except Exception as e:
        return _error_fig(str(e))


# ── Boxplot ───────────────────────────────────────────────────────────────────
def make_boxplot(df: pd.DataFrame, columns: list, title: str = "") -> go.Figure:
    try:
        fig = go.Figure()
        for i, col in enumerate(columns):
            clean = df[col].dropna()
            fig.add_trace(go.Box(
                y=clean, name=col,
                marker_color=CAT_COLORS[i % len(CAT_COLORS)],
                boxmean="sd",
                # Show individual points for small datasets
                boxpoints="outliers",
            ))
        return _apply_defaults(fig, title or "Box Plot", height=450)
    except Exception as e:
        return _error_fig(str(e))


# ── Violin plot ───────────────────────────────────────────────────────────────
def make_violin(df: pd.DataFrame, col: str, group_col: str = None, title: str = "") -> go.Figure:
    try:
        if group_col and group_col in df.columns:
            groups = df[group_col].dropna().unique()[:8]
            fig = go.Figure()
            for i, g in enumerate(groups):
                sub = df[df[group_col] == g][col].dropna()
                fig.add_trace(go.Violin(
                    y=sub, name=str(g),
                    box_visible=True, meanline_visible=True,
                    fillcolor=CAT_COLORS[i % len(CAT_COLORS)],
                    line_color=CAT_COLORS[i % len(CAT_COLORS)],
                    opacity=0.7,
                ))
        else:
            clean = df[col].dropna()
            fig = go.Figure(go.Violin(
                y=clean, box_visible=True, meanline_visible=True,
                fillcolor="#8B5CF6", line_color="#8B5CF6", opacity=0.7,
            ))
        return _apply_defaults(fig, title or f"Violin: {col}")
    except Exception as e:
        return _error_fig(str(e))


# ── Correlation heatmap ───────────────────────────────────────────────────────
def make_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    try:
        z = corr_matrix.values
        cols = corr_matrix.columns.tolist()
        fig = go.Figure(go.Heatmap(
            z=z, x=cols, y=cols,
            colorscale=[[0, "#EF4444"], [0.5, "#1A1A2E"], [1, "#8B5CF6"]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(z, 2), texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        _apply_defaults(fig, title, height=max(400, len(cols) * 40))
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Bar chart ─────────────────────────────────────────────────────────────────
def make_bar_chart(
    labels: list, values: list,
    title: str = "", horizontal: bool = False,
    color: str | list = None,
) -> go.Figure:
    try:
        colors = color if isinstance(color, list) else [color or "#8B5CF6"] * len(labels)
        if horizontal:
            fig = go.Figure(go.Bar(
                y=labels, x=values, orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:,}<extra></extra>",
            ))
        else:
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=colors,
                hovertemplate="%{x}: %{y:,}<extra></extra>",
            ))
        return _apply_defaults(fig, title)
    except Exception as e:
        return _error_fig(str(e))


# ── Scatter with trend ────────────────────────────────────────────────────────
def make_scatter(df: pd.DataFrame, x: str, y: str, color_col: str = None, title: str = "") -> go.Figure:
    try:
        sub = df[[x, y] + ([color_col] if color_col else [])].dropna()
        if color_col:
            fig = px.scatter(
                sub, x=x, y=y, color=color_col,
                color_discrete_sequence=CAT_COLORS,
                template="plotly_dark",
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="markers",
                marker=dict(color="#8B5CF6", opacity=0.6, size=5),
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
            ))
            # OLS trend line
            try:
                m, b = np.polyfit(sub[x].astype(float), sub[y].astype(float), 1)
                xr = np.linspace(sub[x].min(), sub[x].max(), 100)
                fig.add_trace(go.Scatter(
                    x=xr, y=m * xr + b, mode="lines",
                    line=dict(color="#10B981", width=1.5, dash="dash"),
                    name="OLS trend",
                ))
                r = np.corrcoef(sub[x].astype(float), sub[y].astype(float))[0, 1]
                fig.add_annotation(
                    text=f"r = {r:.3f}", xref="paper", yref="paper",
                    x=0.02, y=0.95, showarrow=False,
                    font=dict(color="#C4B5FD", size=11),
                )
            except Exception:
                pass
        return _apply_defaults(fig, title or f"{x} vs {y}")
    except Exception as e:
        return _error_fig(str(e))


# ── Time series ───────────────────────────────────────────────────────────────
def make_time_series(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    try:
        sub = df[[x, y]].dropna().sort_values(x)
        fig = go.Figure(go.Scatter(
            x=sub[x], y=sub[y], mode="lines+markers",
            line=dict(color="#8B5CF6", width=1.5),
            marker=dict(size=3, color="#06B6D4"),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
        ))
        return _apply_defaults(fig, title or f"{y} over time")
    except Exception as e:
        return _error_fig(str(e))


# ── Pair plot (scatter matrix) ────────────────────────────────────────────────
def make_pair_plot(df: pd.DataFrame, columns: list, color_col: str = None, title: str = "") -> go.Figure:
    try:
        sub = df[columns + ([color_col] if color_col else [])].dropna()
        dims = [dict(label=c, values=sub[c]) for c in columns]
        if color_col:
            unique_vals = sub[color_col].astype("category").cat.codes
            fig = go.Figure(go.Splom(
                dimensions=dims,
                marker=dict(
                    color=unique_vals,
                    colorscale=[[i / max(1, len(CAT_COLORS) - 1), c]
                                for i, c in enumerate(CAT_COLORS[:len(unique_vals.unique())])],
                    size=3, opacity=0.6,
                ),
                showupperhalf=False,
            ))
        else:
            fig = go.Figure(go.Splom(
                dimensions=dims,
                marker=dict(color="#8B5CF6", size=3, opacity=0.5),
                showupperhalf=False,
            ))
        _apply_defaults(fig, title or "Pair Plot", height=600)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Missing value heatmap ─────────────────────────────────────────────────────
def make_missing_heatmap(df: pd.DataFrame, title: str = "Missing Value Pattern") -> go.Figure:
    try:
        sample = df if len(df) <= 200 else df.sample(200, random_state=42)
        z = sample.isnull().astype(int).T.values
        fig = go.Figure(go.Heatmap(
            z=z,
            x=[str(i) for i in range(len(sample))],
            y=df.columns.tolist(),
            colorscale=[[0, "#1A1A2E"], [1, "#EF4444"]],
            showscale=False,
            hovertemplate="Row %{x}, Col %{y}: %{z}<extra></extra>",
        ))
        _apply_defaults(fig, title, height=max(300, len(df.columns) * 20))
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Health score gauge ────────────────────────────────────────────────────────
def make_health_gauge(score: float, grade: str = "") -> go.Figure:
    try:
        if score >= 85:
            color = "#10B981"
        elif score >= 70:
            color = "#3B82F6"
        elif score >= 55:
            color = "#F59E0B"
        else:
            color = "#EF4444"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 70, "valueformat": ".1f"},
            number={"suffix": "", "font": {"size": 42, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(26,26,46,0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 55],  "color": "rgba(239,68,68,0.15)"},
                    {"range": [55, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 85], "color": "rgba(59,130,246,0.15)"},
                    {"range": [85, 100],"color": "rgba(16,185,129,0.15)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
            title={"text": f"Dataset Health Score — Grade {grade}", "font": {"size": 14, "color": "#C4B5FD"}},
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(15,15,26,0.0)",
            font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        )
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Radar chart ───────────────────────────────────────────────────────────────
def make_radar_chart(categories: list, values: list, title: str = "") -> go.Figure:
    try:
        cats = categories + [categories[0]]
        vals = values + [values[0]]
        fig = go.Figure(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself", fillcolor="rgba(139,92,246,0.25)",
            line=dict(color="#8B5CF6", width=2),
            marker=dict(color="#06B6D4", size=6),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
                bgcolor="rgba(26,26,46,0.4)",
            ),
            showlegend=False,
            height=380,
            margin=dict(l=40, r=40, t=50, b=30),
            paper_bgcolor="rgba(15,15,26,0.0)",
            font=dict(family="Inter, sans-serif", color="#E2E8F0"),
            title=dict(text=title, font=dict(size=14, color="#C4B5FD")),
        )
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── QQ Plot ───────────────────────────────────────────────────────────────────
def make_qq_plot(series: pd.Series, title: str = "") -> go.Figure:
    try:
        clean = series.dropna().astype(float)
        (quantiles, values), (slope, intercept, _) = scipy_stats.probplot(clean, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quantiles, y=values, mode="markers",
            marker=dict(color="#8B5CF6", size=4, opacity=0.7),
            name="Data",
        ))
        line_x = np.array([quantiles.min(), quantiles.max()])
        fig.add_trace(go.Scatter(
            x=line_x, y=slope * line_x + intercept,
            mode="lines", name="Normal",
            line=dict(color="#10B981", width=2, dash="dash"),
        ))
        return _apply_defaults(fig, title or f"Q-Q Plot: {series.name}")
    except Exception as e:
        return _error_fig(str(e))


# ── Null bar chart (sorted by pct) ────────────────────────────────────────────
def make_null_bar(missing_dict: dict, title: str = "Missing Values (%)") -> go.Figure:
    try:
        from utils.formatters import SEVERITY_COLORS
        sorted_items = sorted(missing_dict.items(), key=lambda x: x[1]["pct"], reverse=True)
        cols = [k for k, _ in sorted_items]
        pcts = [v["pct"] * 100 for _, v in sorted_items]
        colors = [SEVERITY_COLORS[v["severity"]] for _, v in sorted_items]
        fig = go.Figure(go.Bar(
            x=pcts, y=cols, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        return _apply_defaults(fig, title, height=max(300, len(cols) * 28))
    except Exception as e:
        return _error_fig(str(e))


# ── Treemap ───────────────────────────────────────────────────────────────────
def make_treemap(labels: list, values: list, title: str = "") -> go.Figure:
    try:
        parents = [""] * len(labels)
        fig = go.Figure(go.Treemap(
            labels=labels, values=values, parents=parents,
            marker=dict(colorscale="Purples"),
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        _apply_defaults(fig, title)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Pie chart ─────────────────────────────────────────────────────────────────
def make_pie(labels: list, values: list, title: str = "") -> go.Figure:
    try:
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=CAT_COLORS, line=dict(color="#0F0F1A", width=1.5)),
            hole=0.35,
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        _apply_defaults(fig, title, height=380)
        return fig
    except Exception as e:
        return _error_fig(str(e))
