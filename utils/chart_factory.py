"""utils/chart_factory.py — All Plotly chart functions for DataIQ."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats


_TEMPLATE = "plotly_white"
_LAYOUT_DEFAULTS = dict(
    height=420,
    margin=dict(l=30, r=20, t=50, b=30),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0F172A"),
)

CAT_COLORS = [
    "#2563EB", "#0D9488", "#4F46E5", "#0891B2", "#059669",
    "#D97706", "#7C3AED", "#DB2777", "#DC2626", "#475569",
]


def _apply_defaults(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    d = {**_LAYOUT_DEFAULTS, "height": height, "title": dict(text=title, font=dict(size=14, color="#0F172A"))}
    fig.update_layout(**d)
    return fig


def _error_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart unavailable: {msg}",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color="#DC2626"),
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
            marker_color="#2563EB", opacity=0.7,
            histnorm="probability density",
        ))
        # KDE
        kde_x = np.linspace(clean.min(), clean.max(), 200)
        kde = scipy_stats.gaussian_kde(clean)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde(kde_x),
            mode="lines", name="KDE",
            line=dict(color="#059669", width=2),
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
                fillcolor="#2563EB", line_color="#2563EB", opacity=0.7,
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
            colorscale=[[0, "#DC2626"], [0.5, "#F8FAFC"], [1, "#2563EB"]],
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
        colors = color if isinstance(color, list) else [color or "#2563EB"] * len(labels)
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
                template="plotly_white",
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="markers",
                marker=dict(color="#2563EB", opacity=0.6, size=5),
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
            ))
            # OLS trend line
            try:
                m, b = np.polyfit(sub[x].astype(float), sub[y].astype(float), 1)
                xr = np.linspace(sub[x].min(), sub[x].max(), 100)
                fig.add_trace(go.Scatter(
                    x=xr, y=m * xr + b, mode="lines",
                    line=dict(color="#059669", width=1.5, dash="dash"),
                    name="OLS trend",
                ))
                r = np.corrcoef(sub[x].astype(float), sub[y].astype(float))[0, 1]
                fig.add_annotation(
                    text=f"r = {r:.3f}", xref="paper", yref="paper",
                    x=0.02, y=0.95, showarrow=False,
                    font=dict(color="#475569", size=11),
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
            line=dict(color="#2563EB", width=1.5),
            marker=dict(size=3, color="#0891B2"),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
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
                marker=dict(color="#2563EB", size=3, opacity=0.5),
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
            colorscale=[[0, "#F8FAFC"], [1, "#DC2626"]],
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
            color = "#059669"
        elif score >= 70:
            color = "#2563EB"
        elif score >= 55:
            color = "#D97706"
        else:
            color = "#DC2626"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 70, "valueformat": ".1f"},
            number={"suffix": "", "font": {"size": 42, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(241,245,249,1)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 55],  "color": "rgba(220,38,38,0.08)"},
                    {"range": [55, 70], "color": "rgba(217,119,6,0.08)"},
                    {"range": [70, 85], "color": "rgba(37,99,235,0.08)"},
                    {"range": [85, 100],"color": "rgba(5,150,105,0.08)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
            title={"text": f"Dataset Health Score — Grade {grade}", "font": {"size": 14, "color": "#0F172A"}},
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#0F172A"),
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
            fill="toself", fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="#2563EB", width=2),
            marker=dict(color="#0D9488", size=6),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
                bgcolor="rgba(248,250,252,0.8)",
            ),
            showlegend=False,
            height=380,
            margin=dict(l=40, r=40, t=50, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#0F172A"),
            title=dict(text=title, font=dict(size=14, color="#0F172A")),
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
            marker=dict(color="#2563EB", size=4, opacity=0.7),
            name="Data",
        ))
        line_x = np.array([quantiles.min(), quantiles.max()])
        fig.add_trace(go.Scatter(
            x=line_x, y=slope * line_x + intercept,
            mode="lines", name="Normal",
            line=dict(color="#059669", width=2, dash="dash"),
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
            marker=dict(colorscale="Blues"),
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
            marker=dict(colors=CAT_COLORS, line=dict(color="#FFFFFF", width=1.5)),
            hole=0.35,
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        _apply_defaults(fig, title, height=380)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  ML result charts (used by pages/10_🛠_Make_ML_Model.py)
# ══════════════════════════════════════════════════════════════════════════════

# ── Confusion matrix ──────────────────────────────────────────────────────────
def make_confusion_matrix(cm, labels: list, title: str = "Confusion Matrix") -> go.Figure:
    try:
        z = np.asarray(cm)
        labels = [str(l) for l in labels]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            colorscale=[[0, "#F8FAFC"], [1, "#2563EB"]],
            text=z, texttemplate="%{text}", textfont=dict(size=13),
            hovertemplate="Predicted <b>%{x}</b><br>Actual <b>%{y}</b><br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        fig.update_xaxes(title_text="Predicted")
        fig.update_yaxes(title_text="Actual", autorange="reversed")
        _apply_defaults(fig, title, height=max(340, len(labels) * 48))
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── ROC curve ─────────────────────────────────────────────────────────────────
def make_roc_curve(y_true, y_score, auc: float = None, title: str = "ROC Curve") -> go.Figure:
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#2563EB", width=2.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            name=f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#94A3B8", width=1.5, dash="dash"), name="Chance",
        ))
        fig.update_xaxes(title_text="False Positive Rate", range=[0, 1])
        fig.update_yaxes(title_text="True Positive Rate", range=[0, 1.02])
        _apply_defaults(fig, title)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Predicted vs actual ───────────────────────────────────────────────────────
def make_pred_vs_actual(y_true, y_pred, title: str = "Predicted vs Actual") -> go.Figure:
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker=dict(color="#2563EB", opacity=0.55, size=6),
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
            name="Predictions",
        ))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#059669", width=1.5, dash="dash"), name="Perfect fit",
        ))
        fig.update_xaxes(title_text="Actual")
        fig.update_yaxes(title_text="Predicted")
        _apply_defaults(fig, title)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Residual plot ─────────────────────────────────────────────────────────────
def make_residual_plot(y_true, y_pred, title: str = "Residuals") -> go.Figure:
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers",
            marker=dict(color="#7C3AED", opacity=0.55, size=6),
            hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color="#DC2626", width=1.5, dash="dash"))
        fig.update_xaxes(title_text="Predicted")
        fig.update_yaxes(title_text="Residual (actual − predicted)")
        _apply_defaults(fig, title)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Cluster scatter (2-D PCA projection) ──────────────────────────────────────
def make_cluster_scatter(pca_x, pca_y, labels, title: str = "Clusters (PCA projection)") -> go.Figure:
    try:
        df = pd.DataFrame({"x": pca_x, "y": pca_y, "cluster": [str(l) for l in labels]})
        fig = px.scatter(
            df, x="x", y="y", color="cluster",
            color_discrete_sequence=CAT_COLORS, template="plotly_white",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_xaxes(title_text="PC 1")
        fig.update_yaxes(title_text="PC 2")
        _apply_defaults(fig, title, height=460)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Elbow plot ────────────────────────────────────────────────────────────────
def make_elbow_plot(ks: list, inertias: list, title: str = "Elbow Method (K-Means)") -> go.Figure:
    try:
        fig = go.Figure(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            marker=dict(color="#0891B2", size=8),
            hovertemplate="k = %{x}<br>Inertia: %{y:.0f}<extra></extra>",
        ))
        fig.update_xaxes(title_text="Number of clusters (k)", dtick=1)
        fig.update_yaxes(title_text="Inertia (within-cluster SSE)")
        _apply_defaults(fig, title)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Forecast plot (history + forecast + confidence band) ──────────────────────
def make_forecast_plot(
    history_x, history_y, forecast_x, forecast_y,
    lower=None, upper=None, title: str = "Forecast",
) -> go.Figure:
    try:
        fig = go.Figure()
        # Confidence band
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=list(forecast_x) + list(forecast_x)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself", fillcolor="rgba(37,99,235,0.12)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="95% interval",
            ))
        fig.add_trace(go.Scatter(
            x=history_x, y=history_y, mode="lines",
            line=dict(color="#475569", width=1.5), name="History",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=forecast_y, mode="lines",
            line=dict(color="#2563EB", width=2), name="Forecast",
        ))
        fig.update_yaxes(title_text="Value")
        _apply_defaults(fig, title, height=440)
        return fig
    except Exception as e:
        return _error_fig(str(e))


# ── Cross-validation fold scores ──────────────────────────────────────────────
def make_cv_scores(scores: list, scoring: str = "score", title: str = "Cross-Validation Scores") -> go.Figure:
    try:
        scores = list(scores)
        folds = [f"Fold {i+1}" for i in range(len(scores))]
        mean = float(np.mean(scores)) if scores else 0.0
        fig = go.Figure(go.Bar(
            x=folds, y=scores,
            marker_color="#2563EB",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ))
        fig.add_hline(
            y=mean, line=dict(color="#059669", width=2, dash="dash"),
            annotation_text=f"mean = {mean:.4f}", annotation_position="top left",
        )
        fig.update_yaxes(title_text=scoring)
        _apply_defaults(fig, title, height=340)
        return fig
    except Exception as e:
        return _error_fig(str(e))
