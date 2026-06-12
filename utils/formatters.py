"""utils/formatters.py — Number formatting, color maps, display helpers."""
from __future__ import annotations
import numpy as np


# ── Color palettes ────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "low":      "#059669",
    "medium":   "#D97706",
    "high":     "#EA580C",
    "critical": "#DC2626",
}

DTYPE_COLORS = {
    "numerical":   "#2563EB",
    "categorical": "#7C3AED",
    "boolean":     "#059669",
    "datetime":    "#DB2777",
    "text":        "#D97706",
    "identifier":  "#475569",
    "mixed":       "#DC2626",
}

GRADE_COLORS = {
    "A": "#059669",
    "B": "#2563EB",
    "C": "#D97706",
    "D": "#DC2626",
}

CATEGORICAL_PALETTE = [
    "#2563EB", "#0D9488", "#4F46E5", "#0891B2", "#059669",
    "#D97706", "#7C3AED", "#DB2777", "#DC2626", "#475569",
]


def fmt_number(value: float | int, precision: int = 2) -> str:
    """Format a number with commas and specified decimal places."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.{precision}f}"


def fmt_pct(value: float, precision: int = 1) -> str:
    """Format a fraction (0–1) as a percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.{precision}f}%"


def fmt_bytes(bytes_val: float) -> str:
    """Human-readable memory size string."""
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / 1024 ** 2:.2f} MB"
    return f"{bytes_val / 1024 ** 3:.2f} GB"


def severity_badge(severity: str) -> str:
    """Return an HTML badge for a severity level."""
    return f'<span class="severity-{severity}">{severity.upper()}</span>'


def health_grade_html(grade: str) -> str:
    """Return an HTML badge for a health grade."""
    return f'<span class="health-badge grade-{grade}">{grade}</span>'


def truncate_str(s: str, max_len: int = 40) -> str:
    """Truncate a string with ellipsis if it exceeds max_len."""
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def color_for_dtype(dtype_class: str) -> str:
    return DTYPE_COLORS.get(dtype_class, "#94A3B8")


def color_for_severity(severity: str) -> str:
    return SEVERITY_COLORS.get(severity, "#94A3B8")


def classify_imbalance(ratio: float | None) -> str:
    if ratio is None:
        return "N/A"
    if ratio < 1.5:
        return "balanced"
    elif ratio < 3:
        return "mild imbalance"
    elif ratio < 10:
        return "moderate imbalance"
    return "severe imbalance"
