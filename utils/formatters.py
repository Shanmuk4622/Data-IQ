"""utils/formatters.py — Number formatting, color maps, display helpers."""
from __future__ import annotations
import numpy as np


# ── Color palettes ────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "low":      "#10B981",
    "medium":   "#F59E0B",
    "high":     "#F97316",
    "critical": "#EF4444",
}

DTYPE_COLORS = {
    "numerical":   "#60A5FA",
    "categorical": "#A78BFA",
    "boolean":     "#34D399",
    "datetime":    "#F472B6",
    "text":        "#FBBF24",
    "identifier":  "#94A3B8",
    "mixed":       "#F87171",
}

GRADE_COLORS = {
    "A": "#10B981",
    "B": "#3B82F6",
    "C": "#F59E0B",
    "D": "#EF4444",
}

CATEGORICAL_PALETTE = [
    "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#F472B6",
    "#60A5FA", "#34D399", "#FBBF24", "#A78BFA", "#F87171",
    "#4ADE80", "#FB923C", "#E879F9", "#38BDF8", "#A3E635",
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
