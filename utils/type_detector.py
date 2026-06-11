"""utils/type_detector.py — Smart column type classification for DataIQ."""
from __future__ import annotations
import pandas as pd
import numpy as np
import re


# ── Priority-ordered type labels ─────────────────────────────────────────────
DTYPE_CLASSES = [
    "boolean", "identifier", "datetime", "numerical",
    "text", "categorical", "mixed",
]

BOOL_PAIRS = [
    {0, 1}, {"true", "false"}, {"yes", "no"}, {"y", "n"},
    {"t", "f"}, {"male", "female"}, {"m", "f"},
]


def classify_column(series: pd.Series, col_name: str = "") -> str:
    """Return one of DTYPE_CLASSES for the given series."""
    clean = series.dropna()
    if len(clean) == 0:
        return "categorical"

    n_unique = clean.nunique()
    n_total = len(clean)
    unique_pct = n_unique / n_total

    # ── 1. Boolean ───────────────────────────────────────────────────────────
    if n_unique == 2:
        vals_lower = set(str(v).lower().strip() for v in clean.unique())
        if any(vals_lower == s for s in BOOL_PAIRS):
            return "boolean"
        # numeric 0/1
        if pd.api.types.is_numeric_dtype(clean) and set(clean.unique()).issubset({0, 1}):
            return "boolean"

    # ── 2. Identifier ────────────────────────────────────────────────────────
    if unique_pct > 0.95 and (
        pd.api.types.is_integer_dtype(clean) or pd.api.types.is_object_dtype(clean)
    ):
        return "identifier"

    # ── 3. Datetime ──────────────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(clean):
        return "datetime"
    if pd.api.types.is_object_dtype(clean):
        parsed = pd.to_datetime(clean, infer_datetime_format=True, errors="coerce")
        if parsed.notna().mean() > 0.80:
            return "datetime"

    # ── 4. Numerical ─────────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(clean):
        return "numerical"

    # ── 5. Text ──────────────────────────────────────────────────────────────
    if pd.api.types.is_object_dtype(clean):
        avg_len = clean.astype(str).str.len().mean()
        if avg_len > 50:
            return "text"

        # ── 7. Mixed ─────────────────────────────────────────────────────────
        numeric_ok = pd.to_numeric(clean, errors="coerce").notna().mean()
        if 0.05 < numeric_ok < 0.95:
            return "mixed"

        # ── 6. Categorical ───────────────────────────────────────────────────
        if unique_pct < 0.50:
            return "categorical"

        return "identifier"  # high cardinality object that doesn't fit above

    return "categorical"


def get_all_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return {col_name: dtype_class} for all columns."""
    return {col: classify_column(df[col], col) for col in df.columns}
