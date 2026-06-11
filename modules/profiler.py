"""modules/profiler.py — Column profiling, type detection, target candidate scoring."""
from __future__ import annotations
import pandas as pd
import numpy as np
from utils.type_detector import classify_column


TARGET_KEYWORDS = {
    "target", "label", "output", "class", "result",
    "outcome", "predict", "flag", "status", "y",
    "churn", "default", "fraud", "survived", "price", "value",
}


def _target_candidate_score(col: str, series: pd.Series, dtype_class: str, profile_row: dict) -> float:
    """Compute 0.0–1.0 target candidate score for a column."""
    score = 0.0
    col_lower = col.lower()

    # Name-based keywords
    if any(kw in col_lower for kw in TARGET_KEYWORDS):
        score += 0.3

    # Boolean: often a binary target
    if dtype_class == "boolean":
        score += 0.2

    # Low-cardinality categorical: likely a class label
    if dtype_class == "categorical" and profile_row.get("unique_count", 999) <= 10:
        score += 0.2

    # Numerical with moderate skew: regression target candidate
    if dtype_class == "numerical":
        try:
            from scipy.stats import skew as scipy_skew
            sk = scipy_skew(series.dropna().astype(float))
            if abs(sk) <= 2:
                score += 0.15
        except Exception:
            score += 0.05

    # Last column bonus
    if profile_row.get("_is_last_col", False):
        score += 0.15

    # Penalty: identifier columns are not targets
    if dtype_class == "identifier":
        score -= 0.5

    return float(np.clip(score, 0.0, 1.0))


def profile_columns(df: pd.DataFrame) -> dict:
    """
    Build the full columns profile dict.
    Returns {col_name: {...}} with all column-level statistics.
    """
    n_rows = len(df)
    last_col = df.columns[-1]
    result = {}

    for col in df.columns:
        series = df[col]
        clean = series.dropna()

        null_count = int(series.isna().sum())
        non_null = int(len(clean))
        unique_count = int(clean.nunique())

        null_pct = null_count / n_rows if n_rows > 0 else 0.0
        unique_pct = unique_count / non_null if non_null > 0 else 0.0

        dtype_class = classify_column(series, col)

        profile_row = {
            "dtype_raw": str(series.dtype),
            "dtype_class": dtype_class,
            "non_null_count": non_null,
            "null_count": null_count,
            "null_pct": round(null_pct, 4),
            "unique_count": unique_count,
            "unique_pct": round(unique_pct, 4),
            "_is_last_col": col == last_col,
            "is_target_candidate": False,
            "target_candidate_score": 0.0,
        }

        score = _target_candidate_score(col, series, dtype_class, profile_row)
        profile_row["target_candidate_score"] = round(score, 3)
        profile_row["is_target_candidate"] = score >= 0.3
        # Remove internal flag
        del profile_row["_is_last_col"]

        result[col] = profile_row

    return result


def get_top_target_candidates(columns_profile: dict, top_n: int = 3) -> list[dict]:
    """Return top N target column candidates sorted by score."""
    scored = [
        {"col": col, "score": v["target_candidate_score"], "dtype": v["dtype_class"]}
        for col, v in columns_profile.items()
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]


def get_dtype_summary(columns_profile: dict) -> dict[str, int]:
    """Return count of each dtype class."""
    from collections import Counter
    return dict(Counter(v["dtype_class"] for v in columns_profile.values()))
