"""modules/categorical.py — Frequency analysis and class imbalance detection."""
from __future__ import annotations
import pandas as pd
import numpy as np
from utils.formatters import classify_imbalance


def analyze_categorical(series: pd.Series) -> dict:
    """Full frequency analysis for a categorical column."""
    try:
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        counts = clean.value_counts()
        total = len(clean)
        rare_threshold = 0.01

        dominant_pct = float(counts.iloc[0] / total)
        dominant_cat = str(counts.index[0]) if dominant_pct > 0.8 else None

        imbalance_ratio = (
            float(counts.max() / counts.min())
            if len(counts) > 1 and counts.min() > 0
            else None
        )

        return {
            "unique_count": int(clean.nunique()),
            "top_categories": {str(k): int(v) for k, v in counts.head(15).items()},
            "rare_categories": [
                str(v) for v in counts[counts / total < rare_threshold].index.tolist()
            ],
            "dominant_category": dominant_cat,
            "dominant_pct": round(dominant_pct, 4),
            "class_imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
            "imbalance_label": classify_imbalance(imbalance_ratio),
        }
    except Exception:
        return {}


def analyze_all_categorical(df: pd.DataFrame, columns_profile: dict) -> dict:
    """Run categorical analysis for all categorical/boolean columns."""
    result = {}
    for col, info in columns_profile.items():
        if info.get("dtype_class") in ("categorical", "boolean"):
            stats = analyze_categorical(df[col])
            if stats:
                result[col] = stats
    return result
