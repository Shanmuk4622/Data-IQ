"""modules/statistics.py — Statistical summaries and distribution analysis."""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def classify_distribution(skewness: float, kurtosis: float) -> str:
    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return "normal"
    elif skewness > 1:
        return "right_skewed"
    elif skewness < -1:
        return "left_skewed"
    elif abs(skewness) < 0.5 and kurtosis > 3:
        return "leptokurtic"
    elif abs(skewness) < 0.5 and kurtosis < -1:
        return "uniform"
    return "unknown"


def compute_numerical_stats(series: pd.Series) -> dict:
    """Full statistical summary for a numerical series."""
    try:
        clean = series.dropna().astype(float)
        if len(clean) < 2:
            return {}

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        mean_val = float(clean.mean())
        skewness = float(scipy_stats.skew(clean))
        kurtosis = float(scipy_stats.kurtosis(clean))

        mode_result = clean.mode()
        mode_val = float(mode_result.iloc[0]) if len(mode_result) > 0 else None

        return {
            "count": int(len(clean)),
            "mean": round(mean_val, 4),
            "median": round(float(clean.median()), 4),
            "mode": round(mode_val, 4) if mode_val is not None else None,
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "range": round(float(clean.max() - clean.min()), 4),
            "variance": round(float(clean.var()), 4),
            "std": round(float(clean.std()), 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "cv": round(float(clean.std()) / mean_val, 4) if mean_val != 0 else None,
            "distribution_shape": classify_distribution(skewness, kurtosis),
        }
    except Exception:
        return {}


def compute_categorical_stats(series: pd.Series) -> dict:
    """Frequency and balance analysis for a categorical series."""
    try:
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        counts = clean.value_counts()
        total = len(clean)
        rare_threshold = 0.01

        dominant_cat = None
        dominant_pct = float(counts.iloc[0] / total) if len(counts) > 0 else 0.0
        if dominant_pct > 0.8:
            dominant_cat = str(counts.index[0])

        imbalance_ratio = (
            float(counts.max() / counts.min()) if len(counts) > 1 and counts.min() > 0 else None
        )

        return {
            "unique_count": int(clean.nunique()),
            "top_categories": {str(k): int(v) for k, v in counts.head(10).items()},
            "rare_categories": [str(v) for v in counts[counts / total < rare_threshold].index.tolist()],
            "dominant_category": dominant_cat,
            "dominant_pct": round(dominant_pct, 4),
            "class_imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
        }
    except Exception:
        return {}


def compute_statistics(df: pd.DataFrame, columns_profile: dict) -> dict:
    """Compute statistics for all numerical and categorical columns."""
    numerical = {}
    categorical = {}

    for col, info in columns_profile.items():
        dtype = info.get("dtype_class", "")
        if dtype in ("numerical", "boolean"):
            stats = compute_numerical_stats(df[col])
            if stats:
                numerical[col] = stats
        elif dtype in ("categorical", "text"):
            stats = compute_categorical_stats(df[col])
            if stats:
                categorical[col] = stats

    return {"numerical": numerical, "categorical": categorical}
