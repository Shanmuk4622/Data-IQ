"""modules/outliers.py — IQR, Z-score, and Isolation Forest outlier detection."""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def detect_iqr_outliers(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    if len(clean) < 4:
        return {}
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (clean < lower) | (clean > upper)
    return {
        "count": int(mask.sum()),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "examples": clean[mask].head(5).tolist(),
    }


def detect_zscore_outliers(series: pd.Series, threshold: float = 3.0) -> dict:
    clean = series.dropna().astype(float)
    if len(clean) < 4:
        return {}
    try:
        z_scores = np.abs(scipy_stats.zscore(clean))
        count = int((z_scores > threshold).sum())
        return {"count": count, "threshold": threshold}
    except Exception:
        return {}


def detect_isolation_forest(series: pd.Series, contamination: float = 0.05) -> dict:
    """Run Isolation Forest — only for series with >= 500 values."""
    clean = series.dropna().astype(float)
    if len(clean) < 500:
        return {}
    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = model.fit_predict(clean.values.reshape(-1, 1))
        count = int((preds == -1).sum())
        return {"count": count, "contamination": contamination}
    except Exception:
        return {}


def detect_outliers(df: pd.DataFrame, columns_profile: dict) -> dict:
    """Run all three outlier methods on all numerical columns."""
    result = {}
    num_cols = [
        col for col, info in columns_profile.items()
        if info.get("dtype_class") == "numerical"
    ]

    for col in num_cols:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            iqr_res = detect_iqr_outliers(series)
            zscore_res = detect_zscore_outliers(series)
            iso_res = detect_isolation_forest(series)

            n = len(series.dropna())
            iqr_count = iqr_res.get("count", 0)
            outlier_pct = iqr_count / n if n > 0 else 0.0

            result[col] = {
                "iqr_outliers": iqr_count,
                "zscore_outliers": zscore_res.get("count", 0),
                "isolation_forest_outliers": iso_res.get("count", None),
                "outlier_pct": round(outlier_pct, 4),
                "bounds_iqr": {
                    "lower": iqr_res.get("lower", None),
                    "upper": iqr_res.get("upper", None),
                },
                "examples": iqr_res.get("examples", []),
            }
        except Exception:
            pass

    return result
