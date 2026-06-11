"""modules/health_score.py — 0–100 dataset health scoring."""
from __future__ import annotations
import numpy as np


def compute_health_score(profile: dict) -> dict:
    """Compute weighted health score from 5 dimensions. Returns full breakdown."""
    columns_profile = profile.get("columns", {})
    quality = profile.get("quality", {})
    outlier_data = profile.get("outliers", {})
    corr_data = profile.get("correlations", {})

    # ── Missing Score ─────────────────────────────────────────────────────────
    null_pcts = [v.get("null_pct", 0) for v in columns_profile.values()]
    avg_null_pct = float(np.mean(null_pcts)) if null_pcts else 0.0
    missing_score = max(0.0, 100.0 - avg_null_pct * 100)

    # ── Outlier Score ─────────────────────────────────────────────────────────
    outlier_pcts = [v.get("outlier_pct", 0) for v in outlier_data.values()]
    avg_outlier_pct = float(np.mean(outlier_pcts)) if outlier_pcts else 0.0
    outlier_score = max(0.0, 100.0 - avg_outlier_pct * 200)

    # ── Duplicate Score ───────────────────────────────────────────────────────
    dup_pct = quality.get("duplicate_rows", {}).get("pct", 0.0)
    duplicate_score = max(0.0, 100.0 - float(dup_pct) * 300)

    # ── Class Balance Score ───────────────────────────────────────────────────
    cat_stats = profile.get("statistics", {}).get("categorical", {})
    target_col = profile.get("ml_readiness", {}).get("target_column")
    if target_col and target_col in cat_stats:
        ratio = cat_stats[target_col].get("class_imbalance_ratio")
        if ratio and ratio > 0:
            # Perfect balance = ratio of 1.0 → score 100
            # Severe imbalance = ratio > 10 → score ~0
            class_balance_score = max(0.0, 100.0 - (ratio - 1) * 11)
        else:
            class_balance_score = 100.0
    else:
        class_balance_score = 80.0  # neutral when no target / not classification

    # ── Correlation Score ─────────────────────────────────────────────────────
    n_high_corr = len(corr_data.get("high_correlation_pairs", []))
    correlation_score = max(0.0, 100.0 - n_high_corr * 10)

    # ── Weighted Total ────────────────────────────────────────────────────────
    total = (
        missing_score     * 0.30 +
        outlier_score     * 0.20 +
        duplicate_score   * 0.15 +
        class_balance_score * 0.20 +
        correlation_score * 0.15
    )

    grade = (
        "A" if total >= 85 else
        "B" if total >= 70 else
        "C" if total >= 55 else
        "D"
    )

    # ── One-line fix ──────────────────────────────────────────────────────────
    dimensions = {
        "Missing Values":   missing_score,
        "Outliers":         outlier_score,
        "Duplicates":       duplicate_score,
        "Class Balance":    class_balance_score,
        "Correlations":     correlation_score,
    }
    worst = min(dimensions, key=dimensions.get)
    fix_map = {
        "Missing Values":   "Impute or drop columns with high null rates",
        "Outliers":         "Apply IQR clipping or robust scaling to reduce outlier impact",
        "Duplicates":       "Remove exact duplicate rows with df.drop_duplicates()",
        "Class Balance":    "Apply SMOTE oversampling or adjust class weights in your model",
        "Correlations":     "Drop or combine highly correlated feature pairs",
    }

    return {
        "total": round(total, 1),
        "missing_score": round(missing_score, 1),
        "outlier_score": round(outlier_score, 1),
        "duplicate_score": round(duplicate_score, 1),
        "class_balance_score": round(class_balance_score, 1),
        "correlation_score": round(correlation_score, 1),
        "grade": grade,
        "worst_dimension": worst,
        "one_line_fix": fix_map[worst],
        "dimensions": dimensions,
    }
