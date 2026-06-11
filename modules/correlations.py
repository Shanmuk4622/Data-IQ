"""modules/correlations.py — Pearson, Spearman, Kendall correlations + high-corr detection."""
from __future__ import annotations
import pandas as pd
import numpy as np


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlation matrices for all numerical columns."""
    try:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 3:
            return {
                "pearson": {}, "spearman": {}, "kendall": {},
                "high_correlation_pairs": [],
                "insufficient_columns": True,
            }

        # Drop constant columns
        num_df = num_df.loc[:, num_df.std() > 0]

        pearson = num_df.corr(method="pearson")
        spearman = num_df.corr(method="spearman")
        kendall = num_df.corr(method="kendall")

        high_pairs = get_high_correlation_pairs(pearson)

        return {
            "pearson": pearson.round(4).to_dict(),
            "spearman": spearman.round(4).to_dict(),
            "kendall": kendall.round(4).to_dict(),
            "high_correlation_pairs": high_pairs,
            "insufficient_columns": False,
        }
    except Exception as e:
        return {
            "pearson": {}, "spearman": {}, "kendall": {},
            "high_correlation_pairs": [],
            "error": str(e),
        }


def get_high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
    """Flag pairs where |pearson_r| > threshold (default 0.8)."""
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if pd.isna(r):
                continue
            if abs(r) > threshold:
                if abs(r) > 0.95:
                    rec = "Consider dropping one — near-perfect multicollinearity"
                else:
                    rec = "Investigate relationship — may indicate redundancy"
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "pearson_r": round(float(r), 4),
                    "recommendation": rec,
                })
    return sorted(pairs, key=lambda x: abs(x["pearson_r"]), reverse=True)


def get_feature_importance_order(corr_dict: dict, target: str) -> list[tuple[str, float]]:
    """Return columns sorted by |correlation with target|."""
    if not corr_dict or target not in corr_dict:
        return []
    target_corrs = corr_dict[target]
    pairs = [
        (col, abs(val))
        for col, val in target_corrs.items()
        if col != target and val is not None and not pd.isna(val)
    ]
    return sorted(pairs, key=lambda x: x[1], reverse=True)
