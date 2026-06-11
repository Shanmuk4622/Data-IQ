"""modules/dataset_comparator.py — Two-dataset diff analysis."""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def compare_datasets(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    name_a: str = "Dataset A", name_b: str = "Dataset B",
) -> dict:
    """Compare two dataframes and return a comprehensive diff report."""
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    common_cols = cols_a & cols_b

    # Schema diff
    dtype_changes = {}
    for col in common_cols:
        if df_a[col].dtype != df_b[col].dtype:
            dtype_changes[col] = {
                "a": str(df_a[col].dtype),
                "b": str(df_b[col].dtype),
            }

    # Distribution shifts (KS test on numerical columns)
    distribution_shifts = {}
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            a_clean = df_a[col].dropna().astype(float)
            b_clean = df_b[col].dropna().astype(float)
            if len(a_clean) < 3 or len(b_clean) < 3:
                continue
            try:
                ks_stat, ks_pval = scipy_stats.ks_2samp(a_clean, b_clean)
                distribution_shifts[col] = {
                    "mean_a": round(float(a_clean.mean()), 4),
                    "mean_b": round(float(b_clean.mean()), 4),
                    "std_a": round(float(a_clean.std()), 4),
                    "std_b": round(float(b_clean.std()), 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pval), 4),
                    "shift_detected": ks_pval < 0.05,
                }
            except Exception:
                pass

    # Null differences
    null_diff = {}
    for col in common_cols:
        pct_a = float(df_a[col].isna().mean())
        pct_b = float(df_b[col].isna().mean())
        if abs(pct_a - pct_b) > 0.01:
            null_diff[col] = {
                "null_pct_a": round(pct_a, 4),
                "null_pct_b": round(pct_b, 4),
                "change": round(pct_b - pct_a, 4),
            }

    # Row overlap
    overlap_count = 0
    overlap_pct = 0.0
    if len(common_cols) > 0:
        try:
            merged = pd.merge(df_a, df_b, how="inner", on=list(common_cols)[:5])
            overlap_count = len(merged)
            overlap_pct = round(overlap_count / max(len(df_a), 1) * 100, 2)
        except Exception:
            pass

    # Shifted distributions count
    shifted_cols = [c for c, v in distribution_shifts.items() if v["shift_detected"]]

    return {
        "name_a": name_a,
        "name_b": name_b,
        "schema_diff": {
            "cols_only_in_a": sorted(cols_a - cols_b),
            "cols_only_in_b": sorted(cols_b - cols_a),
            "common_cols": len(common_cols),
            "dtype_changes": dtype_changes,
        },
        "size_diff": {
            "rows_a": len(df_a),
            "rows_b": len(df_b),
            "row_diff": len(df_b) - len(df_a),
            "cols_a": len(df_a.columns),
            "cols_b": len(df_b.columns),
        },
        "distribution_shifts": distribution_shifts,
        "shifted_columns": shifted_cols,
        "null_diff": null_diff,
        "overlap": {
            "common_rows": overlap_count,
            "overlap_pct": overlap_pct,
        },
        "summary": {
            "n_schema_changes": len(dtype_changes) + len(cols_a - cols_b) + len(cols_b - cols_a),
            "n_distribution_shifts": len(shifted_cols),
            "n_null_changes": len(null_diff),
        }
    }
