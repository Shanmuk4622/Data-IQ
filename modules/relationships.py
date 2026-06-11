"""modules/relationships.py — Cross-feature analysis with statistical tests."""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def num_vs_num(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Pearson r between two numerical columns."""
    try:
        sub = df[[col_a, col_b]].dropna().astype(float)
        if len(sub) < 5:
            return {}
        r, p = scipy_stats.pearsonr(sub[col_a], sub[col_b])
        return {
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "significant": p < 0.05,
            "interpretation": (
                f"Significant linear relationship (r={r:.3f}, p={p:.4f})"
                if p < 0.05 else
                f"No significant relationship (r={r:.3f}, p={p:.4f})"
            ),
        }
    except Exception:
        return {}


def num_vs_cat(df: pd.DataFrame, num_col: str, cat_col: str) -> dict:
    """One-way ANOVA: does the numerical column differ across categories?"""
    try:
        groups = [
            g[num_col].dropna().astype(float).values
            for _, g in df.groupby(cat_col)
            if len(g[num_col].dropna()) >= 3
        ]
        if len(groups) < 2:
            return {}
        f, p = scipy_stats.f_oneway(*groups)
        return {
            "f_statistic": round(float(f), 4),
            "p_value": round(float(p), 4),
            "significant": p < 0.05,
            "n_groups": len(groups),
            "interpretation": (
                f"Significant difference between groups (F={f:.2f}, p={p:.4f})"
                if p < 0.05 else
                f"No significant difference between groups (F={f:.2f}, p={p:.4f})"
            ),
        }
    except Exception:
        return {}


def cat_vs_cat(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Chi-square test + Cramér's V for two categorical columns."""
    try:
        ct = pd.crosstab(df[col_a], df[col_b])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {}
        chi2, p, dof, _ = scipy_stats.chi2_contingency(ct)
        n = ct.sum().sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
        return {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p), 4),
            "dof": int(dof),
            "cramers_v": round(cramers_v, 4),
            "significant": p < 0.05,
            "interpretation": (
                f"Significant association (χ²={chi2:.2f}, p={p:.4f}, V={cramers_v:.3f})"
                if p < 0.05 else
                f"No significant association (χ²={chi2:.2f}, p={p:.4f})"
            ),
        }
    except Exception:
        return {}
