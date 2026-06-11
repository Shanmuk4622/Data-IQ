"""modules/quality.py — Missing values, duplicates, consistency checks."""
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st


def _null_severity(pct: float) -> str:
    if pct < 0.05:
        return "low"
    elif pct < 0.30:
        return "medium"
    elif pct < 0.80:
        return "high"
    return "critical"


def analyze_missing(df: pd.DataFrame) -> dict:
    """Analyse missing values across all columns."""
    n = len(df)
    result = {}
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count > 0:
            pct = count / n
            result[col] = {
                "count": count,
                "pct": round(pct, 4),
                "severity": _null_severity(pct),
            }
    return result


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """Detect exact duplicate rows and duplicate columns."""
    n = len(df)
    exact_dupes = int(df.duplicated().sum())

    duplicate_cols = []
    cols = df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                if df[cols[i]].equals(df[cols[j]]):
                    duplicate_cols.append((cols[i], cols[j]))
            except Exception:
                pass

    return {
        "count": exact_dupes,
        "pct": round(exact_dupes / n, 4) if n > 0 else 0.0,
        "duplicate_col_pairs": duplicate_cols,
    }


def analyze_consistency(df: pd.DataFrame) -> dict:
    """Detect case inconsistency, mixed types, and symbol contamination."""
    issues = {}

    for col in df.columns:
        series = df[col]
        col_issues = []

        # String columns only — handle both object and pandas 3.x StringDtype
        col_dtype_str = str(series.dtype)
        is_string_col = (
            series.dtype == object
            or col_dtype_str in ("string", "StringDtype", "str")
            or col_dtype_str.startswith("string")
            or pd.api.types.is_string_dtype(series)
        )

        if is_string_col:
            clean = series.dropna().astype(str)
            if len(clean) == 0:
                pass
            else:
                # Case inconsistency
                original_unique = clean.nunique()
                normalised_unique = clean.str.lower().str.strip().nunique()
                if normalised_unique < original_unique:
                    diff = original_unique - normalised_unique
                    examples = _find_case_examples(clean)
                    col_issues.append({
                        "issue_type": "case_inconsistency",
                        "detail": f"{diff} values differ only by case",
                        "examples": examples[:5],
                        "count": diff,
                    })

                # Symbol contamination: looks numeric but has $, %, etc.
                symbol_mask = clean.str.contains(r"[\$%#@€£¥]", regex=True, na=False)
                symbol_count = int(symbol_mask.sum())
                numeric_after_strip = pd.to_numeric(
                    clean.str.replace(r"[\$%,€£¥]", "", regex=True), errors="coerce"
                ).notna().sum()
                if symbol_count > 0 and numeric_after_strip > symbol_count * 0.5:
                    col_issues.append({
                        "issue_type": "symbol_contamination",
                        "detail": f"{symbol_count} values contain currency/symbol chars",
                        "examples": clean[symbol_mask].head(5).tolist(),
                        "count": symbol_count,
                    })

                # Mixed type detection
                numeric_ok = pd.to_numeric(clean, errors="coerce").notna()
                n_numeric = int(numeric_ok.sum())
                n_non_numeric = int((~numeric_ok).sum())
                if len(clean) >= 5 and n_numeric > 5 and n_non_numeric > 5:
                    mix_pct = min(n_numeric, n_non_numeric) / len(clean)
                    if mix_pct > 0.05:
                        col_issues.append({
                            "issue_type": "mixed_types",
                            "detail": f"{n_numeric} numeric / {n_non_numeric} text values mixed",
                            "examples": [],
                            "count": min(n_numeric, n_non_numeric),
                        })

        if col_issues:
            issues[col] = col_issues

    return issues



def _find_case_examples(series: pd.Series) -> list:
    """Find pairs of values that differ only by case."""
    groups: dict[str, list] = {}
    for v in series.unique():
        key = v.lower().strip()
        groups.setdefault(key, []).append(v)
    examples = []
    for key, vals in groups.items():
        if len(vals) > 1:
            examples.append(vals[:3])
    return examples[:5]


def analyze_quality(df: pd.DataFrame) -> dict:
    """Master quality analysis function — returns full quality dict."""
    missing = analyze_missing(df)
    dupes = analyze_duplicates(df)
    consistency = analyze_consistency(df)

    return {
        "missing_values": missing,
        "duplicate_rows": {
            "count": dupes["count"],
            "pct": dupes["pct"],
        },
        "duplicate_cols": dupes["duplicate_col_pairs"],
        "consistency_issues": consistency,
        "invalid_values": {},   # populated by other modules if needed
    }
