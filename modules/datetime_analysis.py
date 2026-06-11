"""modules/datetime_analysis.py — Time series and seasonality analysis."""
from __future__ import annotations
import pandas as pd
import numpy as np


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Detect datetime columns (native or parseable strings)."""
    dt_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
        elif df[col].dtype == object or str(df[col].dtype) == "string":
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.80:
                    dt_cols.append(col)
            except Exception:
                pass
    return dt_cols


def analyze_datetime_column(df: pd.DataFrame, col: str) -> dict:
    """Analyse a single datetime column."""
    try:
        series = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce").dropna()
        if len(series) < 2:
            return {}

        min_date = series.min()
        max_date = series.max()
        range_days = (max_date - min_date).days

        # Check if time component exists
        has_time = (series.dt.hour != 0).any()

        return {
            "min_date": str(min_date.date()),
            "max_date": str(max_date.date()),
            "range_days": range_days,
            "has_time_component": bool(has_time),
            "monthly_counts": series.dt.to_period("M").value_counts().sort_index().to_dict(),
            "weekday_counts": series.dt.dayofweek.value_counts().sort_index().to_dict(),
            "has_seasonality": range_days >= 365,
            "trend_direction": "unknown",
        }
    except Exception:
        return {}


def extract_datetime_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Add temporal feature columns derived from a datetime column."""
    df = df.copy()
    try:
        parsed = pd.to_datetime(df[dt_col], infer_datetime_format=True, errors="coerce")
        df[f"{dt_col}_year"]       = parsed.dt.year
        df[f"{dt_col}_month"]      = parsed.dt.month
        df[f"{dt_col}_day"]        = parsed.dt.day
        df[f"{dt_col}_weekday"]    = parsed.dt.dayofweek
        df[f"{dt_col}_quarter"]    = parsed.dt.quarter
        df[f"{dt_col}_is_weekend"] = parsed.dt.dayofweek >= 5
        has_time = (parsed.dt.hour != 0).any()
        if has_time:
            df[f"{dt_col}_hour"] = parsed.dt.hour
    except Exception:
        pass
    return df


def analyze_datetime(df: pd.DataFrame) -> dict:
    """Analyse all datetime columns in the dataframe."""
    dt_cols = detect_datetime_columns(df)
    result = {}
    for col in dt_cols:
        analysis = analyze_datetime_column(df, col)
        if analysis:
            result[col] = analysis
    return result
