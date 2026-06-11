"""modules/feature_engineer.py — Rule-based feature engineering suggestions."""
from __future__ import annotations
import pandas as pd


RULES = [
    {
        "name": "datetime_extraction",
        "condition": lambda col, profile: profile["columns"].get(col, {}).get("dtype_class") == "datetime",
        "suggestion": "Extract year, month, day, weekday, quarter, is_weekend as separate features",
        "new_cols": ["{col}_year", "{col}_month", "{col}_day", "{col}_weekday", "{col}_quarter"],
        "rationale": "DateTime components often have strong seasonal/cyclical predictive power",
        "code": (
            "df['{col}_year']       = pd.to_datetime(df['{col}']).dt.year\n"
            "df['{col}_month']      = pd.to_datetime(df['{col}']).dt.month\n"
            "df['{col}_day']        = pd.to_datetime(df['{col}']).dt.day\n"
            "df['{col}_weekday']    = pd.to_datetime(df['{col}']).dt.dayofweek\n"
            "df['{col}_quarter']    = pd.to_datetime(df['{col}']).dt.quarter\n"
            "df['{col}_is_weekend'] = pd.to_datetime(df['{col}']).dt.dayofweek >= 5"
        ),
    },
    {
        "name": "age_binning",
        "condition": lambda col, profile: (
            profile["columns"].get(col, {}).get("dtype_class") == "numerical"
            and "age" in col.lower()
        ),
        "suggestion": "Bin age into groups: child(0-12), teen(13-17), adult(18-64), senior(65+)",
        "new_cols": ["{col}_group"],
        "rationale": "Age bins capture non-linear relationships and domain knowledge",
        "code": "df['{col}_group'] = pd.cut(df['{col}'], bins=[0,12,17,64,200], labels=['child','teen','adult','senior'])",
    },
    {
        "name": "text_length",
        "condition": lambda col, profile: profile["columns"].get(col, {}).get("dtype_class") == "text",
        "suggestion": "Create text length feature — often predictive without complex NLP",
        "new_cols": ["{col}_length"],
        "rationale": "String length is a cheap, often predictive proxy for text complexity",
        "code": "df['{col}_length'] = df['{col}'].str.len().fillna(0).astype(int)",
    },
    {
        "name": "log_transform",
        "condition": lambda col, profile: (
            profile["columns"].get(col, {}).get("dtype_class") == "numerical"
            and abs(profile.get("statistics", {}).get("numerical", {}).get(col, {}).get("skewness", 0)) > 1.5
        ),
        "suggestion": "Apply log1p transform to reduce right skew",
        "new_cols": ["{col}_log"],
        "rationale": "Log transform normalises heavily right-skewed distributions, improving model fit",
        "code": "import numpy as np\ndf['{col}_log'] = np.log1p(df['{col}'].clip(lower=0))",
    },
    {
        "name": "category_frequency",
        "condition": lambda col, profile: (
            profile["columns"].get(col, {}).get("dtype_class") == "categorical"
            and profile["columns"].get(col, {}).get("unique_count", 0) > 10
        ),
        "suggestion": "Encode high-cardinality column as frequency (count) encoding",
        "new_cols": ["{col}_freq"],
        "rationale": "Frequency encoding captures popularity without the dimensionality of one-hot encoding",
        "code": "freq_map = df['{col}'].value_counts().to_dict()\ndf['{col}_freq'] = df['{col}'].map(freq_map).fillna(0)",
    },
]


def suggest_features(df: pd.DataFrame, profile: dict) -> list[dict]:
    """Generate feature engineering suggestions based on column types and stats."""
    suggestions = []

    for col in df.columns:
        for rule in RULES:
            try:
                if rule["condition"](col, profile):
                    preview = None
                    new_col_name = rule["new_cols"][0].replace("{col}", col)
                    code = rule["code"].replace("{col}", col)

                    # Generate a preview if simple
                    try:
                        if rule["name"] == "text_length":
                            preview = df[col].str.len().dropna().head(5).tolist()
                        elif rule["name"] == "log_transform":
                            import numpy as np
                            preview = np.log1p(
                                pd.to_numeric(df[col], errors="coerce").dropna().clip(lower=0)
                            ).head(5).round(3).tolist()
                        elif rule["name"] == "category_frequency":
                            freq_map = df[col].value_counts().to_dict()
                            preview = df[col].map(freq_map).dropna().head(5).tolist()
                    except Exception:
                        preview = None

                    suggestions.append({
                        "rule_name": rule["name"],
                        "source_col": col,
                        "suggestion": rule["suggestion"],
                        "new_col_name": new_col_name,
                        "rationale": rule["rationale"],
                        "example_code": code,
                        "preview": preview,
                    })
            except Exception:
                pass

    return suggestions
