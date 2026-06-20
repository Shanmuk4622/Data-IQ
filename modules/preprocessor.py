"""modules/preprocessor.py — sklearn Pipeline builder and automated data cleaning."""
from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
import io


def recommend_preprocessing(col: str, col_profile: dict, col_stats: dict) -> dict:
    """Return recommended preprocessing strategy for a single column."""
    recs: dict = {}
    dtype = col_profile.get("dtype_class", "unknown")
    null_pct = col_profile.get("null_pct", 0.0)

    # Drop recommendations
    if dtype == "identifier":
        recs["drop"] = True
        recs["drop_reason"] = "Identifier column — no predictive value"
        return recs

    if null_pct > 0.80:
        recs["drop"] = True
        recs["drop_reason"] = f"{null_pct*100:.0f}% null — insufficient data"
        return recs

    recs["drop"] = False
    recs["drop_reason"] = None

    # Imputation
    if dtype == "numerical":
        skewness = col_stats.get("skewness", 0) if col_stats else 0
        if skewness and abs(skewness) > 1:
            recs["imputation"] = "median"
        else:
            recs["imputation"] = "mean"
    elif dtype in ("categorical", "boolean", "text"):
        recs["imputation"] = "mode"
    elif dtype == "datetime":
        recs["imputation"] = "none"
    else:
        recs["imputation"] = "mode"

    # Encoding
    if dtype in ("categorical", "boolean"):
        n_unique = col_profile.get("unique_count", 100)
        if n_unique <= 10:
            recs["encoding"] = "one_hot"
        elif n_unique <= 50:
            recs["encoding"] = "ordinal"
            recs["encoding_note"] = "High cardinality — ordinal encoding used"
        else:
            recs["encoding"] = "frequency"
            recs["encoding_note"] = "Very high cardinality — frequency encoding used"
    else:
        recs["encoding"] = None

    # Scaling
    if dtype == "numerical":
        outlier_pct = 0.0
        recs["scaling"] = "robust" if outlier_pct > 0.10 else "standard"
    else:
        recs["scaling"] = None

    return recs


def recommend_preprocessing_all(df: pd.DataFrame, profile: dict) -> dict:
    """Build preprocessing recommendations for all columns."""
    result = {}
    columns_profile = profile.get("columns", {})
    num_stats = profile.get("statistics", {}).get("numerical", {})

    for col, col_profile in columns_profile.items():
        col_stats = num_stats.get(col, {})
        recs = recommend_preprocessing(col, col_profile, col_stats)
        recs["strategy"] = recs.get("encoding", recs.get("scaling", "pass"))
        result[col] = recs

    return result


def build_preprocessor(df: pd.DataFrame, profile: dict, user_choices: dict, target: str | None = None):
    """
    Build and return an *unfitted* sklearn ColumnTransformer over the feature columns.

    The `target` column (if given) is always excluded so the same transformer can be
    embedded in a leak-free Pipeline([preprocessor, estimator]) for training/CV.
    Returns None if no usable feature columns remain.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
    )

    columns_profile = profile.get("columns", {})
    preprocessing_recs = profile.get("preprocessing_recommendations", {})

    cat_cols_ohe = []
    cat_cols_ord = []
    scale_robust = []
    scale_standard = []
    scale_minmax = []

    for col in df.columns:
        if target is not None and col == target:
            continue
        col_choice = user_choices.get(col, preprocessing_recs.get(col, {}))
        if col_choice.get("drop", False):
            continue
        dtype = columns_profile.get(col, {}).get("dtype_class", "unknown")
        if dtype == "numerical":
            scaling = col_choice.get("scaling")
            if scaling == "robust":
                scale_robust.append(col)
            elif scaling == "minmax":
                scale_minmax.append(col)
            else:
                scale_standard.append(col)
        elif dtype in ("categorical", "boolean"):
            enc = col_choice.get("encoding", "one_hot")
            if enc == "one_hot":
                cat_cols_ohe.append(col)
            elif enc in ("ordinal", "frequency"):
                cat_cols_ord.append(col)
            else:
                cat_cols_ohe.append(col)

    transformers = []

    if scale_standard:
        transformers.append((
            "num_standard",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]),
            scale_standard,
        ))

    if scale_robust:
        transformers.append((
            "num_robust",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]),
            scale_robust,
        ))

    if scale_minmax:
        transformers.append((
            "num_minmax",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
            ]),
            scale_minmax,
        ))

    if cat_cols_ohe:
        transformers.append((
            "cat_ohe",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols_ohe,
        ))

    if cat_cols_ord:
        transformers.append((
            "cat_ord",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols_ord,
        ))

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(df: pd.DataFrame, profile: dict, user_choices: dict):
    """Build and return a *fitted* sklearn ColumnTransformer (backwards-compatible)."""
    ct = build_preprocessor(df, profile, user_choices, target=None)
    if ct is None:
        return None
    ct.fit(df)
    return ct


def apply_cleaning(df: pd.DataFrame, profile: dict, user_choices: dict) -> tuple[pd.DataFrame, dict]:
    """Apply automated data cleaning steps. Returns (cleaned_df, report_dict)."""
    original_shape = df.shape
    report = {}
    df = df.copy()

    preprocessing_recs = profile.get("preprocessing_recommendations", {})
    columns_profile = profile.get("columns", {})

    # 1. Drop recommended columns
    cols_to_drop = [
        col for col, choice in user_choices.items()
        if choice.get("drop", preprocessing_recs.get(col, {}).get("drop", False))
        and col in df.columns
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        report["dropped_columns"] = cols_to_drop

    # 2. Remove exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    removed_dupes = n_before - len(df)
    report["removed_duplicates"] = removed_dupes

    # 3. Fix case inconsistency in categoricals
    for col in df.select_dtypes(include=["object", "string"]).columns:
        dtype = columns_profile.get(col, {}).get("dtype_class", "")
        if dtype == "categorical":
            df[col] = df[col].str.lower().str.strip()
    report["normalised_categoricals"] = True

    # 4. Impute missing values
    imputed = {}
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        choice = user_choices.get(col, preprocessing_recs.get(col, {}))
        strategy = choice.get("imputation", "mean")
        dtype = columns_profile.get(col, {}).get("dtype_class", "unknown")

        try:
            if strategy == "mean" and dtype == "numerical":
                val = df[col].astype(float).mean()
                df[col] = df[col].fillna(val)
                imputed[col] = f"mean ({val:.3f})"
            elif strategy == "median" and dtype == "numerical":
                val = df[col].astype(float).median()
                df[col] = df[col].fillna(val)
                imputed[col] = f"median ({val:.3f})"
            elif strategy == "mode":
                val = df[col].mode()
                if len(val) > 0:
                    df[col] = df[col].fillna(val.iloc[0])
                    imputed[col] = f"mode ({val.iloc[0]})"
            elif strategy == "none":
                pass
        except Exception:
            pass

    report["imputed_columns"] = imputed
    report["shape_before"] = original_shape
    report["shape_after"] = df.shape
    report["null_count_before"] = int(sum(profile["columns"][c]["null_count"] for c in profile["columns"] if c in df.columns))
    report["null_count_after"] = int(df.isna().sum().sum())
    report["memory_before_mb"] = round(profile["meta"].get("memory_mb", 0), 3)
    report["memory_after_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 3)

    return df, report


def pipeline_to_bytes(pipeline) -> bytes:
    """Pickle a sklearn pipeline to bytes for download."""
    buf = io.BytesIO()
    pickle.dump(pipeline, buf)
    return buf.getvalue()
