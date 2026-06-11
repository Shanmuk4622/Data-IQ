"""modules/loader.py — File upload, validation, parsing for DataIQ."""
from __future__ import annotations
import io
import pandas as pd
import streamlit as st
from pathlib import Path

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"}
MAX_SIZE_MB = 200
WARN_SIZE_MB = 50
MIN_ROWS = 10
MIN_COLS = 2

SAMPLE_PATHS = {
    "titanic":  "assets/sample_datasets/titanic.csv",
    "housing":  "assets/sample_datasets/california_housing.csv",
    "ecommerce":"assets/sample_datasets/ecommerce_sales.csv",
}


def _read_file(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded file based on its extension."""
    name = uploaded_file.name.lower()
    ext = Path(name).suffix

    if ext == ".csv":
        return pd.read_csv(uploaded_file, low_memory=False)
    elif ext == ".tsv":
        return pd.read_csv(uploaded_file, sep="\t", low_memory=False)
    elif ext in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    elif ext == ".json":
        return pd.read_json(uploaded_file)
    elif ext == ".parquet":
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _deduplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Auto-rename duplicate column names and return list of renames."""
    cols = df.columns.tolist()
    seen: dict[str, int] = {}
    new_cols = []
    renames = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_name = f"{col}_{seen[col]}"
            new_cols.append(new_name)
            renames.append(f"'{col}' → '{new_name}'")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df, renames


def load_file(uploaded_file) -> tuple[pd.DataFrame, dict]:
    """
    Load and validate an uploaded Streamlit file.
    Returns (dataframe, meta_dict) or raises ValueError with user-friendly message.
    """
    # 1. File exists
    if uploaded_file is None:
        raise ValueError("No file provided.")

    # 2. Extension check
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported format '{ext}'. Accepted: {', '.join(SUPPORTED_EXTS)}"
        )

    # 3. File size check
    size_bytes = uploaded_file.size
    size_mb = size_bytes / 1024 ** 2
    if size_mb > MAX_SIZE_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB — maximum allowed is {MAX_SIZE_MB} MB. "
            "Please reduce the file size or sample your data first."
        )
    if size_mb > WARN_SIZE_MB:
        st.warning(
            f"Large file detected ({size_mb:.1f} MB). "
            "Analysis may take longer than usual."
        )

    # 4. Parse
    try:
        df = _read_file(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not parse file: {e}")

    # 5. Empty file
    if len(df) == 0:
        raise ValueError("The uploaded file contains 0 rows after parsing.")

    # 6. Duplicate columns
    df, renames = _deduplicate_columns(df)
    if renames:
        st.warning(
            f"Duplicate column names found and auto-renamed: {', '.join(renames)}"
        )

    # 7. Minimum size
    if df.shape[1] < MIN_COLS:
        raise ValueError(
            f"Dataset has only {df.shape[1]} column(s). Minimum required: {MIN_COLS}."
        )
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(df)} rows. Minimum required: {MIN_ROWS}."
        )

    # 8. All-null columns check (at least one column must have data)
    non_null_counts = df.notna().any(axis=0)
    if not non_null_counts.any():
        raise ValueError("All columns are entirely null. Please upload a valid dataset.")

    # ── Post-processing ───────────────────────────────────────────────────────
    # Store original dtypes
    original_dtypes = df.dtypes.astype(str).to_dict()

    # Convert dtypes for smarter inference
    try:
        df = df.convert_dtypes()
    except Exception:
        pass  # convert_dtypes may fail on edge cases; ignore

    # Strip leading/trailing whitespace from string columns
    for col in df.columns:
        try:
            if df[col].dtype == object or str(df[col].dtype) == "string":
                df[col] = df[col].str.strip()
        except Exception:
            pass

    meta = {
        "file_name": uploaded_file.name,
        "rows": len(df),
        "columns": df.shape[1],
        "total_cells": df.size,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
        "file_size_mb": round(size_mb, 3),
        "original_dtypes": original_dtypes,
    }
    return df, meta


def load_sample_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    """Load one of the three built-in sample datasets by name."""
    if name not in SAMPLE_PATHS:
        raise ValueError(f"Unknown sample: '{name}'. Choose from: {list(SAMPLE_PATHS.keys())}")

    path = SAMPLE_PATHS[name]
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        raise ValueError(
            f"Sample dataset '{name}' not found at '{path}'. "
            "Run `python assets/generate_samples.py` to create it."
        )

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    meta = {
        "file_name": f"{name}_sample.csv",
        "rows": len(df),
        "columns": df.shape[1],
        "total_cells": df.size,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
        "file_size_mb": round(Path(path).stat().st_size / 1024 ** 2, 3),
        "original_dtypes": df.dtypes.astype(str).to_dict(),
    }
    return df, meta
