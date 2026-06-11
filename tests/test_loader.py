"""tests/test_loader.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
import pandas as pd
import pytest
from modules.loader import load_sample_dataset, _deduplicate_columns, _read_file


def test_load_titanic_sample():
    """Titanic sample loads correctly."""
    df, meta = load_sample_dataset("titanic")
    assert len(df) > 100
    assert meta["columns"] > 5
    assert meta["rows"] == len(df)


def test_load_housing_sample():
    df, meta = load_sample_dataset("housing")
    assert len(df) > 100
    assert "median_house_value" in df.columns


def test_load_ecommerce_sample():
    df, meta = load_sample_dataset("ecommerce")
    assert len(df) > 100
    assert "order_status" in df.columns


def test_deduplicate_columns():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
    df, renames = _deduplicate_columns(df)
    assert "a_1" in df.columns
    assert len(renames) == 1


def test_invalid_sample_name():
    with pytest.raises(ValueError, match="Unknown sample"):
        load_sample_dataset("nonexistent")


def test_meta_fields():
    df, meta = load_sample_dataset("titanic")
    for field in ["file_name", "rows", "columns", "total_cells", "memory_mb", "file_size_mb"]:
        assert field in meta
