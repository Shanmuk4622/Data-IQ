"""tests/test_quality.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from modules.quality import analyze_missing, analyze_duplicates, analyze_consistency


def make_df():
    return pd.DataFrame({
        "a": [1, 2, None, 4, 5, 1, 2, None, 4, 5],
        "b": ["Male", "Female", "male", "MALE", "female", "Male", "Female", "male", "MALE", "female"],
        "c": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })


def test_missing_detection():
    df = make_df()
    missing = analyze_missing(df)
    assert "a" in missing
    assert missing["a"]["count"] == 2
    assert missing["a"]["severity"] in ("low", "medium", "high", "critical")


def test_no_missing():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    missing = analyze_missing(df)
    assert len(missing) == 0


def test_duplicate_detection():
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    result = analyze_duplicates(df)
    assert result["count"] == 1
    assert result["pct"] > 0


def test_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = analyze_duplicates(df)
    assert result["count"] == 0


def test_case_inconsistency():
    df = pd.DataFrame({"col": ["Male", "male", "Female", "FEMALE", "Male", "male"]})
    issues = analyze_consistency(df)
    assert "col" in issues
    found = any(i["issue_type"] == "case_inconsistency" for i in issues["col"])
    assert found


def test_severity_levels():
    n = 100
    df = pd.DataFrame({
        "low_missing": [None if i < 3 else i for i in range(n)],
        "high_missing": [None if i < 40 else i for i in range(n)],
        "critical_missing": [None if i < 85 else i for i in range(n)],
    })
    missing = analyze_missing(df)
    assert missing["low_missing"]["severity"] == "low"
    assert missing["high_missing"]["severity"] in ("medium", "high")
    assert missing["critical_missing"]["severity"] == "critical"
