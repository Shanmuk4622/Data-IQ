"""tests/test_health_score.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from modules.health_score import compute_health_score


def _make_profile(null_pct=0.0, dup_pct=0.0, outlier_pct=0.0, n_high_corr=0):
    return {
        "columns": {
            "a": {"null_pct": null_pct, "dtype_class": "numerical"},
            "b": {"null_pct": null_pct, "dtype_class": "numerical"},
        },
        "quality": {
            "missing_values": {},
            "duplicate_rows": {"count": 0, "pct": dup_pct},
        },
        "outliers": {
            "a": {"outlier_pct": outlier_pct},
        },
        "correlations": {
            "high_correlation_pairs": [{"col_a": "x", "col_b": "y"}] * n_high_corr,
        },
        "statistics": {"categorical": {}},
        "ml_readiness": {"target_column": None},
    }


def test_perfect_score():
    profile = _make_profile()
    result = compute_health_score(profile)
    assert result["total"] > 85
    assert result["grade"] == "A"


def test_high_missing_lowers_score():
    profile_clean = _make_profile(null_pct=0.0)
    profile_dirty = _make_profile(null_pct=0.5)
    clean = compute_health_score(profile_clean)
    dirty = compute_health_score(profile_dirty)
    assert clean["total"] > dirty["total"]


def test_grade_assignment():
    for null_pct, expected_max_grade in [(0.0, "A"), (0.3, "B"), (0.6, "C"), (0.9, "D")]:
        profile = _make_profile(null_pct=null_pct)
        result = compute_health_score(profile)
        assert result["grade"] in ["A", "B", "C", "D"]


def test_all_required_fields():
    profile = _make_profile()
    result = compute_health_score(profile)
    for field in ["total", "missing_score", "outlier_score", "duplicate_score",
                  "class_balance_score", "correlation_score", "grade", "one_line_fix"]:
        assert field in result


def test_score_bounds():
    profile = _make_profile(null_pct=1.0, dup_pct=1.0, outlier_pct=1.0, n_high_corr=20)
    result = compute_health_score(profile)
    assert 0 <= result["total"] <= 100
    assert 0 <= result["missing_score"] <= 100
