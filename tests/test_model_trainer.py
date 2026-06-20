"""tests/test_model_trainer.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from modules.profiler import profile_columns
from modules import model_trainer as mt


# ── Synthetic datasets + minimal profiles ────────────────────────────────────
def _make_clf_df(n=200):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    # A learnable, separable target
    y = ((x1 + x2 + (cat == "a") * 1.5) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": y})


def _make_reg_df(n=200):
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 * x1 - 2.0 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _profile_for(df):
    return {
        "columns": profile_columns(df),
        "preprocessing_recommendations": {},
        "meta": {"rows": len(df), "columns": df.shape[1]},
    }


_DEFAULT_OPTS = {"test_size": 0.25, "random_state": 42, "stratify": True,
                 "cv_folds": 0, "imbalance": "none"}


# ── Registry / estimator factory ──────────────────────────────────────────────
def test_list_models():
    assert "Random Forest Classifier" in mt.list_models("classification")
    assert "XGBoost Regressor" in mt.list_models("regression")
    assert "K-Means" in mt.list_models("clustering")


def test_get_estimator_returns_right_class():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(mt.get_estimator("Ridge Regression", "regression"), Ridge)
    assert isinstance(
        mt.get_estimator("Random Forest Classifier", "classification"),
        RandomForestClassifier,
    )


def test_default_hyperparams():
    hp = mt.get_default_hyperparams("classification", "Random Forest Classifier")
    assert hp["n_estimators"] == 200
    assert "min_samples_split" in hp


# ── Supervised training ───────────────────────────────────────────────────────
def test_train_classification():
    df = _make_clf_df()
    profile = _profile_for(df)
    hp = {"n_estimators": 40, "max_depth": 0, "min_samples_split": 2}
    res = mt.train_supervised(df, "target", "classification",
                              "Random Forest Classifier", hp, _DEFAULT_OPTS, profile)
    assert res["ok"], res.get("error")
    m = res["metrics"]
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["accuracy"] > 0.6  # data is separable
    assert len(m["confusion_matrix"]) == 2
    assert res["feature_importance"]  # non-empty


def test_train_regression():
    df = _make_reg_df()
    profile = _profile_for(df)
    res = mt.train_supervised(df, "target", "regression",
                              "Linear Regression", {}, _DEFAULT_OPTS, profile)
    assert res["ok"], res.get("error")
    m = res["metrics"]
    assert m["r2"] > 0.8          # near-linear signal
    assert m["rmse"] >= 0.0


def test_cross_validation_runs():
    df = _make_reg_df()
    profile = _profile_for(df)
    opts = dict(_DEFAULT_OPTS, cv_folds=3)
    res = mt.train_supervised(df, "target", "regression",
                              "Ridge Regression", {"alpha": 1.0}, opts, profile)
    assert res["ok"], res.get("error")
    assert res["cv_scores"] is not None
    assert len(res["cv_scores"]) == 3


def test_missing_target_handled():
    df = _make_reg_df()
    profile = _profile_for(df)
    res = mt.train_supervised(df, "does_not_exist", "regression",
                              "Linear Regression", {}, _DEFAULT_OPTS, profile)
    assert res["ok"] is False
    assert "error" in res


# ── Leaderboard ───────────────────────────────────────────────────────────────
def test_leaderboard_ranked():
    df = _make_reg_df(150)
    profile = _profile_for(df)
    rows = mt.run_leaderboard(df, "target", "regression", profile)
    ok_rows = [r for r in rows if r.get("ok")]
    assert len(ok_rows) >= 2
    # ranked descending by R²
    scores = [r["R²"] for r in ok_rows]
    assert scores == sorted(scores, reverse=True)


# ── Clustering ────────────────────────────────────────────────────────────────
def test_clustering_kmeans():
    df = _make_reg_df().drop(columns=["target"])
    profile = _profile_for(df)
    res = mt.train_clustering(df, profile, "K-Means", {"n_clusters": 3, "n_init": 5})
    assert res["ok"], res.get("error")
    assert res["n_clusters"] == 3
    assert res["pca_x"] is not None


# ── Modeling guide ────────────────────────────────────────────────────────────
def test_generate_modeling_guide():
    df = _make_clf_df()
    profile = _profile_for(df)
    # add the bits the guide reads so it exercises those branches
    from modules.quality import analyze_quality
    profile["quality"] = analyze_quality(df)
    profile["ml_readiness"] = {"problem_type": "classification", "recommended_models": [
        {"name": "Random Forest Classifier"}]}
    guide = mt.generate_modeling_guide(profile, "target", "classification")
    assert isinstance(guide, str)
    assert "How to model" in guide
    assert "Recommended models" in guide
