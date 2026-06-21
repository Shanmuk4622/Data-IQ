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


# ── Orange: expanded registry ─────────────────────────────────────────────────
def test_new_learners_instantiate():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.dummy import DummyClassifier, DummyRegressor
    assert isinstance(mt.get_estimator("Decision Tree", "classification"), DecisionTreeClassifier)
    assert isinstance(mt.get_estimator("k-Nearest Neighbors", "classification"), KNeighborsClassifier)
    assert isinstance(mt.get_estimator("Naive Bayes", "classification"), GaussianNB)
    assert isinstance(mt.get_estimator("Constant (baseline)", "classification"), DummyClassifier)
    assert isinstance(mt.get_estimator("Constant (baseline)", "regression"), DummyRegressor)


# ── Orange: full metric suite ─────────────────────────────────────────────────
def test_full_metric_keys():
    df = _make_clf_df()
    profile = _profile_for(df)
    res = mt.train_supervised(df, "target", "classification", "Decision Tree",
                              {"max_depth": 0, "min_samples_split": 2, "criterion": "gini"},
                              _DEFAULT_OPTS, profile)
    assert res["ok"], res.get("error")
    for k in ("specificity", "logloss", "mcc"):
        assert k in res["metrics"]
    assert -1.0 <= res["metrics"]["mcc"] <= 1.0

    dfr = _make_reg_df()
    rr = mt.train_supervised(dfr, "target", "regression", "Linear Regression", {}, _DEFAULT_OPTS,
                             _profile_for(dfr))
    assert "mse" in rr["metrics"] and "cvrmse" in rr["metrics"]


# ── Orange: Test & Score ──────────────────────────────────────────────────────
def test_test_and_score_classification():
    df = _make_clf_df()
    profile = _profile_for(df)
    models = ["Logistic Regression", "Decision Tree", "Constant (baseline)"]
    res = mt.test_and_score(df, "target", "classification", models, {},
                            {"method": "cross_validation", "k": 3, "stratified": True}, profile)
    assert res["ok"], res.get("error")
    assert set(res["models"].keys()) == set(models)
    for name, r in res["models"].items():
        assert r["ok"], f"{name}: {r.get('error')}"
        assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
        assert r["fitted"] is not None
    assert res["comparison"] is not None  # CV → pairwise comparison


def test_test_and_score_regression():
    df = _make_reg_df()
    profile = _profile_for(df)
    res = mt.test_and_score(df, "target", "regression",
                            ["Linear Regression", "Random Forest Regressor"], {},
                            {"method": "cross_validation", "k": 3}, profile)
    assert res["ok"], res.get("error")
    assert res["models"]["Linear Regression"]["metrics"]["r2"] > 0.7


def test_test_and_score_random_sampling():
    df = _make_clf_df()
    res = mt.test_and_score(df, "target", "classification", ["Logistic Regression"], {},
                            {"method": "random_sampling", "test_pct": 0.3, "repeats": 2,
                             "stratified": True}, _profile_for(df))
    assert res["ok"], res.get("error")
    assert res["models"]["Logistic Regression"]["ok"]


# ── Orange: Predictions + viewers ─────────────────────────────────────────────
def test_predictions_and_tree_dot():
    df = _make_clf_df()
    profile = _profile_for(df)
    res = mt.test_and_score(df, "target", "classification",
                            ["Random Forest Classifier"], {},
                            {"method": "cross_validation", "k": 3}, profile)
    fitted = {n: r["fitted"] for n, r in res["models"].items() if r["ok"]}
    table = mt.build_predictions_table(fitted, df, "target", "classification",
                                       res["label_mapping"], max_rows=10)
    assert len(table) == 10
    assert "actual" in table.columns
    dot = mt.export_tree_dot(fitted["Random Forest Classifier"], res["classes"])
    assert dot and "digraph" in dot


# ── Orange: Rank ──────────────────────────────────────────────────────────────
def test_rank_features_methods():
    df = _make_clf_df()
    profile = _profile_for(df)
    for method in mt.RANK_METHODS["classification"]:
        ranked = mt.rank_features(df, "target", "classification", method, profile)
        assert ranked, f"{method} returned nothing"
        assert all(isinstance(n, str) for n, _ in ranked)

    dfr = _make_reg_df()
    for method in mt.RANK_METHODS["regression"]:
        ranked = mt.rank_features(dfr, "target", "regression", method, _profile_for(dfr))
        assert ranked, f"{method} returned nothing"


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
