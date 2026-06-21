"""modules/model_trainer.py — In-app ML training engine for DataIQ.

Pure logic (no Streamlit). Trains real models on the user's dataset using a leak-free
Pipeline([preprocessor, estimator]) built from the existing preprocessing recommendations.

Public surface used by pages/10_🛠_Make_ML_Model.py:
    MODEL_REGISTRY, list_models, get_model_spec, get_default_hyperparams, get_estimator
    train_supervised, run_cross_validation, tune_hyperparameters, run_leaderboard
    train_clustering, train_timeseries, extract_feature_importance
    generate_modeling_guide, model_to_bytes
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from modules.preprocessor import build_preprocessor, pipeline_to_bytes

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Estimator factories (lazy imports keep module load cheap)
# ══════════════════════════════════════════════════════════════════════════════
def _md(hp, default):
    """Read a max_depth-style param where 0 / <=0 means 'no limit' (None)."""
    v = int(hp.get("max_depth", default))
    return None if v <= 0 else v


# ── Classification ────────────────────────────────────────────────────────────
def _make_logreg(hp):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=float(hp.get("C", 1.0)),
        max_iter=int(hp.get("max_iter", 1000)),
        random_state=RANDOM_STATE,
    )


def _make_rf_clf(hp):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=int(hp.get("n_estimators", 200)),
        max_depth=_md(hp, 0),
        min_samples_split=int(hp.get("min_samples_split", 2)),
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _make_xgb_clf(hp):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=int(hp.get("n_estimators", 300)),
        learning_rate=float(hp.get("learning_rate", 0.1)),
        max_depth=int(hp.get("max_depth", 6)),
        eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _make_lgbm_clf(hp):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=int(hp.get("n_estimators", 300)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        num_leaves=int(hp.get("num_leaves", 31)),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


# ── Regression ────────────────────────────────────────────────────────────────
def _make_linreg(hp):
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _make_ridge(hp):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=float(hp.get("alpha", 1.0)), random_state=RANDOM_STATE)


def _make_rf_reg(hp):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=int(hp.get("n_estimators", 200)),
        max_depth=_md(hp, 0),
        min_samples_split=int(hp.get("min_samples_split", 2)),
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _make_xgb_reg(hp):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=int(hp.get("n_estimators", 300)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        max_depth=int(hp.get("max_depth", 6)),
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
    )


# ── Clustering ────────────────────────────────────────────────────────────────
def _make_kmeans(hp):
    from sklearn.cluster import KMeans
    return KMeans(
        n_clusters=int(hp.get("n_clusters", 4)),
        n_init=int(hp.get("n_init", 10)),
        random_state=RANDOM_STATE,
    )


def _make_dbscan(hp):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=float(hp.get("eps", 0.5)), min_samples=int(hp.get("min_samples", 5)))


def _make_agglo(hp):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=int(hp.get("n_clusters", 4)), linkage=hp.get("linkage", "ward"),
    )


# ── Extra Orange learners ─────────────────────────────────────────────────────
def _hidden(val):
    """Parse a hidden-layer spec like '100' or '100,50' into a tuple."""
    try:
        parts = [int(x) for x in str(val).replace(" ", "").split(",") if x]
        return tuple(parts) if parts else (100,)
    except Exception:
        return (100,)


def _make_dtree_clf(hp):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(
        max_depth=_md(hp, 0), min_samples_split=int(hp.get("min_samples_split", 2)),
        criterion=hp.get("criterion", "gini"), random_state=RANDOM_STATE,
    )


def _make_knn_clf(hp):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(
        n_neighbors=int(hp.get("n_neighbors", 5)), weights=hp.get("weights", "uniform"),
    )


def _make_svm_clf(hp):
    from sklearn.svm import SVC
    return SVC(
        C=float(hp.get("C", 1.0)), kernel=hp.get("kernel", "rbf"),
        probability=True, random_state=RANDOM_STATE,
    )


def _make_nb_clf(hp):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def _make_gb_clf(hp):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=int(hp.get("n_estimators", 100)),
        learning_rate=float(hp.get("learning_rate", 0.1)),
        max_depth=int(hp.get("max_depth", 3)), random_state=RANDOM_STATE,
    )


def _make_ada_clf(hp):
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(
        n_estimators=int(hp.get("n_estimators", 50)),
        learning_rate=float(hp.get("learning_rate", 1.0)), random_state=RANDOM_STATE,
    )


def _make_mlp_clf(hp):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=_hidden(hp.get("hidden_layer_sizes", "100")),
        alpha=float(hp.get("alpha", 0.0001)), max_iter=int(hp.get("max_iter", 300)),
        random_state=RANDOM_STATE,
    )


def _make_const_clf(hp):
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="prior")


def _make_dtree_reg(hp):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(
        max_depth=_md(hp, 0), min_samples_split=int(hp.get("min_samples_split", 2)),
        random_state=RANDOM_STATE,
    )


def _make_knn_reg(hp):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(
        n_neighbors=int(hp.get("n_neighbors", 5)), weights=hp.get("weights", "uniform"),
    )


def _make_svr(hp):
    from sklearn.svm import SVR
    return SVR(C=float(hp.get("C", 1.0)), kernel=hp.get("kernel", "rbf"))


def _make_gb_reg(hp):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=int(hp.get("n_estimators", 100)),
        learning_rate=float(hp.get("learning_rate", 0.1)),
        max_depth=int(hp.get("max_depth", 3)), random_state=RANDOM_STATE,
    )


def _make_ada_reg(hp):
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor(
        n_estimators=int(hp.get("n_estimators", 50)),
        learning_rate=float(hp.get("learning_rate", 1.0)), random_state=RANDOM_STATE,
    )


def _make_mlp_reg(hp):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=_hidden(hp.get("hidden_layer_sizes", "100")),
        alpha=float(hp.get("alpha", 0.0001)), max_iter=int(hp.get("max_iter", 300)),
        random_state=RANDOM_STATE,
    )


def _make_const_reg(hp):
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy="mean")


# ── Shared hyperparameter schema fragments for the extra learners ─────────────
_HP_KNN = [
    {"name": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
    {"name": "weights", "type": "select", "default": "uniform", "options": ["uniform", "distance"]},
]
_HP_SVM = [
    {"name": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1},
    {"name": "kernel", "type": "select", "default": "rbf", "options": ["rbf", "linear", "poly"]},
]
_HP_MLP = [
    {"name": "hidden_layer_sizes", "type": "select", "default": "100",
     "options": ["50", "100", "100,50", "100,100"]},
    {"name": "alpha", "type": "float", "default": 0.0001, "min": 0.00001, "max": 0.1, "step": 0.0001},
    {"name": "max_iter", "type": "int", "default": 300, "min": 100, "max": 1000, "step": 50},
]
_HP_GB = [
    {"name": "n_estimators", "type": "int", "default": 100, "min": 50, "max": 500, "step": 50},
    {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01},
    {"name": "max_depth", "type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
]
_HP_ADA = [
    {"name": "n_estimators", "type": "int", "default": 50, "min": 20, "max": 300, "step": 10},
    {"name": "learning_rate", "type": "float", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
]
_HP_DTREE = [
    {"name": "max_depth", "type": "int", "default": 0, "min": 0, "max": 50, "step": 1, "help": "0 = no limit"},
    {"name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 20, "step": 1},
    {"name": "criterion", "type": "select", "default": "gini", "options": ["gini", "entropy"]},
]


# ══════════════════════════════════════════════════════════════════════════════
#  Model registry — one source of truth for factories + UI schema + tuning grids
# ══════════════════════════════════════════════════════════════════════════════
# hyperparam schema entry: {name, type: int|float|select, default, min, max, step, options, help}
_HP_N_EST = {"name": "n_estimators", "type": "int", "default": 200, "min": 50, "max": 1000, "step": 50}
_HP_MAXDEPTH = {"name": "max_depth", "type": "int", "default": 0, "min": 0, "max": 50, "step": 1,
                "help": "0 = no limit"}
_HP_MINSPLIT = {"name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 20, "step": 1}
_HP_LR = {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}
_HP_XGB_DEPTH = {"name": "max_depth", "type": "int", "default": 6, "min": 1, "max": 15, "step": 1}

MODEL_REGISTRY: dict[str, list[dict]] = {
    "classification": [
        {
            "name": "Logistic Regression", "factory": _make_logreg,
            "proba": True, "importance": "linear",
            "hyperparams": [
                {"name": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01},
                {"name": "max_iter", "type": "int", "default": 1000, "min": 200, "max": 5000, "step": 100},
            ],
            "param_distributions": {"model__C": [0.01, 0.1, 1.0, 10.0], "model__max_iter": [500, 1000, 2000]},
        },
        {
            "name": "Random Forest Classifier", "factory": _make_rf_clf,
            "proba": True, "importance": "tree",
            "hyperparams": [dict(_HP_N_EST), dict(_HP_MAXDEPTH), dict(_HP_MINSPLIT)],
            "param_distributions": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
        },
        {
            "name": "XGBoost Classifier", "factory": _make_xgb_clf,
            "proba": True, "importance": "tree",
            "hyperparams": [
                {"name": "n_estimators", "type": "int", "default": 300, "min": 50, "max": 1000, "step": 50},
                dict(_HP_LR), dict(_HP_XGB_DEPTH),
            ],
            "param_distributions": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7, 9],
            },
        },
        {
            "name": "LightGBM Classifier", "factory": _make_lgbm_clf,
            "proba": True, "importance": "tree",
            "hyperparams": [
                {"name": "n_estimators", "type": "int", "default": 300, "min": 50, "max": 1000, "step": 50},
                {"name": "learning_rate", "type": "float", "default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01},
                {"name": "num_leaves", "type": "int", "default": 31, "min": 7, "max": 127, "step": 4},
            ],
            "param_distributions": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
            },
        },
    ],
    "regression": [
        {
            "name": "Linear Regression", "factory": _make_linreg,
            "proba": False, "importance": "linear",
            "hyperparams": [],
            "param_distributions": {},
        },
        {
            "name": "Ridge Regression", "factory": _make_ridge,
            "proba": False, "importance": "linear",
            "hyperparams": [{"name": "alpha", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}],
            "param_distributions": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        },
        {
            "name": "Random Forest Regressor", "factory": _make_rf_reg,
            "proba": False, "importance": "tree",
            "hyperparams": [dict(_HP_N_EST), dict(_HP_MAXDEPTH), dict(_HP_MINSPLIT)],
            "param_distributions": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
        },
        {
            "name": "XGBoost Regressor", "factory": _make_xgb_reg,
            "proba": False, "importance": "tree",
            "hyperparams": [
                {"name": "n_estimators", "type": "int", "default": 300, "min": 50, "max": 1000, "step": 50},
                {"name": "learning_rate", "type": "float", "default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01},
                dict(_HP_XGB_DEPTH),
            ],
            "param_distributions": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7, 9],
            },
        },
    ],
    "clustering": [
        {
            "name": "K-Means", "factory": _make_kmeans,
            "proba": False, "importance": None,
            "hyperparams": [
                {"name": "n_clusters", "type": "int", "default": 4, "min": 2, "max": 15, "step": 1},
                {"name": "n_init", "type": "int", "default": 10, "min": 1, "max": 25, "step": 1},
            ],
            "param_distributions": {},
        },
        {
            "name": "DBSCAN", "factory": _make_dbscan,
            "proba": False, "importance": None,
            "hyperparams": [
                {"name": "eps", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1},
                {"name": "min_samples", "type": "int", "default": 5, "min": 2, "max": 25, "step": 1},
            ],
            "param_distributions": {},
        },
    ],
    "timeseries": [
        {"name": "ARIMA (statsmodels)", "factory": None, "proba": False, "importance": None,
         "hyperparams": [], "param_distributions": {}},
        {"name": "Prophet (optional)", "factory": None, "proba": False, "importance": None,
         "hyperparams": [], "param_distributions": {}},
    ],
}


# ── Extra Orange learners appended to the registry (model column) ─────────────
MODEL_REGISTRY["classification"].extend([
    {"name": "Decision Tree", "factory": _make_dtree_clf, "proba": True, "importance": "tree", "viewer": "tree",
     "hyperparams": _HP_DTREE,
     "param_distributions": {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}},
    {"name": "k-Nearest Neighbors", "factory": _make_knn_clf, "proba": True, "importance": None,
     "hyperparams": _HP_KNN,
     "param_distributions": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"]}},
    {"name": "SVM", "factory": _make_svm_clf, "proba": True, "importance": None,
     "hyperparams": _HP_SVM,
     "param_distributions": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]}},
    {"name": "Naive Bayes", "factory": _make_nb_clf, "proba": True, "importance": None,
     "hyperparams": [], "param_distributions": {}},
    {"name": "Gradient Boosting", "factory": _make_gb_clf, "proba": True, "importance": "tree", "viewer": "tree",
     "hyperparams": _HP_GB,
     "param_distributions": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.05, 0.1],
                             "model__max_depth": [2, 3, 5]}},
    {"name": "AdaBoost", "factory": _make_ada_clf, "proba": True, "importance": "tree",
     "hyperparams": _HP_ADA,
     "param_distributions": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.5, 1.0]}},
    {"name": "Neural Network", "factory": _make_mlp_clf, "proba": True, "importance": None,
     "hyperparams": _HP_MLP,
     "param_distributions": {"model__alpha": [0.0001, 0.001, 0.01]}},
    {"name": "Constant (baseline)", "factory": _make_const_clf, "proba": True, "importance": None,
     "hyperparams": [], "param_distributions": {}},
])

MODEL_REGISTRY["regression"].extend([
    {"name": "Decision Tree", "factory": _make_dtree_reg, "proba": False, "importance": "tree", "viewer": "tree",
     "hyperparams": [
         {"name": "max_depth", "type": "int", "default": 0, "min": 0, "max": 50, "step": 1, "help": "0 = no limit"},
         {"name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 20, "step": 1},
     ],
     "param_distributions": {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}},
    {"name": "k-Nearest Neighbors", "factory": _make_knn_reg, "proba": False, "importance": None,
     "hyperparams": _HP_KNN,
     "param_distributions": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"]}},
    {"name": "SVM", "factory": _make_svr, "proba": False, "importance": None,
     "hyperparams": _HP_SVM,
     "param_distributions": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]}},
    {"name": "Gradient Boosting", "factory": _make_gb_reg, "proba": False, "importance": "tree", "viewer": "tree",
     "hyperparams": _HP_GB,
     "param_distributions": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.05, 0.1],
                             "model__max_depth": [2, 3, 5]}},
    {"name": "AdaBoost", "factory": _make_ada_reg, "proba": False, "importance": "tree",
     "hyperparams": _HP_ADA,
     "param_distributions": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.5, 1.0]}},
    {"name": "Neural Network", "factory": _make_mlp_reg, "proba": False, "importance": None,
     "hyperparams": _HP_MLP,
     "param_distributions": {"model__alpha": [0.0001, 0.001, 0.01]}},
    {"name": "Constant (baseline)", "factory": _make_const_reg, "proba": False, "importance": None,
     "hyperparams": [], "param_distributions": {}},
])

MODEL_REGISTRY["clustering"].append(
    {"name": "Hierarchical", "factory": _make_agglo, "proba": False, "importance": None,
     "hyperparams": [
         {"name": "n_clusters", "type": "int", "default": 4, "min": 2, "max": 15, "step": 1},
         {"name": "linkage", "type": "select", "default": "ward",
          "options": ["ward", "complete", "average", "single"]},
     ], "param_distributions": {}},
)

# Viewer tags on pre-existing learners
for _s in MODEL_REGISTRY["classification"]:
    if _s["name"] == "Random Forest Classifier":
        _s["viewer"] = "tree"
    elif _s["name"] == "Logistic Regression":
        _s["viewer"] = "nomogram"
for _s in MODEL_REGISTRY["regression"]:
    if _s["name"] == "Random Forest Regressor":
        _s["viewer"] = "tree"


def list_models(problem_type: str) -> list[str]:
    return [s["name"] for s in MODEL_REGISTRY.get(problem_type, [])]


def get_model_spec(problem_type: str, model_name: str) -> dict | None:
    for s in MODEL_REGISTRY.get(problem_type, []):
        if s["name"] == model_name:
            return s
    return None


def get_default_hyperparams(problem_type: str, model_name: str) -> dict:
    spec = get_model_spec(problem_type, model_name)
    if not spec:
        return {}
    return {p["name"]: p["default"] for p in spec.get("hyperparams", [])}


def get_estimator(model_name: str, problem_type: str, hyperparams: dict | None = None):
    """Instantiate an estimator from the registry."""
    spec = get_model_spec(problem_type, model_name)
    if spec is None or spec.get("factory") is None:
        raise ValueError(f"No estimator factory for '{model_name}' ({problem_type}).")
    return spec["factory"](hyperparams or {})


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════
def _specificity(cm) -> float:
    """Macro-averaged specificity (TN / (TN + FP)) from a confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    specs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        if denom > 0:
            specs.append(tn / denom)
    return float(np.mean(specs)) if specs else 0.0


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Full Orange-style classification metric set: AUC, CA, F1, Precision, Recall,
    Specificity, LogLoss, MCC (+ confusion matrix)."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, log_loss, matthews_corrcoef,
    )
    cm = confusion_matrix(y_true, y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity": _specificity(cm),
        "roc_auc": None,
        "logloss": None,
        "mcc": None,
        "confusion_matrix": cm.tolist(),
    }
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        pass
    try:
        if y_prob is not None:
            y_prob = np.asarray(y_prob, dtype=float)
            n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
            labels = list(range(n_classes))
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            try:
                out["logloss"] = float(log_loss(y_true, y_prob, labels=labels))
            except Exception:
                out["logloss"] = None
    except Exception:
        out["roc_auc"] = None
    return out


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Full Orange-style regression metric set: MSE, RMSE, MAE, MAPE, R², CVRMSE."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # MAPE — guard against division by zero
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    # CVRMSE — RMSE normalised by mean of actuals
    mean_true = float(np.mean(y_true))
    cvrmse = float(rmse / mean_true * 100) if mean_true != 0 else None
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "cvrmse": cvrmse}


# ══════════════════════════════════════════════════════════════════════════════
#  Feature importance
# ══════════════════════════════════════════════════════════════════════════════
def extract_feature_importance(fitted_pipeline, top_n: int = 25) -> list[tuple[str, float]]:
    """Return [(feature_name, importance)] sorted desc, from a fitted Pipeline."""
    try:
        model = fitted_pipeline.named_steps.get("model")
        pre = fitted_pipeline.named_steps.get("pre")
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = None

        if hasattr(model, "feature_importances_"):
            imps = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            imps = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return []

        if names is None or len(names) != len(imps):
            names = [f"f{i}" for i in range(len(imps))]

        pairs = sorted(zip(names, imps.tolist()), key=lambda x: abs(x[1]), reverse=True)
        return [(str(n), float(v)) for n, v in pairs[:top_n]]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Supervised training
# ══════════════════════════════════════════════════════════════════════════════
def _make_pipeline(preprocessor, estimator, imbalance: str):
    """sklearn Pipeline, or imblearn Pipeline when SMOTE is requested."""
    if imbalance == "smote":
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline([
            ("pre", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ])
    from sklearn.pipeline import Pipeline
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def run_cross_validation(pipeline, X, y, cv: int, problem_type: str, scoring: str | None = None):
    """Return an array of per-fold scores (or None on failure)."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    try:
        if problem_type == "classification":
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
            scoring = scoring or "accuracy"
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
            scoring = scoring or "r2"
        scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=scoring, n_jobs=-1)
        return np.asarray(scores, dtype=float), scoring
    except Exception:
        return None, scoring


def train_supervised(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    model_name: str,
    hyperparams: dict,
    options: dict,
    profile: dict,
    user_choices: dict | None = None,
) -> dict:
    """Train a supervised model and return a full results dict.

    options: {test_size, random_state, stratify(bool), cv_folds(int, 0=off),
              imbalance: 'none'|'smote'|'class_weight'}
    """
    from sklearn.model_selection import train_test_split

    user_choices = user_choices or {}
    opt = {
        "test_size": float(options.get("test_size", 0.2)),
        "random_state": int(options.get("random_state", RANDOM_STATE)),
        "stratify": bool(options.get("stratify", True)),
        "cv_folds": int(options.get("cv_folds", 0)),
        "imbalance": options.get("imbalance", "none"),
    }

    try:
        if target not in df.columns:
            return {"ok": False, "error": f"Target column '{target}' not found."}

        work = df.dropna(subset=[target]).copy()
        if len(work) < 20:
            return {"ok": False, "error": "Not enough rows with a non-null target (need ≥ 20)."}

        X = work.drop(columns=[target])
        y_raw = work[target]

        label_mapping = None
        if problem_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            label_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
        else:
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
            ok_mask = ~np.isnan(y)
            X, y = X.loc[ok_mask], y[ok_mask]
            if len(y) < 20:
                return {"ok": False, "error": "Target is not numeric enough for regression."}

        preprocessor = build_preprocessor(X, profile, user_choices, target=None)
        if preprocessor is None:
            return {"ok": False, "error": "No usable feature columns after preprocessing (all dropped/identifiers)."}

        estimator = get_estimator(model_name, problem_type, hyperparams)
        if opt["imbalance"] == "class_weight":
            try:
                if "class_weight" in estimator.get_params():
                    estimator.set_params(class_weight="balanced")
            except Exception:
                pass

        pipeline = _make_pipeline(preprocessor, estimator, opt["imbalance"])

        # Train/test split (stratify only when valid)
        stratify = None
        if problem_type == "classification" and opt["stratify"]:
            _, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2:
                stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=opt["test_size"], random_state=opt["random_state"], stratify=stratify,
        )

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        y_prob = None
        if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)
            except Exception:
                y_prob = None

        if problem_type == "classification":
            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            primary = ("Accuracy", metrics["accuracy"])
        else:
            metrics = compute_regression_metrics(y_test, y_pred)
            primary = ("R²", metrics["r2"])

        importance = extract_feature_importance(pipeline)

        cv_scores, cv_scoring = (None, None)
        if opt["cv_folds"] and opt["cv_folds"] >= 2:
            cv_scores, cv_scoring = run_cross_validation(pipeline, X, y, opt["cv_folds"], problem_type)

        return {
            "ok": True,
            "problem_type": problem_type,
            "model_name": model_name,
            "target": target,
            "metrics": metrics,
            "primary_metric": primary,
            "y_test": np.asarray(y_test),
            "y_pred": np.asarray(y_pred),
            "y_prob": y_prob,
            "label_mapping": label_mapping,
            "feature_importance": importance,
            "cv_scores": None if cv_scores is None else cv_scores.tolist(),
            "cv_mean": None if cv_scores is None else float(cv_scores.mean()),
            "cv_std": None if cv_scores is None else float(cv_scores.std()),
            "cv_scoring": cv_scoring,
            "train_time": round(train_time, 3),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(len(importance)) if importance else None,
            "pipeline": pipeline,
            "options": opt,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════
def tune_hyperparameters(
    df: pd.DataFrame, target: str, problem_type: str, model_name: str,
    profile: dict, user_choices: dict | None = None, n_iter: int = 20, cv: int = 5,
) -> dict:
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold

    user_choices = user_choices or {}
    spec = get_model_spec(problem_type, model_name)
    if not spec or not spec.get("param_distributions"):
        return {"ok": False, "error": f"'{model_name}' has no tunable hyperparameters."}

    try:
        work = df.dropna(subset=[target]).copy()
        X = work.drop(columns=[target])
        if problem_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(work[target].astype(str))
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
            scoring = "accuracy"
        else:
            y = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float)
            ok = ~np.isnan(y)
            X, y = X.loc[ok], y[ok]
            splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
            scoring = "r2"

        preprocessor = build_preprocessor(X, profile, user_choices, target=None)
        if preprocessor is None:
            return {"ok": False, "error": "No usable feature columns for tuning."}

        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("pre", preprocessor), ("model", get_estimator(model_name, problem_type))])

        search = RandomizedSearchCV(
            pipe, spec["param_distributions"], n_iter=n_iter, cv=splitter,
            scoring=scoring, random_state=RANDOM_STATE, n_jobs=-1,
        )
        t0 = time.time()
        search.fit(X, y)
        elapsed = time.time() - t0

        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        return {
            "ok": True, "best_params": best, "best_score": float(search.best_score_),
            "scoring": scoring, "n_iter": n_iter, "cv": cv, "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-model leaderboard
# ══════════════════════════════════════════════════════════════════════════════
def run_leaderboard(
    df: pd.DataFrame, target: str, problem_type: str, profile: dict,
    user_choices: dict | None = None, options: dict | None = None,
) -> list[dict]:
    """Train every model for the problem type with default params; return ranked rows."""
    options = dict(options or {})
    options.setdefault("cv_folds", 0)  # leaderboard is fast: skip CV by default
    rows = []
    for name in list_models(problem_type):
        hp = get_default_hyperparams(problem_type, name)
        res = train_supervised(df, target, problem_type, name, hp, options, profile, user_choices)
        if not res.get("ok"):
            rows.append({"model": name, "ok": False, "error": res.get("error", "failed")})
            continue
        m = res["metrics"]
        if problem_type == "classification":
            rows.append({
                "model": name, "ok": True,
                "Accuracy": round(m["accuracy"], 4),
                "F1 (weighted)": round(m["f1_weighted"], 4),
                "ROC-AUC": round(m["roc_auc"], 4) if m.get("roc_auc") is not None else None,
                "Train (s)": res["train_time"], "_sort": m["accuracy"],
            })
        else:
            rows.append({
                "model": name, "ok": True,
                "R²": round(m["r2"], 4), "RMSE": round(m["rmse"], 4), "MAE": round(m["mae"], 4),
                "Train (s)": res["train_time"], "_sort": m["r2"],
            })
    ranked = sorted(
        [r for r in rows if r.get("ok")], key=lambda r: r.get("_sort", float("-inf")), reverse=True
    )
    return ranked + [r for r in rows if not r.get("ok")]


# ══════════════════════════════════════════════════════════════════════════════
#  Clustering
# ══════════════════════════════════════════════════════════════════════════════
def train_clustering(
    df: pd.DataFrame, profile: dict, model_name: str, hyperparams: dict,
    user_choices: dict | None = None,
) -> dict:
    user_choices = user_choices or {}
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        preprocessor = build_preprocessor(df, profile, user_choices, target=None)
        if preprocessor is None:
            return {"ok": False, "error": "No usable feature columns for clustering."}
        X = preprocessor.fit_transform(df)
        X = np.asarray(X, dtype=float)

        estimator = get_estimator(model_name, "clustering", hyperparams)
        labels = estimator.fit_predict(X)
        labels = np.asarray(labels)

        unique = set(labels.tolist())
        n_clusters = len(unique - {-1})
        n_noise = int((labels == -1).sum())

        sil = None
        sil_samples = None
        sil_sample_labels = None
        try:
            mask = labels != -1
            if n_clusters >= 2 and mask.sum() > n_clusters:
                sil = float(silhouette_score(X[mask], labels[mask]))
                from sklearn.metrics import silhouette_samples
                vals = silhouette_samples(X[mask], labels[mask])
                sil_samples = vals.tolist()
                sil_sample_labels = labels[mask].tolist()
        except Exception:
            sil = None

        # 2-D projection for plotting
        try:
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(X)
            pca_x, pca_y = coords[:, 0].tolist(), coords[:, 1].tolist()
        except Exception:
            pca_x = pca_y = None

        # Elbow curve (KMeans only)
        elbow_k, elbow_inertias = None, None
        if model_name == "K-Means":
            try:
                from sklearn.cluster import KMeans
                ks, inertias = [], []
                for k in range(2, min(11, len(X))):
                    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
                    ks.append(k); inertias.append(float(km.inertia_))
                elbow_k, elbow_inertias = ks, inertias
            except Exception:
                pass

        return {
            "ok": True, "model_name": model_name, "labels": labels.tolist(),
            "n_clusters": n_clusters, "n_noise": n_noise, "silhouette": sil,
            "sil_samples": sil_samples, "sil_sample_labels": sil_sample_labels,
            "pca_x": pca_x, "pca_y": pca_y, "elbow_k": elbow_k, "elbow_inertias": elbow_inertias,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Time series
# ══════════════════════════════════════════════════════════════════════════════
def train_timeseries(
    df: pd.DataFrame, date_col: str, value_col: str,
    order: tuple = (1, 1, 1), horizon: int = 30, engine: str = "arima",
) -> dict:
    try:
        s = df[[date_col, value_col]].copy()
        s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
        s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
        s = s.dropna().sort_values(date_col)
        if len(s) < 10:
            return {"ok": False, "error": "Need ≥ 10 valid (date, value) points for forecasting."}

        ts = s.groupby(date_col)[value_col].mean()
        freq = pd.infer_freq(ts.index)

        # forecast index
        last = ts.index[-1]
        if freq:
            future_idx = pd.date_range(last, periods=horizon + 1, freq=freq)[1:]
        else:
            delta = ts.index.to_series().diff().median()
            if pd.isna(delta):
                delta = pd.Timedelta(days=1)
            future_idx = pd.DatetimeIndex([last + delta * (i + 1) for i in range(horizon)])

        hist_x = [str(d.date()) for d in ts.index[-500:]]
        hist_y = ts.values[-500:].astype(float).tolist()

        if engine == "prophet":
            try:
                from prophet import Prophet
            except Exception:
                return {"ok": False, "error": "Prophet is not installed. Run `pip install prophet` or use ARIMA."}
            pdf = pd.DataFrame({"ds": ts.index, "y": ts.values})
            m = Prophet()
            m.fit(pdf)
            fut = m.make_future_dataframe(periods=horizon, freq=freq or "D")
            fc = m.predict(fut).tail(horizon)
            return {
                "ok": True, "engine": "prophet", "history_x": hist_x, "history_y": hist_y,
                "forecast_x": [str(pd.Timestamp(d).date()) for d in fc["ds"]],
                "forecast_y": fc["yhat"].tolist(),
                "lower": fc["yhat_lower"].tolist(), "upper": fc["yhat_upper"].tolist(),
                "metric_label": "Model", "metric_value": "Prophet",
            }

        # statsmodels ARIMA
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(ts.values.astype(float), order=tuple(order))
        fitted = model.fit()
        fc = fitted.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
        return {
            "ok": True, "engine": "arima", "order": tuple(order),
            "history_x": hist_x, "history_y": hist_y,
            "forecast_x": [str(d.date()) for d in future_idx],
            "forecast_y": mean.tolist(),
            "lower": ci[:, 0].tolist(), "upper": ci[:, 1].tolist(),
            "metric_label": "AIC", "metric_value": round(float(fitted.aic), 2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def model_to_bytes(pipeline) -> bytes:
    """Pickle a trained pipeline/model for download (reuses preprocessor helper)."""
    return pipeline_to_bytes(pipeline)


# ══════════════════════════════════════════════════════════════════════════════
#  Deterministic modeling guide (no API key required)
# ══════════════════════════════════════════════════════════════════════════════
def generate_modeling_guide(profile: dict, target: str | None, problem_type: str) -> str:
    """Build a dataset-specific 'how to model this data' markdown walkthrough from `profile`."""
    meta = profile.get("meta", {})
    ml = profile.get("ml_readiness", {})
    quality = profile.get("quality", {})
    stats = profile.get("statistics", {})
    corr = profile.get("correlations", {})
    health = profile.get("health_score", {})
    cols = profile.get("columns", {})

    rows = meta.get("rows", 0)
    n_cols = meta.get("columns", 0)
    missing = quality.get("missing_values", {})
    dup_pct = quality.get("duplicate_rows", {}).get("pct", 0) * 100
    high_corr = corr.get("high_correlation_pairs", [])
    rec_models = [m["name"] for m in ml.get("recommended_models", [])[:3]]
    blockers = ml.get("blockers", [])
    warnings = ml.get("warnings", [])

    pt_label = {"classification": "Classification", "regression": "Regression",
                "clustering": "Clustering (unsupervised)", "timeseries": "Time-Series Forecasting"}.get(
        problem_type, problem_type.title())

    L = []
    L.append(f"## 🎯 How to model `{meta.get('file_name', 'this dataset')}`")
    L.append(f"**Detected task:** {pt_label}  •  **Target:** `{target or 'None (unsupervised)'}`  "
             f"•  **Shape:** {rows:,} rows × {n_cols} cols  "
             f"•  **Health:** {health.get('total', 0):.0f}/100 (Grade {health.get('grade', '?')})")
    L.append("")

    # 1. Framing
    L.append("### 1. Frame the problem")
    if problem_type == "classification":
        tgt_stats = stats.get("categorical", {}).get(target, {}) if target else {}
        ratio = tgt_stats.get("class_imbalance_ratio")
        L.append(f"- Predict the discrete class in `{target}`. "
                 + (f"Class imbalance ratio is **{ratio:.1f}:1** — "
                    + ("watch this closely." if ratio and ratio > 3 else "fairly balanced.")
                    if ratio else "Check class balance before trusting accuracy."))
        L.append("- Prefer **F1 / ROC-AUC** over raw accuracy if classes are imbalanced.")
    elif problem_type == "regression":
        tgt_stats = stats.get("numerical", {}).get(target, {}) if target else {}
        sk = tgt_stats.get("skewness")
        L.append(f"- Predict the continuous value in `{target}`.")
        if sk is not None and abs(sk) > 1:
            L.append(f"- The target is skewed (skew={sk:.2f}); a **log-transform** often improves linear models.")
        L.append("- Track **RMSE** (penalises large errors) alongside **R²**.")
    elif problem_type == "clustering":
        L.append("- No target column — group similar rows. Use the **silhouette score** and the PCA scatter "
                 "to judge cluster quality, and the elbow plot to choose *k* for K-Means.")
    else:
        L.append("- A datetime column is present — forecast a numeric value forward in time. "
                 "Pick the date column and the value to forecast, then set the horizon.")
    L.append("")

    # 2. Fix data issues
    L.append("### 2. Data issues to address first")
    issues = []
    if blockers:
        issues.append(f"🚫 **Blockers:** {'; '.join(blockers)}")
    if missing:
        worst = sorted(missing.items(), key=lambda x: x[1]["pct"], reverse=True)[:3]
        issues.append("🕳 **Missing values** in "
                      + ", ".join(f"`{c}` ({v['pct']*100:.0f}%)" for c, v in worst)
                      + " — handled automatically by the pipeline's imputers.")
    if dup_pct > 0:
        issues.append(f"🗃 **{dup_pct:.1f}% duplicate rows** — clean them on the Preprocessing page.")
    if high_corr:
        pair = high_corr[0]
        issues.append(f"🔗 **Multicollinearity**: `{pair['col_a']}` ↔ `{pair['col_b']}` (r={pair['pearson_r']}). "
                      "Tree models tolerate it; linear models benefit from dropping one.")
    if warnings:
        issues.append("⚠️ " + "; ".join(warnings[:2]))
    if not issues:
        issues.append("✅ No major data issues detected — you can train directly.")
    L += [f"- {it}" for it in issues]
    L.append("")

    # 3. Recommended approach
    L.append("### 3. Recommended models")
    if rec_models:
        L.append("Based on your data profile, start with:")
        L += [f"{i+1}. **{m}**" for i, m in enumerate(rec_models)]
        L.append("Begin with the simplest as a baseline, then try the boosted trees for best accuracy.")
    else:
        L.append("- Train a couple of models from the dropdown and compare on the **Leaderboard** tab.")
    L.append("")

    # 4. Workflow
    L.append("### 4. Suggested workflow on this page")
    L.append("1. **Train** tab → pick a model, keep defaults, click **Train Model**.")
    L.append("2. Inspect the metrics, confusion matrix / residuals, and feature importance.")
    L.append("3. Turn on **cross-validation** to confirm the score is stable, not a lucky split.")
    if problem_type == "classification":
        L.append("4. If a class is rare, enable **SMOTE** or **class-weight** in the options.")
    L.append("5. **Leaderboard** tab → train all models at once and compare.")
    L.append("6. **Tune** tab → run a randomized search on the winner.")
    L.append("7. Download the fitted pipeline `.pkl` and reuse it in your own code.")
    L.append("")
    L.append("> 💬 Ask the AI chat on the right for anything specific — it can see this dataset's profile.")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
#  Orange-style: Test & Score (multi-learner comparison)
# ══════════════════════════════════════════════════════════════════════════════
def _build_xy(df, target, problem_type):
    """Split off the target; label-encode it for classification. Returns (X, y, label_mapping)."""
    work = df.dropna(subset=[target]).copy()
    X = work.drop(columns=[target])
    label_mapping = None
    if problem_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(work[target].astype(str))
        label_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
    else:
        y = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(y)
        X, y = X.loc[m], y[m]
    return X, y, label_mapping


def _evaluate(pipe, X, y, problem_type, sampling, splitter):
    """Return pooled (y_true, y_pred, y_prob) for the chosen sampling scheme."""
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, cross_val_predict, LeaveOneOut
    method = sampling.get("method", "cross_validation")
    is_clf = problem_type == "classification"

    if method == "test_on_train":
        p = clone(pipe); p.fit(X, y)
        y_prob = p.predict_proba(X) if (is_clf and hasattr(p, "predict_proba")) else None
        return np.asarray(y), np.asarray(p.predict(X)), y_prob

    if method == "random_sampling":
        repeats = int(sampling.get("repeats", 3))
        test_pct = float(sampling.get("test_pct", 0.3))
        stratified = bool(sampling.get("stratified", True))
        yt, yp, ypr = [], [], []
        for r in range(repeats):
            strat = y if (is_clf and stratified) else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_pct, random_state=RANDOM_STATE + r, stratify=strat)
            p = clone(pipe); p.fit(Xtr, ytr)
            yt.append(np.asarray(yte)); yp.append(np.asarray(p.predict(Xte)))
            if is_clf and hasattr(p, "predict_proba"):
                ypr.append(np.asarray(p.predict_proba(Xte)))
        y_prob = np.concatenate(ypr) if (ypr and len(ypr) == repeats) else None
        return np.concatenate(yt), np.concatenate(yp), y_prob

    if method == "leave_one_out":
        cv = LeaveOneOut()
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        y_prob = None
        if is_clf:
            try:
                y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
            except Exception:
                y_prob = None
        return np.asarray(y), np.asarray(y_pred), y_prob

    # cross_validation (default)
    y_pred = cross_val_predict(pipe, X, y, cv=splitter, n_jobs=-1)
    y_prob = None
    if is_clf:
        try:
            y_prob = cross_val_predict(pipe, X, y, cv=splitter, method="predict_proba", n_jobs=-1)
        except Exception:
            y_prob = None
    return np.asarray(y), np.asarray(y_pred), y_prob


def _pairwise_comparison(fold_scores: dict) -> dict:
    """Approximate Orange's model comparison: P(row score > column score) from paired fold scores."""
    from scipy.stats import ttest_rel
    names = list(fold_scores.keys())
    matrix = {}
    for a in names:
        row = {}
        for b in names:
            if a == b:
                row[b] = None
                continue
            sa, sb = np.asarray(fold_scores[a]), np.asarray(fold_scores[b])
            n = min(len(sa), len(sb))
            try:
                diff = sa[:n] - sb[:n]
                if np.allclose(diff, 0):
                    row[b] = 0.5
                else:
                    t, p = ttest_rel(sa[:n], sb[:n])
                    row[b] = round(float(1 - p / 2 if t > 0 else p / 2), 3)
            except Exception:
                row[b] = None
        matrix[a] = row
    return {"models": names, "matrix": matrix}


def test_and_score(
    df, target, problem_type, selected_models, hyperparams_map, sampling,
    profile, user_choices=None,
) -> dict:
    """Evaluate several learners under one sampling scheme (Orange's Test & Score)."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

    user_choices = user_choices or {}
    try:
        if not selected_models:
            return {"ok": False, "error": "Select at least one model."}

        X, y, label_mapping = _build_xy(df, target, problem_type)
        if len(y) < 20:
            return {"ok": False, "error": "Not enough labelled rows (need ≥ 20)."}

        method = sampling.get("method", "cross_validation")
        if method == "leave_one_out" and len(y) > 1000:
            return {"ok": False, "error": "Leave-one-out is too slow for >1000 rows — use cross-validation."}

        preprocessor = build_preprocessor(X, profile, user_choices, target=None)
        if preprocessor is None:
            return {"ok": False, "error": "No usable feature columns after preprocessing."}

        k = int(sampling.get("k", 5))
        if problem_type == "classification":
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
            primary_scoring = "accuracy"
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
            primary_scoring = "r2"

        results, fold_scores = {}, {}
        for name in selected_models:
            try:
                hp = hyperparams_map.get(name) or get_default_hyperparams(problem_type, name)
                pipe = Pipeline([("pre", clone(preprocessor)), ("model", get_estimator(name, problem_type, hp))])

                t0 = time.time()
                y_true, y_pred, y_prob = _evaluate(pipe, X, y, problem_type, sampling, splitter)
                elapsed = time.time() - t0

                if problem_type == "classification":
                    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
                else:
                    metrics = compute_regression_metrics(y_true, y_pred)
                metrics["time"] = round(elapsed, 3)

                fitted = Pipeline([("pre", clone(preprocessor)),
                                   ("model", get_estimator(name, problem_type, hp))])
                fitted.fit(X, y)

                if method == "cross_validation":
                    try:
                        fold_scores[name] = np.asarray(
                            cross_val_score(pipe, X, y, cv=splitter, scoring=primary_scoring, n_jobs=-1),
                            dtype=float)
                    except Exception:
                        pass

                results[name] = {
                    "ok": True, "metrics": metrics,
                    "y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
                    "y_prob": (None if y_prob is None else np.asarray(y_prob)),
                    "fitted": fitted,
                }
            except Exception as e:
                results[name] = {"ok": False, "error": str(e)}

        comparison = _pairwise_comparison(fold_scores) if len(fold_scores) >= 2 else None
        classes = [label_mapping[i] for i in sorted(label_mapping)] if label_mapping else None

        return {
            "ok": True, "problem_type": problem_type, "sampling": sampling, "target": target,
            "label_mapping": label_mapping, "classes": classes,
            "models": results, "comparison": comparison, "primary_scoring": primary_scoring,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Orange-style: Predictions
# ══════════════════════════════════════════════════════════════════════════════
def build_predictions_table(
    fitted_models: dict, df, target, problem_type, label_mapping=None, max_rows: int = 500,
) -> "pd.DataFrame":
    """Per-row predictions for each fitted model + probabilities/errors beside the actuals."""
    sample = df.iloc[:max_rows].copy()
    out = pd.DataFrame(index=range(len(sample)))
    has_target = bool(target) and target in sample.columns
    if has_target:
        out["actual"] = sample[target].values
    X = sample.drop(columns=[target]) if has_target else sample

    for name, pipe in fitted_models.items():
        try:
            preds = pipe.predict(X)
            if problem_type == "classification":
                if label_mapping:
                    out[f"{name} (pred)"] = [label_mapping.get(int(p), p) for p in preds]
                else:
                    out[f"{name} (pred)"] = preds
                if hasattr(pipe, "predict_proba"):
                    proba = np.asarray(pipe.predict_proba(X))
                    for ci in range(proba.shape[1]):
                        cls = label_mapping.get(int(ci), ci) if label_mapping else ci
                        out[f"{name} p({cls})"] = np.round(proba[:, ci], 3)
            else:
                out[f"{name} (pred)"] = np.round(np.asarray(preds, dtype=float), 4)
                if has_target:
                    err = pd.to_numeric(sample[target], errors="coerce").to_numpy(dtype=float) - preds
                    out[f"{name} (error)"] = np.round(err, 4)
        except Exception as e:
            out[f"{name} (pred)"] = f"error: {e}"
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Orange-style: Rank (feature scoring)
# ══════════════════════════════════════════════════════════════════════════════
RANK_METHODS = {
    "classification": ["Information Gain", "ANOVA", "Chi²", "Gini (tree)", "Random Forest"],
    "regression": ["Univariate Regression", "Mutual Information", "Random Forest"],
}


def _score_features(X, y, discrete, problem_type, method):
    discrete = np.asarray(discrete, dtype=bool)
    if problem_type == "classification":
        if method == "ANOVA":
            from sklearn.feature_selection import f_classif
            s, _ = f_classif(X, y)
        elif method == "Chi²":
            from sklearn.feature_selection import chi2
            from sklearn.preprocessing import MinMaxScaler
            s, _ = chi2(MinMaxScaler().fit_transform(X), y)
        elif method == "Gini (tree)":
            from sklearn.tree import DecisionTreeClassifier
            s = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X, y).feature_importances_
        elif method == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            s = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y).feature_importances_
        else:  # Information Gain
            from sklearn.feature_selection import mutual_info_classif
            s = mutual_info_classif(X, y, discrete_features=discrete, random_state=RANDOM_STATE)
    else:
        if method == "Mutual Information":
            from sklearn.feature_selection import mutual_info_regression
            s = mutual_info_regression(X, y, discrete_features=discrete, random_state=RANDOM_STATE)
        elif method == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            s = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y).feature_importances_
        else:  # Univariate Regression (F-score)
            from sklearn.feature_selection import f_regression
            s, _ = f_regression(X, y)
    return np.nan_to_num(np.asarray(s, dtype=float), nan=0.0)


def rank_features(df, target, problem_type, method, profile, top_n: int = 30) -> list[tuple[str, float]]:
    """Score original feature columns against the target, ranked descending."""
    try:
        from sklearn.preprocessing import LabelEncoder
        cols_profile = profile.get("columns", {})
        feats = []
        for c in df.columns:
            if c == target:
                continue
            info = cols_profile.get(c, {})
            cls = info.get("dtype_class")
            if cls in ("identifier", "datetime"):
                continue
            # Skip ID-like near-unique *non-numeric* columns (they overfit scores like mutual info);
            # genuine continuous numeric features are naturally near-unique and must be kept.
            if cls != "numerical" and info.get("unique_pct", 0.0) >= 0.9:
                continue
            feats.append(c)
        if not feats:
            return []

        work = df.dropna(subset=[target]).copy()
        X_cols, names, discrete = [], [], []
        for c in feats:
            s = work[c]
            cls = cols_profile.get(c, {}).get("dtype_class")
            if cls == "numerical":
                col = pd.to_numeric(s, errors="coerce")
                col = col.fillna(col.median())
                X_cols.append(col.to_numpy(dtype=float)); discrete.append(False)
            else:
                col = LabelEncoder().fit_transform(s.astype(str).fillna("nan"))
                X_cols.append(col.astype(float)); discrete.append(True)
            names.append(c)
        X = np.column_stack(X_cols)

        if problem_type == "classification":
            y = LabelEncoder().fit_transform(work[target].astype(str))
        else:
            y = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            X, y = X[m], y[m]

        scores = _score_features(X, y, discrete, problem_type, method)
        pairs = list(zip(names, scores.tolist()))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Orange-style: model viewers (Tree Viewer, Nomogram)
# ══════════════════════════════════════════════════════════════════════════════
def export_tree_dot(fitted_pipeline, class_names=None, max_depth: int = 3) -> str | None:
    """Graphviz DOT for a single tree from a fitted Pipeline (DecisionTree or first ensemble tree)."""
    try:
        from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
        model = fitted_pipeline.named_steps.get("model")
        pre = fitted_pipeline.named_steps.get("pre")
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = None

        tree = None
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            tree = model
        elif hasattr(model, "estimators_"):
            est0 = model.estimators_[0]
            if isinstance(est0, np.ndarray):   # GradientBoosting: 2-D array of trees
                est0 = est0[0]
            tree = est0
        if tree is None:
            return None

        cn = [str(c) for c in class_names] if (class_names and isinstance(tree, DecisionTreeClassifier)) else None
        return export_graphviz(
            tree, out_file=None, feature_names=feat_names, class_names=cn,
            filled=True, rounded=True, max_depth=max_depth, impurity=False,
            proportion=True, special_characters=True,
        )
    except Exception:
        return None


def nomogram_data(fitted_pipeline, top_n: int = 15) -> list[tuple[str, float]] | None:
    """Per-feature coefficients (contributions) for a fitted Logistic Regression pipeline."""
    try:
        model = fitted_pipeline.named_steps.get("model")
        pre = fitted_pipeline.named_steps.get("pre")
        if not hasattr(model, "coef_"):
            return None
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(len(coef))]
        if len(names) != len(coef):
            names = [f"f{i}" for i in range(len(coef))]
        pairs = sorted(zip(names, coef.tolist()), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        return [(str(n), float(v)) for n, v in pairs]
    except Exception:
        return None
