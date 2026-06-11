"""modules/ml_advisor.py — Problem type detection and model recommendations."""
from __future__ import annotations
import pandas as pd


# ── Problem type detection ────────────────────────────────────────────────────
def detect_problem_type(
    df: pd.DataFrame, target_col: str | None, columns_profile: dict
) -> tuple[str, float]:
    """Returns (problem_type, confidence). problem_type ∈ classification|regression|clustering|timeseries."""

    if target_col is None:
        return "clustering", 0.70

    target_profile = columns_profile.get(target_col, {})
    has_datetime = any(
        v.get("dtype_class") == "datetime" for v in columns_profile.values()
    )

    if has_datetime:
        return "timeseries", 0.75

    dtype_class = target_profile.get("dtype_class", "unknown")
    unique_count = target_profile.get("unique_count", 0)

    if dtype_class in ("categorical", "boolean"):
        if unique_count == 2:
            return "classification", 0.95
        elif 2 < unique_count <= 20:
            return "classification", 0.85
        else:
            return "classification", 0.60

    if dtype_class == "numerical":
        return "regression", 0.90

    return "classification", 0.50


# ── Readiness checks ──────────────────────────────────────────────────────────
def check_ml_readiness(
    df: pd.DataFrame,
    target_col: str | None,
    columns_profile: dict,
    problem_type: str,
) -> tuple[list[str], list[str]]:
    """Returns (blockers, warnings)."""
    blockers = []
    warnings = []
    n_rows = len(df)

    # Blockers
    if n_rows < 50:
        blockers.append(f"Dataset has only {n_rows} rows — minimum 50 required for ML.")

    if target_col:
        target_null = columns_profile.get(target_col, {}).get("null_pct", 0)
        if target_null > 0.5:
            blockers.append(
                f"Target column '{target_col}' has {target_null*100:.0f}% missing values — "
                "must be < 50% for ML."
            )

    non_id_cols = [
        c for c, v in columns_profile.items()
        if v.get("dtype_class") != "identifier" and c != target_col
    ]
    if len(non_id_cols) == 0:
        blockers.append("All columns are identifiers — no features available for ML.")

    if target_col and len(non_id_cols) < 1:
        blockers.append("Only 1 feature column remaining after removing target — need at least 2.")

    # Warnings
    if 50 <= n_rows < 200:
        warnings.append(f"Small dataset ({n_rows} rows) — expect high variance in model results.")

    high_missing = [
        c for c, v in columns_profile.items()
        if v.get("null_pct", 0) > 0.20
    ]
    if len(high_missing) > len(columns_profile) * 0.30:
        warnings.append(
            f"{len(high_missing)} columns have >20% missing values — "
            "imputation strategy is critical."
        )

    low_var = sum(1 for c, v in columns_profile.items() if v.get("dtype_class") == "numerical")
    # (variance check done dynamically)

    if target_col:
        cat_stats = columns_profile.get(target_col, {})
        if cat_stats.get("dtype_class") == "categorical":
            # imbalance check via statistics
            pass  # done in ml_readiness builder

    return blockers, warnings


# ── Model recommendations ─────────────────────────────────────────────────────
_CLASSIFICATION_MODELS = [
    {
        "name": "Logistic Regression",
        "priority": 1,
        "best_for": "Linearly separable data, interpretability required",
        "not_suitable_when": "Complex nonlinear relationships exist",
        "reason": "Fast baseline, interpretable coefficients, works well with scaled features",
        "starter_code": """from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))""",
    },
    {
        "name": "Random Forest Classifier",
        "priority": 2,
        "best_for": "Tabular data, handles missing values, non-linear patterns",
        "not_suitable_when": "Very high-dimensional sparse data",
        "reason": "Robust ensemble method, handles outliers and class imbalance well, built-in feature importance",
        "starter_code": """from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
model.fit(X_train, y_train)""",
    },
    {
        "name": "XGBoost Classifier",
        "priority": 3,
        "best_for": "Tabular competitions, imbalanced data (scale_pos_weight), missing values",
        "not_suitable_when": "Very small datasets (<200 rows)",
        "reason": "State-of-the-art gradient boosting, handles missing values natively, excellent on tabular data",
        "starter_code": """from xgboost import XGBClassifier

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=300, learning_rate=0.05,
                                  use_label_encoder=False, eval_metric='logloss',
                                  random_state=42))
])
model.fit(X_train, y_train)""",
    },
    {
        "name": "LightGBM Classifier",
        "priority": 4,
        "best_for": "Large datasets, fast training, categorical features",
        "not_suitable_when": "Very small datasets",
        "reason": "Fastest gradient boosting implementation, handles high cardinality categoricals natively",
        "starter_code": """from lightgbm import LGBMClassifier

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(n_estimators=300, random_state=42, verbose=-1))
])
model.fit(X_train, y_train)""",
    },
]

_REGRESSION_MODELS = [
    {
        "name": "Linear Regression",
        "priority": 1,
        "best_for": "Linear relationships, interpretability",
        "not_suitable_when": "Non-linear data, many outliers",
        "reason": "Fast interpretable baseline — always run first to establish a benchmark",
        "starter_code": """from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")
print(f"R²: {r2_score(y_test, preds):.4f}")""",
    },
    {
        "name": "Ridge Regression",
        "priority": 2,
        "best_for": "Multicollinear features, many correlated predictors",
        "not_suitable_when": "Feature selection needed",
        "reason": "L2 regularisation stabilises coefficients when features are correlated",
        "starter_code": """from sklearn.linear_model import Ridge

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
model.fit(X_train, y_train)""",
    },
    {
        "name": "Random Forest Regressor",
        "priority": 3,
        "best_for": "Non-linear relationships, outlier-robust predictions",
        "not_suitable_when": "Requires extrapolation beyond training range",
        "reason": "Ensemble of trees — handles non-linearity and interaction effects automatically",
        "starter_code": """from sklearn.ensemble import RandomForestRegressor

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])
model.fit(X_train, y_train)""",
    },
    {
        "name": "XGBoost Regressor",
        "priority": 4,
        "best_for": "Most tabular regression tasks, feature interactions",
        "not_suitable_when": "Very small datasets (<200 rows)",
        "reason": "Best-in-class gradient boosting for structured data",
        "starter_code": """from xgboost import XGBRegressor

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
])
model.fit(X_train, y_train)""",
    },
]

_CLUSTERING_MODELS = [
    {
        "name": "K-Means Clustering",
        "priority": 1,
        "best_for": "Spherical clusters, when number of clusters is known",
        "not_suitable_when": "Irregular cluster shapes, noise-heavy data",
        "reason": "Simple and scalable — use Elbow method to find optimal K",
        "starter_code": """from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Find optimal k using elbow method
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit with chosen k
model = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")""",
    },
    {
        "name": "DBSCAN",
        "priority": 2,
        "best_for": "Arbitrary cluster shapes, noise/outlier detection",
        "not_suitable_when": "High-dimensional data, varying density clusters",
        "reason": "Discovers clusters of arbitrary shape and automatically identifies outliers as noise",
        "starter_code": """from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters found: {n_clusters}, Noise points: {(labels == -1).sum()}")""",
    },
]

_TIMESERIES_MODELS = [
    {
        "name": "ARIMA",
        "priority": 1,
        "best_for": "Univariate time series, stationary or near-stationary data",
        "not_suitable_when": "Multivariate inputs, very long series",
        "reason": "Classic statistical approach — interpretable parameters, good for forecasting",
        "starter_code": """from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(1, 1, 1))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
print(fitted.summary())""",
    },
    {
        "name": "Prophet (by Meta)",
        "priority": 2,
        "best_for": "Daily/weekly/yearly seasonality, missing data, holidays",
        "not_suitable_when": "High-frequency data without clear seasonality",
        "reason": "Handles multiple seasonalities and missing data automatically — great for business forecasting",
        "starter_code": """# pip install prophet
from prophet import Prophet

df_prophet = df[['ds', 'y']].copy()  # rename date/value columns
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df_prophet)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)""",
    },
]


def get_model_recommendations(problem_type: str) -> list[dict]:
    """Return ranked list of model recommendations for the detected problem type."""
    mapping = {
        "classification": _CLASSIFICATION_MODELS,
        "regression":     _REGRESSION_MODELS,
        "clustering":     _CLUSTERING_MODELS,
        "timeseries":     _TIMESERIES_MODELS,
    }
    return mapping.get(problem_type, _CLASSIFICATION_MODELS)


def assess_ml_readiness(
    df: pd.DataFrame,
    target_col: str | None,
    columns_profile: dict,
    profile: dict,
) -> dict:
    """Full ML readiness assessment."""
    problem_type, confidence = detect_problem_type(df, target_col, columns_profile)
    blockers, warnings = check_ml_readiness(df, target_col, columns_profile, problem_type)
    recommendations = get_model_recommendations(problem_type)

    return {
        "problem_type": problem_type,
        "problem_confidence": round(confidence, 3),
        "target_column": target_col,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_models": recommendations,
        "is_ready": len(blockers) == 0,
    }
