"""modules/notebook_generator.py — Jupyter .ipynb generation."""
from __future__ import annotations
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import json


def generate_preprocessing_cell(profile: dict, pipeline_params: dict) -> nbformat.NotebookNode:
    """Generate the preprocessing code cell."""
    preprocessing_recs = profile.get("preprocessing_recommendations", {})

    num_cols = [
        col for col, v in preprocessing_recs.items()
        if not v.get("drop", False) and
        profile["columns"].get(col, {}).get("dtype_class") == "numerical"
    ]
    cat_ohe = [
        col for col, v in preprocessing_recs.items()
        if not v.get("drop", False) and
        v.get("encoding") == "one_hot"
    ]

    code = f"""from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numerical columns: {num_cols[:5]}
# Categorical columns (one-hot): {cat_ohe[:5]}

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

numerical_cols = {num_cols}
categorical_cols = {cat_ohe}

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
])
"""
    return new_code_cell(code)


def generate_model_cell(profile: dict) -> nbformat.NotebookNode:
    """Generate model training cell for top recommended model."""
    ml = profile.get("ml_readiness", {})
    models = ml.get("recommended_models", [])

    if not models:
        return new_code_cell("# No model recommendations available")

    top_model = models[0]
    code = f"""# Top recommended model: {top_model['name']}
# Reason: {top_model['reason']}

{top_model.get('starter_code', '# starter code not available')}
"""
    return new_code_cell(code)


def generate_evaluation_cell(problem_type: str) -> nbformat.NotebookNode:
    """Generate evaluation metrics cell."""
    if problem_type == "classification":
        code = """from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

if y_prob is not None:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
"""
    elif problem_type == "regression":
        code = """from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
"""
    else:
        code = """from sklearn.metrics import silhouette_score
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
"""

    return new_code_cell(code)


def generate_notebook(profile: dict, pipeline_params: dict = None) -> nbformat.NotebookNode:
    """Build and return a complete Jupyter notebook."""
    nb = new_notebook()
    cells = []
    ml = profile.get("ml_readiness", {})
    target = ml.get("target_column")
    problem_type = ml.get("problem_type", "classification")
    file_name = profile.get("meta", {}).get("file_name", "dataset.csv")

    # Title
    cells.append(new_markdown_cell(
        f"# DataIQ — Generated ML Pipeline\n"
        f"*Auto-generated for `{file_name}` by DataIQ*\n\n"
        f"**Detected problem:** {problem_type.title()} | "
        f"**Target:** {target or 'None'}"
    ))

    # Imports
    cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_squared_error, r2_score
"""))

    # Load data
    cells.append(new_code_cell(
        f"df = pd.read_csv('{file_name}')  # Update path as needed\n"
        f"print(f'Shape: {{df.shape}}')\n"
        f"df.head()"
    ))

    # EDA quick summary
    cells.append(new_code_cell("""# Quick EDA
print(df.dtypes)
print("\\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
df.describe()
"""))

    # Preprocessing
    cells.append(generate_preprocessing_cell(profile, pipeline_params or {}))

    # Train/test split
    if target:
        cells.append(new_code_cell(
            f"X = df.drop(columns=['{target}'])\n"
            f"y = df['{target}']\n"
            f"X_train, X_test, y_train, y_test = train_test_split(\n"
            f"    X, y, test_size=0.2, random_state=42\n"
            f")\n"
            f"print(f'Train: {{X_train.shape}}, Test: {{X_test.shape}}')"
        ))
    else:
        cells.append(new_code_cell(
            "# No target column detected — preparing unsupervised learning\n"
            "X = df.copy()\n"
            "X_scaled = preprocessor.fit_transform(X)"
        ))

    # Model
    cells.append(generate_model_cell(profile))

    # Evaluation
    cells.append(generate_evaluation_cell(problem_type))

    nb.cells = cells
    return nb


def notebook_to_bytes(nb: nbformat.NotebookNode) -> bytes:
    """Serialize notebook to bytes for download."""
    return nbformat.writes(nb).encode("utf-8")
