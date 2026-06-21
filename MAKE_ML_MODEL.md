# 🛠 Make ML Model — User Guide (Orange-style)

The **Make ML Model** page (sidebar → `10 🛠 Make ML Model`) is an **Orange-inspired ML studio**.
Instead of training one model at a time, you wire **several learners** into a **Test & Score** step and
inspect them through **Predictions**, **Evaluation curves**, **Rank**, and **model viewers** — the same
workflow Orange Data Mining is built around:

```
📁 Data → ⚙️ Preprocess → 🧠 Models → 🧪 Test & Score → 🔮 Predictions → 📊 Evaluate → 📈 Rank
```

---

## ⚡ The fast path

1. **Load data** (sidebar upload or a demo) and, optionally, run **06 ⚙️ Preprocessing** → "Clean Dataset".
2. Open **10 🛠 Make ML Model**. Confirm the **Problem type** and **Target** at the top.
3. **🧠 Models** → tick the learners you want to compare (e.g. Logistic Regression, Random Forest,
   Gradient Boosting, and a **Constant** baseline).
4. **🧪 Test & Score** → choose a sampling method → **Run**. Read the comparison table.
5. **📊 Evaluate** → ROC / Lift / Calibration / Confusion Matrix + Tree Viewer / Nomogram.
6. **🔮 Predictions** → per-row predictions & probabilities; download the trained model `.pkl`.

---

## The tabs (Orange widgets)

### 🧠 Models — the learner bench
Multi-select the models to compare (Orange's "model column"). Each gets an expander with its
**hyperparameters** and an **Auto-tune** button (RandomizedSearchCV that writes the best params back into
the controls). Available learners:

| Problem | Learners |
|--------|----------|
| **Classification** | Logistic Regression, Random Forest, XGBoost, LightGBM, **Decision Tree**, **k-NN**, **SVM**, **Naive Bayes**, **Gradient Boosting**, **AdaBoost**, **Neural Network**, **Constant (baseline)** |
| **Regression** | Linear, Ridge, Random Forest, XGBoost, **Decision Tree**, **k-NN**, **SVM**, **Gradient Boosting**, **AdaBoost**, **Neural Network**, **Constant (baseline)** |
| **Clustering** | K-Means, DBSCAN, **Hierarchical** |
| **Time series** | ARIMA (built-in), Prophet (optional) |

> The **Constant** baseline predicts the majority class / mean — always include it to see how much your
> real models actually beat "no model."

### 🧪 Test & Score
Evaluate every selected learner under one **sampling** scheme and compare them in a single table:

- **Sampling:** Cross-validation (k folds) · Random sampling (test % + repeats) · Leave-one-out · Test on train data.
- **Classification metrics:** AUC, CA (accuracy), F1, Precision, Recall, Specificity, LogLoss, MCC, Time.
- **Regression metrics:** R², RMSE, MAE, MAPE, MSE, CVRMSE, Time.
- Best value per column is highlighted. For cross-validation, a **model comparison** matrix shows the
  approximate probability each model scores higher than another (paired t-test on fold scores).

Predictions are pooled out-of-fold, so the metrics and the Evaluate curves come from held-out data — not
the training set.

### 🔮 Predictions
Per-row table with each model's **predicted class + per-class probabilities** (classification) or
**predicted value + error** (regression), beside the actual target. Sortable, **downloadable as CSV**, and
you can **upload a new dataset** (same columns) to score it.

### 📊 Evaluate
- **Confusion Matrix** (counts or proportions) per model.
- **ROC Analysis** — all models overlaid with AUC (binary targets).
- **Performance Curve** — switch between **Lift**, **Cumulative Gains**, and **Precision-Recall**.
- **Calibration Plot** — how trustworthy each model's probabilities are.
- **Model viewers:** **Tree Viewer** (renders a decision tree for tree-based models) and **Nomogram**
  (per-feature log-odds contributions for Logistic Regression).
- **Download** any fitted model as a `.pkl` pipeline (preprocessing + model).

### 📈 Rank — feature scoring
Score features against the target, independent of any trained model:

- **Classification:** Information Gain, ANOVA, Chi², Gini (tree), Random Forest importance.
- **Regression:** Univariate Regression (F-score), Mutual Information, Random Forest importance.

ID-like / near-unique columns are excluded automatically so they don't dominate the scores.

### 📘 Guide & AI Chat
A dataset-specific modeling guide (no API key needed) plus a multi-turn Groq chat that knows your dataset
profile and selected models.

### Clustering & Time series
- **Clustering:** pick K-Means / DBSCAN / Hierarchical → **Silhouette Plot** (per-sample), 2-D PCA scatter,
  and an elbow plot (K-Means) to choose *k*.
- **Time series:** pick the date + value columns, ARIMA `p,d,q` (or Prophet), and a forecast horizon.

---

## Reusing a downloaded model

The `.pkl` is a complete fitted scikit-learn `Pipeline` (preprocessing **+** model):

```python
import pickle, pandas as pd

with open("random_forest_classifier.pkl", "rb") as f:
    model = pickle.load(f)

new = pd.read_csv("new_rows.csv")                  # same columns as training, minus the target
preds = model.predict(new.drop(columns=["Survived"], errors="ignore"))
```

For **classification**, predictions are integer class indices in the order shown on the confusion-matrix
axes; `model.predict_proba(...)` gives per-class probabilities.

---

## Tips & gotchas

- **Demo datasets are randomly generated** — they have no real signal, so every model (including the
  baseline) scores near chance. Use them to learn the workflow; use your own data for real scores.
- **Always include the Constant baseline** — if your fancy model can't beat it, the features aren't predictive.
- **Cross-validation before trusting a number** — a single split can be lucky.
- **Leave-one-out** is disabled above 1,000 rows (too slow) — use cross-validation.
- **SVM / Neural Network** are slower than the tree models — fine on moderate data, patient on large.
- ROC / Lift / Calibration are shown for **binary** classification; multiclass shows the confusion matrix.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| "Select models in the Models tab first" | Tick at least one learner in **🧠 Models**. |
| "Run Test & Score first" (Predictions/Evaluate) | Those tabs read the fitted models from Test & Score. |
| "No usable feature columns" | All columns dropped/identifiers — un-drop some on **06 Preprocessing**. |
| Tree Viewer / Nomogram missing | Add a tree-based model (tree view) or Logistic Regression (nomogram) in Models. |
| Chat asks for a key | Enter your Groq key in the sidebar; the written **Guide** works without it. |
| Prophet "not installed" | Use **ARIMA**, or `pip install prophet` in the `cv_conda` env. |
