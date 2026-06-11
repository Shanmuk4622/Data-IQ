"""pages/07_🤖_ML_Recommendation.py — Model recommendations and starter code."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🤖 ML Recommendations")

df = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

ml = profile.get("ml_readiness", {})
target = st.session_state.get("target_col") or ml.get("target_column")
problem_type = ml.get("problem_type", "unknown")
confidence = ml.get("problem_confidence", 0)
is_ready = ml.get("is_ready", False)
blockers = ml.get("blockers", [])
warnings = ml.get("warnings", [])
models = ml.get("recommended_models", [])

# ── Problem Type Banner ───────────────────────────────────────────────────────
type_colors = {
    "classification": "#8B5CF6",
    "regression":     "#06B6D4",
    "clustering":     "#10B981",
    "timeseries":     "#F59E0B",
}
type_icons = {
    "classification": "🏷",
    "regression":     "📉",
    "clustering":     "🔮",
    "timeseries":     "📅",
}
color = type_colors.get(problem_type, "#94A3B8")
icon = type_icons.get(problem_type, "🤖")

st.markdown(
    f"""<div class="metric-card" style="border-color:{color}55;text-align:center;">
    <div style="font-size:2rem;">{icon}</div>
    <div style="font-size:1.4rem;font-weight:700;color:{color};margin:0.3rem 0;">
        {problem_type.replace("timeseries", "Time Series").title()} Problem
    </div>
    <div style="color:#94A3B8;">
        Confidence: {confidence*100:.0f}% | Target: <strong>{target or "None (unsupervised)"}</strong>
    </div>
    </div>""",
    unsafe_allow_html=True,
)

st.divider()

# ── Blockers and Warnings ─────────────────────────────────────────────────────
if blockers:
    st.markdown("## 🚫 ML Blockers (Must Fix)")
    for b in blockers:
        st.markdown(f'<div class="error-box">❌ {b}</div>', unsafe_allow_html=True)
    st.divider()

if warnings:
    st.markdown("## ⚠️ Warnings")
    for w in warnings:
        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
    st.divider()

if is_ready:
    st.markdown('<div class="success-box">✅ Dataset is ML-ready — no blockers detected.</div>', unsafe_allow_html=True)
    st.divider()

# ── Model Recommendations ─────────────────────────────────────────────────────
st.markdown("## 🏆 Recommended Models")
st.caption(f"Ranked by suitability for your {problem_type} problem")

for i, model in enumerate(models):
    priority_badge = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(i, 4)]
    with st.expander(f"{priority_badge} **{model['name']}**", expanded=(i == 0)):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**Why this model?** {model['reason']}")
            st.markdown(f"**Best for:** {model['best_for']}")
            st.markdown(f"**Not suitable when:** {model['not_suitable_when']}")
        with c2:
            st.markdown(
                f"""<div class="model-card">
                <div class="model-name">{model['name']}</div>
                <div class="model-reason">Priority #{model.get('priority', i+1)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        
        # Starter code with actual column names
        if model.get("starter_code"):
            st.markdown("**Starter Code:**")
            code = model["starter_code"]
            # Inject actual target column
            if target:
                code = f"# Target column: '{target}'\n" + code
            st.code(code, language="python")

st.divider()

# ── Cross-validation template ─────────────────────────────────────────────────
st.markdown("## 🔄 Cross-Validation Template")
cv_code = """from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# Use StratifiedKFold for classification, KFold for regression
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X, y,
    cv=cv,
    scoring='accuracy',  # Change to 'roc_auc', 'f1', 'neg_rmse' etc.
    n_jobs=-1,
)
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
"""
st.code(cv_code, language="python")

# ── Hyperparameter tuning template ────────────────────────────────────────────
st.markdown("## 🎛 Hyperparameter Tuning Template")
hp_code = """from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
}

search = RandomizedSearchCV(
    model, param_grid,
    n_iter=20, cv=5, scoring='accuracy',
    random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV score: {search.best_score_:.4f}")
"""
st.code(hp_code, language="python")
