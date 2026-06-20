"""pages/10_🛠_Make_ML_Model.py — Train real ML models on the dataset + AI modeling guide/chat."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
import streamlit as st

from modules.model_trainer import (
    list_models, get_model_spec,
    train_supervised, tune_hyperparameters, run_leaderboard,
    train_clustering, train_timeseries, generate_modeling_guide, model_to_bytes,
)
from utils.chart_factory import (
    make_confusion_matrix, make_roc_curve, make_pred_vs_actual, make_residual_plot,
    make_cluster_scatter, make_elbow_plot, make_forecast_plot, make_cv_scores, make_bar_chart,
)


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🛠 Make ML Model")
st.caption("Configure, train, tune, and compare real ML models on your dataset — then download the fitted pipeline.")

df_raw = st.session_state.get("df_raw")
profile = st.session_state.get("profile")

if df_raw is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

health = st.session_state.get("health_score") or profile.get("health_score", {})
columns_profile = profile.get("columns", {})
ml = profile.get("ml_readiness", {})
groq_key = st.session_state.get("groq_api_key", "")


# ── Small helpers ─────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="font-size:1.3rem;">{value}</div>{sub_html}</div>'
    )


def render_hyperparams(spec: dict, key_prefix: str) -> dict:
    """Render dynamic hyperparameter widgets from a registry schema; return chosen values."""
    params = spec.get("hyperparams", [])
    if not params:
        st.caption("This model has no tunable hyperparameters.")
        return {}
    hp = {}
    cols = st.columns(min(3, len(params)))
    for i, p in enumerate(params):
        c = cols[i % len(cols)]
        key = f"hp_{key_prefix}_{p['name']}"
        if p["type"] == "int":
            hp[p["name"]] = c.number_input(
                p["name"], min_value=int(p["min"]), max_value=int(p["max"]),
                value=int(p["default"]), step=int(p.get("step", 1)),
                key=key, help=p.get("help"),
            )
        elif p["type"] == "float":
            hp[p["name"]] = c.number_input(
                p["name"], min_value=float(p["min"]), max_value=float(p["max"]),
                value=float(p["default"]), step=float(p.get("step", 0.01)),
                key=key, help=p.get("help"),
            )
        elif p["type"] == "select":
            opts = p["options"]
            hp[p["name"]] = c.selectbox(p["name"], opts, index=opts.index(p["default"]), key=key)
    return hp


def render_supervised_results(res: dict, problem_type: str):
    if not res.get("ok"):
        st.error(f"Training failed: {res.get('error')}")
        return

    m = res["metrics"]
    st.markdown("#### Results")
    if problem_type == "classification":
        cards = [
            ("Accuracy", f"{m['accuracy']*100:.2f}%"),
            ("F1 (weighted)", f"{m['f1_weighted']:.3f}"),
            ("Precision", f"{m['precision_weighted']:.3f}"),
            ("Recall", f"{m['recall_weighted']:.3f}"),
            ("ROC-AUC", f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "—"),
        ]
    else:
        cards = [
            ("R²", f"{m['r2']:.4f}"),
            ("RMSE", f"{m['rmse']:.4f}"),
            ("MAE", f"{m['mae']:.4f}"),
            ("MAPE", f"{m['mape']:.1f}%"),
        ]
    cols = st.columns(len(cards))
    for c, (label, val) in zip(cols, cards):
        c.markdown(metric_card(label, val), unsafe_allow_html=True)

    st.caption(
        f"Trained **{res['model_name']}** on {res['n_train']:,} rows · "
        f"tested on {res['n_test']:,} · {res['train_time']}s"
    )

    # Diagnostic plots
    if problem_type == "classification":
        cm = m["confusion_matrix"]
        lm = res.get("label_mapping") or {}
        labels = [lm.get(i, str(i)) for i in range(len(cm))]
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(make_confusion_matrix(cm, labels), use_container_width=True)
        with cc2:
            if res.get("y_prob") is not None and len(cm) == 2:
                yprob = np.asarray(res["y_prob"])
                st.plotly_chart(
                    make_roc_curve(res["y_test"], yprob[:, 1], auc=m.get("roc_auc")),
                    use_container_width=True,
                )
            else:
                st.info("ROC curve shown for binary classification only.")
    else:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(make_pred_vs_actual(res["y_test"], res["y_pred"]), use_container_width=True)
        with cc2:
            st.plotly_chart(make_residual_plot(res["y_test"], res["y_pred"]), use_container_width=True)

    # Feature importance
    fi = res.get("feature_importance")
    if fi:
        top = fi[:20][::-1]
        st.plotly_chart(
            make_bar_chart([n for n, _ in top], [v for _, v in top],
                           title="Top Feature Importance", horizontal=True),
            use_container_width=True,
        )

    # Cross-validation
    if res.get("cv_scores"):
        st.plotly_chart(
            make_cv_scores(res["cv_scores"], res.get("cv_scoring", "score")),
            use_container_width=True,
        )
        st.markdown(
            f'<div class="info-box">Cross-validation ({res.get("cv_scoring")}): '
            f'<strong>{res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}</strong> across folds.</div>',
            unsafe_allow_html=True,
        )

    # Downloads
    st.markdown("#### Export")
    d1, d2 = st.columns(2)
    with d1:
        try:
            pkl = model_to_bytes(res["pipeline"])
            st.download_button(
                "📥 Download trained model (.pkl)", data=pkl,
                file_name=f"{res['model_name'].replace(' ', '_').lower()}.pkl",
                mime="application/octet-stream", use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Model export unavailable: {e}")
    with d2:
        report = {
            "model_name": res["model_name"], "problem_type": res["problem_type"],
            "target": res["target"], "metrics": res["metrics"],
            "cv_mean": res.get("cv_mean"), "cv_std": res.get("cv_std"),
            "train_time": res["train_time"], "n_train": res["n_train"], "n_test": res["n_test"],
            "feature_importance": res.get("feature_importance"), "options": res.get("options"),
        }
        st.download_button(
            "📥 Download training report (JSON)",
            data=json.dumps(report, indent=2, default=str).encode("utf-8"),
            file_name="training_report.json", mime="application/json", use_container_width=True,
        )


# ── Data source + problem framing ─────────────────────────────────────────────
df_clean = st.session_state.get("df_clean")
src1, src2 = st.columns([1, 1])
with src1:
    use_clean = False
    if df_clean is not None:
        use_clean = st.checkbox("Use cleaned dataset (from Preprocessing)", value=True)
data = df_clean if (df_clean is not None and use_clean) else df_raw
st.caption(f"Training on: **{'cleaned' if (df_clean is not None and use_clean) else 'raw'} dataset** "
           f"({len(data):,} rows × {data.shape[1]} cols)")

detected_pt = ml.get("problem_type", "classification")
pt_options = ["classification", "regression", "clustering", "timeseries"]
pt_labels = {"classification": "Classification", "regression": "Regression",
             "clustering": "Clustering", "timeseries": "Time Series"}

f1, f2 = st.columns([1, 1])
with f1:
    problem_type = st.selectbox(
        "Problem type", pt_options,
        index=pt_options.index(detected_pt) if detected_pt in pt_options else 0,
        format_func=lambda x: pt_labels[x],
        help="Auto-detected from your data — override if needed.",
    )
with f2:
    target_opts = ["None (unsupervised)"] + list(data.columns)
    cur_target = st.session_state.get("target_col")
    t_default = cur_target if cur_target in data.columns else "None (unsupervised)"
    target_sel = st.selectbox(
        "Target column", target_opts,
        index=target_opts.index(t_default) if t_default in target_opts else 0,
    )
target = None if target_sel == "None (unsupervised)" else target_sel

user_choices = st.session_state.get("cleaning_choices", {})

# Reset stale results when the problem type changes
mc = st.session_state.get("model_config", {}) or {}
if mc.get("pt") != problem_type:
    st.session_state["training_results"] = None
    st.session_state["leaderboard_results"] = None
    mc["pt"] = problem_type
    st.session_state["model_config"] = mc

st.divider()

needs_target = problem_type in ("classification", "regression")


# ══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════════════
if problem_type in ("classification", "regression"):
    tab_train, tab_tune, tab_board, tab_guide = st.tabs(
        ["🎯 Train Model", "🎛 Tune", "🏆 Leaderboard", "📘 Guide & AI Chat"]
    )
else:
    tab_train, tab_guide = st.tabs(["🎯 Train Model", "📘 Guide & AI Chat"])
    tab_tune = tab_board = None


# ── Train tab ─────────────────────────────────────────────────────────────────
with tab_train:
    if problem_type in ("classification", "regression"):
        if target is None:
            st.warning("⚠️ Select a **target column** above to train a supervised model.")
        else:
            model_name = st.selectbox("Model", list_models(problem_type), key="train_model_select")
            spec = get_model_spec(problem_type, model_name)

            with st.expander("⚙️ Hyperparameters", expanded=True):
                tuned = st.session_state.get("tuned_params")
                if tuned and tuned.get("model") == model_name and tuned.get("problem_type") == problem_type:
                    if st.button("✨ Apply tuned hyperparameters"):
                        for p in spec.get("hyperparams", []):
                            if p["name"] in tuned["params"]:
                                v = tuned["params"][p["name"]]
                                if p["name"] == "max_depth" and v is None:
                                    v = 0
                                st.session_state[f"hp_{problem_type}_{model_name}_{p['name']}"] = v
                        st.rerun()
                hp = render_hyperparams(spec, f"{problem_type}_{model_name}")

            with st.expander("🎚 Training options", expanded=False):
                o1, o2, o3 = st.columns(3)
                test_size = o1.slider("Test size", 0.1, 0.4, 0.2, 0.05)
                random_state = int(o2.number_input("Random state", 0, 99999, 42))
                cv_folds = int(o3.selectbox("Cross-validation folds", [0, 3, 5, 10], index=0,
                                            help="0 = skip CV"))
                if problem_type == "classification":
                    i1, i2 = st.columns(2)
                    stratify = i1.checkbox("Stratified split", value=True)
                    imbalance = i2.selectbox(
                        "Class-imbalance handling", ["none", "class_weight", "smote"],
                        help="SMOTE oversamples the minority class on the training folds only.",
                    )
                else:
                    stratify, imbalance = True, "none"

            if st.button("🚀 Train Model", type="primary"):
                options = {
                    "test_size": test_size, "random_state": random_state,
                    "stratify": stratify, "cv_folds": cv_folds, "imbalance": imbalance,
                }
                with st.spinner(f"Training {model_name}…"):
                    res = train_supervised(data, target, problem_type, model_name, hp,
                                           options, profile, user_choices)
                st.session_state["training_results"] = res
                mc = st.session_state.get("model_config", {}) or {}
                mc["last_model"] = model_name
                st.session_state["model_config"] = mc
                if res.get("ok"):
                    st.session_state["trained_model"] = res["pipeline"]
                    st.success("Training complete!")

            res = st.session_state.get("training_results")
            if res and res.get("problem_type") in ("classification", "regression"):
                st.divider()
                render_supervised_results(res, problem_type)

    elif problem_type == "clustering":
        st.markdown("### Unsupervised clustering")
        model_name = st.selectbox("Algorithm", list_models("clustering"), key="cluster_model_select")
        spec = get_model_spec("clustering", model_name)
        with st.expander("⚙️ Hyperparameters", expanded=True):
            hp = render_hyperparams(spec, f"clustering_{model_name}")
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner(f"Running {model_name}…"):
                res = train_clustering(data, profile, model_name, hp, user_choices)
            st.session_state["training_results"] = res

        res = st.session_state.get("training_results")
        if res and "labels" in res:
            if not res.get("ok"):
                st.error(res.get("error"))
            else:
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.markdown(metric_card("Clusters", str(res["n_clusters"])), unsafe_allow_html=True)
                sil = res.get("silhouette")
                c2.markdown(metric_card("Silhouette", f"{sil:.3f}" if sil is not None else "—",
                                        "−1 to 1, higher is better"), unsafe_allow_html=True)
                c3.markdown(metric_card("Noise points", str(res.get("n_noise", 0))), unsafe_allow_html=True)
                if res.get("pca_x"):
                    st.plotly_chart(
                        make_cluster_scatter(res["pca_x"], res["pca_y"], res["labels"]),
                        use_container_width=True,
                    )
                if res.get("elbow_k"):
                    st.plotly_chart(
                        make_elbow_plot(res["elbow_k"], res["elbow_inertias"]),
                        use_container_width=True,
                    )
                    st.caption("Pick the *k* at the 'elbow' where inertia stops dropping sharply.")

    elif problem_type == "timeseries":
        st.markdown("### Time-series forecasting")
        dt_cols = list(profile.get("datetime_analysis", {}).keys())
        if not dt_cols:
            dt_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") == "datetime"]
        num_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") == "numerical"]

        if not dt_cols:
            st.info("No datetime column detected — time-series forecasting needs a date/time column.")
        elif not num_cols:
            st.info("No numerical column available to forecast.")
        else:
            t1, t2 = st.columns(2)
            date_col = t1.selectbox("Date column", dt_cols)
            value_col = t2.selectbox("Value to forecast", num_cols)
            engine = st.radio("Engine", ["arima", "prophet"], horizontal=True,
                              format_func=lambda x: "ARIMA (built-in)" if x == "arima" else "Prophet (optional)")
            order = (1, 1, 1)
            if engine == "arima":
                oc = st.columns(3)
                p = int(oc[0].number_input("p (AR)", 0, 5, 1))
                d = int(oc[1].number_input("d (diff)", 0, 2, 1))
                q = int(oc[2].number_input("q (MA)", 0, 5, 1))
                order = (p, d, q)
            horizon = st.slider("Forecast horizon (steps)", 7, 180, 30, 1)

            if st.button("🔮 Forecast", type="primary"):
                with st.spinner("Fitting model and forecasting…"):
                    res = train_timeseries(data, date_col, value_col, order, horizon, engine)
                st.session_state["training_results"] = res

            res = st.session_state.get("training_results")
            if res and ("forecast_y" in res or res.get("error")):
                if not res.get("ok"):
                    st.error(res.get("error"))
                else:
                    st.divider()
                    st.markdown(metric_card(res["metric_label"], str(res["metric_value"]),
                                            f"engine: {res['engine']}"), unsafe_allow_html=True)
                    st.plotly_chart(
                        make_forecast_plot(
                            res["history_x"], res["history_y"], res["forecast_x"], res["forecast_y"],
                            res.get("lower"), res.get("upper"),
                            title=f"{value_col} forecast (+{horizon} steps)",
                        ),
                        use_container_width=True,
                    )


# ── Tune tab ──────────────────────────────────────────────────────────────────
if tab_tune is not None:
    with tab_tune:
        st.markdown("### Randomized hyperparameter search")
        if target is None:
            st.warning("⚠️ Select a target column above to tune.")
        else:
            model_name_t = st.selectbox("Model to tune", list_models(problem_type), key="tune_model_select")
            spec_t = get_model_spec(problem_type, model_name_t)
            if not spec_t.get("param_distributions"):
                st.info(f"**{model_name_t}** has no tunable hyperparameters.")
            else:
                c1, c2 = st.columns(2)
                n_iter = int(c1.slider("Search iterations", 5, 50, 20, 5))
                cv_t = int(c2.selectbox("CV folds", [3, 5], index=0))
                if st.button("🎛 Run Search", type="primary"):
                    with st.spinner("Searching hyperparameter space (trains many models)…"):
                        out = tune_hyperparameters(data, target, problem_type, model_name_t,
                                                   profile, user_choices, n_iter, cv_t)
                    st.session_state["_tune_out"] = out
                    st.session_state["tuned_params"] = (
                        {"model": model_name_t, "problem_type": problem_type,
                         "params": out.get("best_params", {}), "score": out.get("best_score"),
                         "scoring": out.get("scoring")}
                        if out.get("ok") else None
                    )
                out = st.session_state.get("_tune_out")
                if out:
                    if out.get("ok"):
                        st.markdown(
                            f'<div class="success-box">Best <strong>{out["scoring"]}</strong> = '
                            f'<strong>{out["best_score"]:.4f}</strong> '
                            f'({out["n_iter"]} iterations · {out["elapsed"]}s)</div>',
                            unsafe_allow_html=True,
                        )
                        st.json(out["best_params"])
                        st.caption("Open the **Train** tab — an *Apply tuned hyperparameters* "
                                   "button now appears for this model.")
                    else:
                        st.error(out.get("error"))


# ── Leaderboard tab ───────────────────────────────────────────────────────────
if tab_board is not None:
    with tab_board:
        st.markdown("### Train all models and compare")
        st.caption("Trains every model for this problem type with default hyperparameters and ranks them.")
        if target is None:
            st.warning("⚠️ Select a target column above to run the leaderboard.")
        else:
            if st.button("🏁 Run Leaderboard", type="primary"):
                with st.spinner("Training all models…"):
                    rows = run_leaderboard(data, target, problem_type, profile, user_choices)
                st.session_state["leaderboard_results"] = rows

            rows = st.session_state.get("leaderboard_results")
            if rows:
                ok_rows = [r for r in rows if r.get("ok")]
                if ok_rows:
                    board = pd.DataFrame([
                        {k: v for k, v in r.items() if not k.startswith("_") and k != "ok"}
                        for r in ok_rows
                    ])
                    st.dataframe(board, use_container_width=True, hide_index=True)

                    metric_key = "Accuracy" if problem_type == "classification" else "R²"
                    labels = [r["model"] for r in ok_rows][::-1]
                    vals = [r[metric_key] for r in ok_rows][::-1]
                    st.plotly_chart(
                        make_bar_chart(labels, vals, title=f"{metric_key} by model", horizontal=True),
                        use_container_width=True,
                    )
                    best = ok_rows[0]
                    st.markdown(
                        f'<div class="success-box">🏆 Best model: <strong>{best["model"]}</strong> '
                        f'({metric_key} = {best[metric_key]})</div>',
                        unsafe_allow_html=True,
                    )
                for r in [r for r in rows if not r.get("ok")]:
                    st.caption(f"⚠️ {r['model']}: {r.get('error')}")


# ── Guide & AI Chat tab ───────────────────────────────────────────────────────
with tab_guide:
    g_col, c_col = st.columns([1, 1])

    with g_col:
        st.markdown("### 📘 Modeling guide")
        st.markdown(generate_modeling_guide(profile, target, problem_type))

    with c_col:
        st.markdown("### 💬 AI Chat")
        st.caption("Grounded in your dataset profile and current model choice.")
        if not groq_key:
            st.markdown(
                '<div class="warning-box">⚠️ Add your Groq API key in the sidebar to enable the chat. '
                'Free at <a href="https://console.groq.com" target="_blank">console.groq.com</a>.</div>',
                unsafe_allow_html=True,
            )

        history = st.session_state.get("model_chat_history", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        with st.form("model_chat_form", clear_on_submit=True):
            user_msg = st.text_input(
                "Your message",
                placeholder="e.g. Which model fits this data best? How do I handle the imbalance?",
                label_visibility="collapsed",
            )
            sent = st.form_submit_button("Send", disabled=not groq_key, use_container_width=True)

        if sent and user_msg.strip():
            from modules.groq_client import build_profile_summary, chat_with_dataset
            history.append({"role": "user", "content": user_msg.strip()})
            summary = build_profile_summary(profile, health)
            model_ctx = {
                "problem_type": problem_type, "target": target,
                "model_name": (st.session_state.get("model_config", {}) or {}).get("last_model"),
            }
            with st.spinner("Thinking…"):
                reply = chat_with_dataset(history, summary, groq_key, model_ctx)
            history.append({"role": "assistant", "content": reply})
            st.session_state["model_chat_history"] = history
            st.rerun()

        if history:
            if st.button("🗑 Clear chat"):
                st.session_state["model_chat_history"] = []
                st.rerun()
