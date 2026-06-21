"""pages/10_🛠_Make_ML_Model.py — Orange-style ML studio.

A workflow of connected "widgets": Data → Preprocess → Models → Test & Score →
Predictions → Evaluate, plus Rank (feature scoring) and model viewers.
View-only: all compute lives in modules/, all plots in utils/chart_factory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
import streamlit as st

from modules.model_trainer import (
    list_models, get_model_spec,
    test_and_score, build_predictions_table, rank_features, tune_hyperparameters,
    export_tree_dot, nomogram_data, train_clustering, train_timeseries,
    generate_modeling_guide, model_to_bytes, RANK_METHODS,
)
from utils.chart_factory import (
    make_confusion_matrix, make_roc_overlay, make_performance_curve, make_calibration_plot,
    make_pred_vs_actual, make_residual_plot, make_cluster_scatter, make_elbow_plot,
    make_silhouette_plot, make_nomogram, make_forecast_plot, make_bar_chart,
)


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🛠 Make ML Model")
st.caption("An Orange-style workflow — wire several learners into Test & Score, then inspect "
           "Predictions, Evaluation curves, Rank, and model viewers.")

df_raw = st.session_state.get("df_raw")
profile = st.session_state.get("profile")
if df_raw is None or profile is None:
    st.info("👈 Upload a dataset or try a demo from the sidebar.")
    st.stop()

health = st.session_state.get("health_score") or profile.get("health_score", {})
columns_profile = profile.get("columns", {})
ml = profile.get("ml_readiness", {})
groq_key = st.session_state.get("groq_api_key", "")


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="font-size:1.3rem;">{value}</div>{sub_html}</div>')


def pipeline_header(stages):
    chips = []
    for i, s in enumerate(stages):
        chips.append(
            f'<span style="display:inline-block;padding:4px 12px;border:1px solid #CBD5E1;'
            f'border-radius:999px;background:#F8FAFC;color:#334155;font-size:0.8rem;'
            f'font-weight:500;">{s}</span>'
        )
        if i < len(stages) - 1:
            chips.append('<span style="color:#94A3B8;margin:0 4px;">→</span>')
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:0.5rem 0 0.25rem;">'
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def render_hyperparams(spec: dict, key_prefix: str) -> dict:
    params = spec.get("hyperparams", [])
    if not params:
        st.caption("No tunable hyperparameters.")
        return {}
    hp = {}
    cols = st.columns(min(3, len(params)))
    for i, p in enumerate(params):
        c = cols[i % len(cols)]
        key = f"hp_{key_prefix}_{p['name']}"
        if p["type"] == "int":
            hp[p["name"]] = c.number_input(
                p["name"], min_value=int(p["min"]), max_value=int(p["max"]),
                value=int(p["default"]), step=int(p.get("step", 1)), key=key, help=p.get("help"))
        elif p["type"] == "float":
            hp[p["name"]] = c.number_input(
                p["name"], min_value=float(p["min"]), max_value=float(p["max"]),
                value=float(p["default"]), step=float(p.get("step", 0.01)), key=key, help=p.get("help"))
        elif p["type"] == "select":
            opts = p["options"]
            hp[p["name"]] = c.selectbox(p["name"], opts, index=opts.index(p["default"]), key=key)
    return hp


def style_metric_table(rows: list, problem_type: str):
    """Build a styled DataFrame highlighting the best value per metric column."""
    df = pd.DataFrame(rows)
    higher = (["AUC", "CA", "F1", "Precision", "Recall", "Specificity", "MCC"]
              if problem_type == "classification" else ["R²"])
    lower = (["LogLoss", "Time(s)"] if problem_type == "classification"
             else ["RMSE", "MAE", "MAPE", "MSE", "CVRMSE", "Time(s)"])
    for col in higher + lower:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sty = df.style.format(precision=3, na_rep="—")
    for col in higher:
        if col in df.columns and df[col].notna().any():
            sty = sty.highlight_max(subset=[col], color="#DCFCE7")
    for col in lower:
        if col in df.columns and df[col].notna().any():
            sty = sty.highlight_min(subset=[col], color="#DCFCE7")
    return sty


# ── Data source + problem framing ─────────────────────────────────────────────
df_clean = st.session_state.get("df_clean")
src1, _ = st.columns([1, 1])
with src1:
    use_clean = False
    if df_clean is not None:
        use_clean = st.checkbox("Use cleaned dataset (from Preprocessing)", value=True)
data = df_clean if (df_clean is not None and use_clean) else df_raw

detected_pt = ml.get("problem_type", "classification")
pt_options = ["classification", "regression", "clustering", "timeseries"]
pt_labels = {"classification": "Classification", "regression": "Regression",
             "clustering": "Clustering", "timeseries": "Time Series"}
f1, f2 = st.columns([1, 1])
with f1:
    problem_type = st.selectbox(
        "Problem type", pt_options,
        index=pt_options.index(detected_pt) if detected_pt in pt_options else 0,
        format_func=lambda x: pt_labels[x], help="Auto-detected — override if needed.")
with f2:
    target_opts = ["None (unsupervised)"] + list(data.columns)
    cur_target = st.session_state.get("target_col")
    t_default = cur_target if cur_target in data.columns else "None (unsupervised)"
    target_sel = st.selectbox("Target column", target_opts,
                              index=target_opts.index(t_default) if t_default in target_opts else 0)
target = None if target_sel == "None (unsupervised)" else target_sel
st.caption(f"Training on the **{'cleaned' if (df_clean is not None and use_clean) else 'raw'}** "
           f"dataset ({len(data):,} rows × {data.shape[1]} cols).")

user_choices = st.session_state.get("cleaning_choices", {})

# Reset stale results when problem type changes
mc = st.session_state.get("model_config", {}) or {}
if mc.get("pt") != problem_type:
    for k in ("test_score_results", "predictions_table", "rank_results", "training_results"):
        st.session_state[k] = None
    mc["pt"] = problem_type
    st.session_state["model_config"] = mc

pipeline_header(["📁 Data", "⚙️ Preprocess", "🧠 Models", "🧪 Test & Score",
                 "🔮 Predictions", "📊 Evaluate", "📈 Rank"])

# Workspace mode (Canvas only for supervised problems)
mode = "📑 Tabs"
if problem_type in ("classification", "regression"):
    mode = st.radio("Workspace", ["🎨 Canvas", "📑 Tabs"], horizontal=True,
                    help="Canvas: a drag-and-drop node workflow (Orange-style). Tabs: the guided view.")
st.divider()


def _score_rows(ts, problem_type):
    rows = []
    for name, r in ts["models"].items():
        if not r.get("ok"):
            rows.append({"Model": name, "error": r.get("error")})
            continue
        m = r["metrics"]
        if problem_type == "classification":
            rows.append({"Model": name, "AUC": m["roc_auc"], "CA": m["accuracy"], "F1": m["f1_weighted"],
                         "Precision": m["precision_weighted"], "Recall": m["recall_weighted"],
                         "Specificity": m["specificity"], "LogLoss": m["logloss"], "MCC": m["mcc"],
                         "Time(s)": m["time"]})
        else:
            rows.append({"Model": name, "R²": m["r2"], "RMSE": m["rmse"], "MAE": m["mae"],
                         "MAPE": m["mape"], "MSE": m["mse"], "CVRMSE": m["cvrmse"], "Time(s)": m["time"]})
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Canvas mode — real drag-and-drop node editor (Orange-style)
# ══════════════════════════════════════════════════════════════════════════════
if problem_type in ("classification", "regression") and mode == "🎨 Canvas":
    try:
        from streamlit_flow import streamlit_flow
        from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
        from streamlit_flow.state import StreamlitFlowState
        from modules.flow_engine import (
            STAGE_INFO, ADDABLE_STAGES, node_label, default_pipeline,
            default_model_names, execute_workflow,
        )
        from modules.model_trainer import get_model_spec
        _flow_ok = True
    except Exception as e:
        _flow_ok = False
        st.warning(f"Canvas component unavailable ({e}). Falling back to the Tabs view — "
                   "switch the Workspace toggle to 📑 Tabs.")

    if _flow_ok:
        def _next_id(stage):
            st.session_state["flow_nid"] = st.session_state.get("flow_nid", 0) + 1
            return f"{stage}-{st.session_state['flow_nid']}"

        def _default_cfg(stage):
            if stage == "model":
                return {"model": default_model_names(problem_type)[0], "hp": {}}
            if stage == "test_score":
                return {"sampling": {"method": "cross_validation", "k": 5}}
            if stage == "rank":
                return {"method": RANK_METHODS[problem_type][0]}
            if stage == "data":
                return {"target": target}
            return {}

        def _make_node(stage, cfg, pos, nid):
            info = STAGE_INFO[stage]
            nt = "input" if stage == "data" else ("output" if info["kind"] == "sink" else "default")
            return StreamlitFlowNode(
                id=nid, pos=(float(pos[0]), float(pos[1])),
                data={"content": node_label(stage, cfg), "stage": stage, "cfg": cfg},
                node_type=nt, source_position="bottom", target_position="top",
                draggable=True, connectable=True, selectable=True, deletable=(stage != "data"),
                style={"background": "#FFFFFF", "border": f"2px solid {info['color']}",
                       "borderRadius": "8px", "color": "#0F172A", "width": "165px", "fontSize": "12px"},
            )

        def _build_default_state():
            st.session_state["flow_nid"] = 0
            specs, edge_pairs = default_pipeline(problem_type, target)
            ids, nodes = [], []
            for stage, cfg, pos in specs:
                nid = _next_id(stage)
                ids.append(nid)
                nodes.append(_make_node(stage, cfg, pos, nid))
            edges = [StreamlitFlowEdge(id=f"e{i}", source=ids[s], target=ids[t],
                                       animated=True, deletable=True)
                     for i, (s, t) in enumerate(edge_pairs)]
            return StreamlitFlowState(nodes=nodes, edges=edges)

        if st.session_state.get("flow_state") is None or st.session_state.get("flow_pt") != problem_type:
            st.session_state.flow_state = _build_default_state()
            st.session_state.flow_pt = problem_type

        # ── Palette + actions ────────────────────────────────────────────────
        st.markdown("##### 🧩 Add widgets to the canvas")
        pcols = st.columns(len(ADDABLE_STAGES) + 1)
        for c, stage in zip(pcols, ADDABLE_STAGES):
            if c.button(f"➕ {STAGE_INFO[stage]['label']}", key=f"add_{stage}", use_container_width=True):
                nid = _next_id(stage)
                st.session_state.flow_state.nodes.append(
                    _make_node(stage, _default_cfg(stage), (220, 130), nid))
                st.session_state.flow_state = StreamlitFlowState(
                    nodes=st.session_state.flow_state.nodes, edges=st.session_state.flow_state.edges)
                st.rerun()
        if pcols[-1].button("♻️ Reset", key="reset_canvas", use_container_width=True):
            st.session_state.flow_state = _build_default_state()
            st.rerun()

        st.caption("Drag a node's handle to another to **wire** them · drag nodes to move · "
                   "right-click a node/edge to delete · click a node to **configure** it below.")

        # ── The canvas ───────────────────────────────────────────────────────
        new_state = streamlit_flow(
            "ml_canvas", st.session_state.flow_state, height=470,
            fit_view=True, show_controls=True, allow_new_edges=True, animate_new_edges=True,
            get_node_on_click=True, enable_node_menu=True, enable_edge_menu=True,
            hide_watermark=True,
        )
        st.session_state.flow_state = new_state

        run_col, insp_col = st.columns([1, 1])

        # ── Inspector for the selected node ──────────────────────────────────
        with insp_col:
            sel = getattr(new_state, "selected_id", None)
            node = next((n for n in new_state.nodes if n.id == sel), None) if sel else None
            if node is None:
                st.markdown("##### 🔧 Inspector")
                st.caption("Click a node on the canvas to configure it.")
            else:
                stage = node.data.get("stage")
                st.markdown(f"##### 🔧 Inspector — {STAGE_INFO.get(stage, {}).get('label', stage)}")
                cfg = dict(node.data.get("cfg", {}))

                def _apply(new_cfg):
                    node.data["cfg"] = new_cfg
                    node.data["content"] = node_label(stage, new_cfg)
                    st.session_state.flow_state = StreamlitFlowState(
                        nodes=new_state.nodes, edges=new_state.edges)
                    st.rerun()

                if stage == "data":
                    opts = ["None (unsupervised)"] + list(data.columns)
                    cur = cfg.get("target") or "None (unsupervised)"
                    sel_t = st.selectbox("Target column", opts,
                                         index=opts.index(cur) if cur in opts else 0, key=f"d_{node.id}")
                    if st.button("Apply", key=f"da_{node.id}"):
                        _apply({"target": None if sel_t.startswith("None") else sel_t})
                elif stage == "model":
                    models = list_models(problem_type)
                    cur = cfg.get("model", models[0])
                    mc = st.selectbox("Algorithm", models,
                                      index=models.index(cur) if cur in models else 0, key=f"m_{node.id}")
                    spec = get_model_spec(problem_type, mc)
                    with st.form(f"mform_{node.id}"):
                        hp = render_hyperparams(spec, f"node_{node.id}")
                        if st.form_submit_button("Apply to node"):
                            _apply({"model": mc, "hp": hp})
                elif stage == "test_score":
                    sm = {"cross_validation": "Cross-validation", "random_sampling": "Random sampling",
                          "leave_one_out": "Leave-one-out", "test_on_train": "Test on train data"}
                    s = cfg.get("sampling", {"method": "cross_validation", "k": 5})
                    meth = st.selectbox("Sampling", list(sm), format_func=lambda x: sm[x],
                                        index=list(sm).index(s.get("method", "cross_validation")),
                                        key=f"ts_{node.id}")
                    new_s = {"method": meth}
                    if meth == "cross_validation":
                        new_s["k"] = st.slider("Folds", 2, 10, int(s.get("k", 5)), key=f"k_{node.id}")
                        new_s["stratified"] = True
                    elif meth == "random_sampling":
                        new_s["test_pct"] = st.slider("Test %", 0.1, 0.5, float(s.get("test_pct", 0.3)),
                                                      0.05, key=f"tp_{node.id}")
                        new_s["repeats"] = st.slider("Repeats", 1, 10, int(s.get("repeats", 3)),
                                                     key=f"rp_{node.id}")
                        new_s["stratified"] = True
                    if st.button("Apply", key=f"tsa_{node.id}"):
                        _apply({"sampling": new_s})
                elif stage == "rank":
                    methods = RANK_METHODS[problem_type]
                    cur = cfg.get("method", methods[0])
                    rm = st.selectbox("Scoring method", methods,
                                      index=methods.index(cur) if cur in methods else 0, key=f"r_{node.id}")
                    if st.button("Apply", key=f"ra_{node.id}"):
                        _apply({"method": rm})
                else:
                    st.caption("This widget has no settings — it uses the upstream data / models.")

        # ── Run ──────────────────────────────────────────────────────────────
        with run_col:
            st.markdown("##### ▶ Run")
            st.caption("Executes the wired graph: Models → Test & Score → outputs.")
            if st.button("▶ Run workflow", type="primary", use_container_width=True):
                graph = {
                    "nodes": [{"id": n.id, "stage": n.data.get("stage"), "cfg": n.data.get("cfg", {})}
                              for n in new_state.nodes],
                    "edges": [{"source": e.source, "target": e.target} for e in new_state.edges],
                }
                with st.spinner("Running workflow…"):
                    st.session_state["canvas_results"] = execute_workflow(
                        graph, data, target, profile, user_choices, problem_type)

        # ── Outputs ──────────────────────────────────────────────────────────
        out = st.session_state.get("canvas_results")
        if out:
            st.divider()
            for msg in out.get("messages", []):
                st.warning(msg)
            ts = out.get("test_score")
            if ts and ts.get("ok"):
                ok_models = {n: r for n, r in ts["models"].items() if r.get("ok")}
                st.markdown("#### 🧪 Test & Score")
                st.dataframe(style_metric_table(_score_rows(ts, problem_type), problem_type),
                             use_container_width=True, hide_index=True)

                if out.get("want_evaluate"):
                    st.markdown("#### 📊 Evaluate")
                    classes = ts.get("classes") or []
                    if problem_type == "classification":
                        first = next(iter(ok_models))
                        cm = ok_models[first]["metrics"]["confusion_matrix"]
                        labels = classes if classes else list(range(len(cm)))
                        ev1, ev2 = st.columns(2)
                        ev1.plotly_chart(make_confusion_matrix(cm, labels, title=f"Confusion — {first}"),
                                         use_container_width=True)
                        if len(classes) == 2:
                            cmods = [{"name": n, "y_true": r["y_true"],
                                      "y_score": np.asarray(r["y_prob"])[:, 1]}
                                     for n, r in ok_models.items() if r.get("y_prob") is not None]
                            if cmods:
                                ev2.plotly_chart(make_roc_overlay(cmods), use_container_width=True)
                    else:
                        first = next(iter(ok_models))
                        r = ok_models[first]
                        ev1, ev2 = st.columns(2)
                        ev1.plotly_chart(make_pred_vs_actual(r["y_true"], r["y_pred"]),
                                         use_container_width=True)
                        ev2.plotly_chart(make_residual_plot(r["y_true"], r["y_pred"]),
                                         use_container_width=True)

                if out.get("want_predictions"):
                    st.markdown("#### 🔮 Predictions")
                    fitted = {n: r["fitted"] for n, r in ok_models.items()}
                    table = build_predictions_table(fitted, data.head(50), ts["target"],
                                                    problem_type, ts.get("label_mapping"), 50)
                    st.dataframe(table, use_container_width=True, hide_index=True)

            rk = out.get("rank")
            if rk and rk.get("ranked"):
                st.markdown("#### 📈 Rank")
                top = rk["ranked"][:20][::-1]
                st.plotly_chart(make_bar_chart([n for n, _ in top], [s for _, s in top],
                                               title=f"{rk['method']} score", horizontal=True),
                                use_container_width=True)

        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Supervised: Orange widget tabs
# ══════════════════════════════════════════════════════════════════════════════
if problem_type in ("classification", "regression"):
    tab_models, tab_score, tab_pred, tab_eval, tab_rank, tab_guide = st.tabs(
        ["🧠 Models", "🧪 Test & Score", "🔮 Predictions", "📊 Evaluate", "📈 Rank", "📘 Guide & AI Chat"]
    )

    # ── Models (the learner bench) ────────────────────────────────────────────
    with tab_models:
        st.markdown("### Learner bench")
        st.caption("Pick the models to compare (Orange's model column), then configure each.")
        if target is None:
            st.warning("⚠️ Select a target column above for supervised modeling.")
        else:
            all_models = list_models(problem_type)
            default_sel = st.session_state.get("selected_learners") or all_models[:3]
            default_sel = [m for m in default_sel if m in all_models] or all_models[:3]
            selected = st.multiselect("Models to include", all_models, default=default_sel)
            st.session_state["selected_learners"] = selected

            hp_map = {}
            for name in selected:
                spec = get_model_spec(problem_type, name)
                with st.expander(f"⚙️ {name}", expanded=False):
                    hp_map[name] = render_hyperparams(spec, f"{problem_type}_{name}")
                    if spec.get("param_distributions"):
                        if st.button(f"🎛 Auto-tune {name}", key=f"tunebtn_{problem_type}_{name}"):
                            with st.spinner(f"Tuning {name}…"):
                                out = tune_hyperparameters(data, target, problem_type, name,
                                                           profile, user_choices, n_iter=15, cv=3)
                            if out.get("ok"):
                                for p in spec["hyperparams"]:
                                    if p["name"] in out["best_params"]:
                                        v = out["best_params"][p["name"]]
                                        if p["name"] == "max_depth" and v is None:
                                            v = 0
                                        st.session_state[f"hp_{problem_type}_{name}_{p['name']}"] = v
                                st.success(f"Tuned — best {out['scoring']} = {out['best_score']:.4f}")
                                st.rerun()
                            else:
                                st.error(out.get("error"))
            st.session_state["learner_hyperparams"] = hp_map
            st.info("➡️ Now open **🧪 Test & Score** to evaluate these models together.")

    # ── Test & Score ──────────────────────────────────────────────────────────
    with tab_score:
        st.markdown("### Test & Score")
        selected = st.session_state.get("selected_learners") or []
        if target is None:
            st.warning("⚠️ Select a target column above.")
        elif not selected:
            st.warning("⚠️ Select models in the **🧠 Models** tab first.")
        else:
            sm_labels = {"cross_validation": "Cross-validation", "random_sampling": "Random sampling",
                         "leave_one_out": "Leave-one-out", "test_on_train": "Test on train data"}
            method = st.radio("Sampling", list(sm_labels), format_func=lambda x: sm_labels[x],
                              horizontal=True)
            sampling = {"method": method}
            if method == "cross_validation":
                sampling["k"] = st.slider("Number of folds", 2, 10, 5)
                sampling["stratified"] = True
            elif method == "random_sampling":
                rc1, rc2 = st.columns(2)
                sampling["test_pct"] = rc1.slider("Test proportion", 0.1, 0.5, 0.3, 0.05)
                sampling["repeats"] = rc2.slider("Repeats", 1, 10, 3)
                sampling["stratified"] = True

            if st.button("🧪 Run Test & Score", type="primary"):
                with st.spinner("Evaluating models…"):
                    res = test_and_score(data, target, problem_type, selected,
                                         st.session_state.get("learner_hyperparams", {}),
                                         sampling, profile, user_choices)
                st.session_state["test_score_results"] = res

            res = st.session_state.get("test_score_results")
            if res:
                if not res.get("ok"):
                    st.error(res.get("error"))
                else:
                    rows = []
                    for name, r in res["models"].items():
                        if not r.get("ok"):
                            rows.append({"Model": name, "CA": None, "error": r.get("error")})
                            continue
                        m = r["metrics"]
                        if problem_type == "classification":
                            rows.append({"Model": name, "AUC": m["roc_auc"], "CA": m["accuracy"],
                                         "F1": m["f1_weighted"], "Precision": m["precision_weighted"],
                                         "Recall": m["recall_weighted"], "Specificity": m["specificity"],
                                         "LogLoss": m["logloss"], "MCC": m["mcc"], "Time(s)": m["time"]})
                        else:
                            rows.append({"Model": name, "R²": m["r2"], "RMSE": m["rmse"], "MAE": m["mae"],
                                         "MAPE": m["mape"], "MSE": m["mse"], "CVRMSE": m["cvrmse"],
                                         "Time(s)": m["time"]})
                    st.dataframe(style_metric_table(rows, problem_type),
                                 use_container_width=True, hide_index=True)
                    st.caption(f"Sampling: {sm_labels.get(res['sampling']['method'])}. Best value per "
                               "metric highlighted. ➡️ See **Predictions** and **Evaluate** next.")

                    comp = res.get("comparison")
                    if comp:
                        with st.expander("Model comparison — P(row scores higher than column)"):
                            cm = pd.DataFrame(comp["matrix"]).T
                            st.dataframe(cm.style.format(precision=2, na_rep="—"),
                                         use_container_width=True)
                            st.caption("Approximate probability (paired t-test on per-fold scores). "
                                       "Cross-validation only.")

    # ── Predictions ───────────────────────────────────────────────────────────
    with tab_pred:
        st.markdown("### Predictions")
        res = st.session_state.get("test_score_results")
        if not (res and res.get("ok")):
            st.info("Run **🧪 Test & Score** first to fit the models.")
        else:
            fitted = {n: r["fitted"] for n, r in res["models"].items() if r.get("ok")}
            chosen = st.multiselect("Models to apply", list(fitted), default=list(fitted))
            src = st.radio("Predict on", ["Current dataset", "Upload new data"], horizontal=True)
            pred_df = data
            if src == "Upload new data":
                up = st.file_uploader("New data (same columns)", type=["csv", "xlsx", "json"],
                                      key="pred_upload")
                if up is not None:
                    try:
                        if up.name.endswith(".csv"):
                            pred_df = pd.read_csv(up)
                        elif up.name.endswith(".json"):
                            pred_df = pd.read_json(up)
                        else:
                            pred_df = pd.read_excel(up)
                    except Exception as e:
                        st.error(f"Could not read file: {e}")
                        pred_df = data
            n = st.slider("Rows to show", 10, min(2000, len(pred_df)), min(100, len(pred_df)))
            if chosen:
                table = build_predictions_table(
                    {k: fitted[k] for k in chosen}, pred_df.head(n),
                    res["target"], problem_type, res.get("label_mapping"), max_rows=n)
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.download_button("📥 Download predictions (CSV)",
                                   table.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    with tab_eval:
        st.markdown("### Evaluate")
        res = st.session_state.get("test_score_results")
        if not (res and res.get("ok")):
            st.info("Run **🧪 Test & Score** first.")
        else:
            ok_models = {n: r for n, r in res["models"].items() if r.get("ok")}
            classes = res.get("classes") or []

            if problem_type == "classification":
                cpick = st.selectbox("Confusion matrix — model", list(ok_models))
                norm = st.checkbox("Show proportions", value=False)
                cm = ok_models[cpick]["metrics"]["confusion_matrix"]
                labels = classes if classes else list(range(len(cm)))
                st.plotly_chart(make_confusion_matrix(cm, labels, normalize=norm),
                                use_container_width=True)

                if len(classes) == 2:
                    curve_models = [
                        {"name": n, "y_true": r["y_true"], "y_score": np.asarray(r["y_prob"])[:, 1]}
                        for n, r in ok_models.items() if r.get("y_prob") is not None
                    ]
                    if curve_models:
                        st.plotly_chart(make_roc_overlay(curve_models), use_container_width=True)
                        kind_labels = {"lift": "Lift", "gains": "Cumulative Gains",
                                       "prec_recall": "Precision-Recall"}
                        kind = st.radio("Performance curve", list(kind_labels),
                                        format_func=lambda x: kind_labels[x], horizontal=True)
                        st.plotly_chart(make_performance_curve(curve_models, kind),
                                        use_container_width=True)
                        st.plotly_chart(make_calibration_plot(curve_models), use_container_width=True)
                else:
                    st.caption("ROC / Lift / Calibration are shown for binary targets.")
            else:
                dpick = st.selectbox("Diagnostics — model", list(ok_models))
                r = ok_models[dpick]
                ec1, ec2 = st.columns(2)
                ec1.plotly_chart(make_pred_vs_actual(r["y_true"], r["y_pred"]), use_container_width=True)
                ec2.plotly_chart(make_residual_plot(r["y_true"], r["y_pred"]), use_container_width=True)

            # Viewers
            st.markdown("#### 🌳 Model viewers")
            tree_models = [n for n in ok_models
                           if (get_model_spec(problem_type, n) or {}).get("viewer") == "tree"]
            nomo_models = [n for n in ok_models
                           if (get_model_spec(problem_type, n) or {}).get("viewer") == "nomogram"]
            if tree_models:
                tv = st.selectbox("Tree Viewer — model", tree_models, key="treeview")
                depth = st.slider("Tree depth to display", 1, 6, 3)
                dot = export_tree_dot(ok_models[tv]["fitted"], res.get("classes"), max_depth=depth)
                if dot:
                    st.graphviz_chart(dot, use_container_width=True)
                else:
                    st.caption("Tree view unavailable for this model.")
            if nomo_models:
                nm = st.selectbox("Nomogram — model", nomo_models, key="nomoview")
                nd = nomogram_data(ok_models[nm]["fitted"])
                if nd:
                    st.plotly_chart(make_nomogram([n for n, _ in nd], [v for _, v in nd]),
                                    use_container_width=True)
                else:
                    st.caption("Nomogram unavailable for this model.")
            if not tree_models and not nomo_models:
                st.caption("Add a Decision Tree / Random Forest (tree view) or Logistic Regression "
                           "(nomogram) in the Models tab to enable viewers.")

            # Download fitted model
            st.markdown("#### 💾 Download fitted model")
            dl = st.selectbox("Model", list(ok_models), key="dlmodel")
            try:
                st.download_button(f"📥 {dl} (.pkl)", model_to_bytes(ok_models[dl]["fitted"]),
                                   f"{dl.replace(' ', '_').lower()}.pkl", "application/octet-stream")
            except Exception as e:
                st.caption(f"Export unavailable: {e}")

    # ── Rank ──────────────────────────────────────────────────────────────────
    with tab_rank:
        st.markdown("### Rank — feature scoring")
        if target is None:
            st.warning("⚠️ Select a target column above to rank features.")
        else:
            methods = RANK_METHODS.get(problem_type, [])
            method = st.selectbox("Scoring method", methods)
            if st.button("📈 Rank features", type="primary"):
                with st.spinner("Scoring features…"):
                    ranked = rank_features(data, target, problem_type, method, profile)
                st.session_state["rank_results"] = {"method": method, "ranked": ranked}
            rr = st.session_state.get("rank_results")
            if rr:
                ranked = rr["ranked"]
                if not ranked:
                    st.info("No scorable features (identifiers / near-unique / datetime excluded).")
                else:
                    top = ranked[:20][::-1]
                    st.plotly_chart(
                        make_bar_chart([n for n, _ in top], [s for _, s in top],
                                       title=f"{rr['method']} score", horizontal=True),
                        use_container_width=True)
                    st.dataframe(pd.DataFrame(ranked, columns=["Feature", rr["method"]]),
                                 use_container_width=True, hide_index=True)

    # ── Guide & AI Chat ───────────────────────────────────────────────────────
    with tab_guide:
        g_col, c_col = st.columns([1, 1])
        with g_col:
            st.markdown("### 📘 Modeling guide")
            st.markdown(generate_modeling_guide(profile, target, problem_type))
        with c_col:
            st.markdown("### 💬 AI Chat")
            if not groq_key:
                st.markdown('<div class="warning-box">⚠️ Add your Groq API key in the sidebar to chat.</div>',
                            unsafe_allow_html=True)
            history = st.session_state.get("model_chat_history", [])
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            with st.form("model_chat_form", clear_on_submit=True):
                user_msg = st.text_input("Your message",
                                         placeholder="e.g. Which model wins and why?",
                                         label_visibility="collapsed")
                sent = st.form_submit_button("Send", disabled=not groq_key, use_container_width=True)
            if sent and user_msg.strip():
                from modules.groq_client import build_profile_summary, chat_with_dataset
                history.append({"role": "user", "content": user_msg.strip()})
                summary = build_profile_summary(profile, health)
                model_ctx = {"problem_type": problem_type, "target": target,
                             "model_name": ", ".join(st.session_state.get("selected_learners", []) or [])}
                with st.spinner("Thinking…"):
                    reply = chat_with_dataset(history, summary, groq_key, model_ctx)
                history.append({"role": "assistant", "content": reply})
                st.session_state["model_chat_history"] = history
                st.rerun()
            if history and st.button("🗑 Clear chat"):
                st.session_state["model_chat_history"] = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  Clustering
# ══════════════════════════════════════════════════════════════════════════════
elif problem_type == "clustering":
    tab_cluster, tab_guide = st.tabs(["🔮 Cluster", "📘 Guide & AI Chat"])
    with tab_cluster:
        st.markdown("### Unsupervised clustering")
        model_name = st.selectbox("Algorithm", list_models("clustering"))
        spec = get_model_spec("clustering", model_name)
        with st.expander("⚙️ Hyperparameters", expanded=True):
            hp = render_hyperparams(spec, f"clustering_{model_name}")
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner(f"Running {model_name}…"):
                st.session_state["training_results"] = train_clustering(
                    data, profile, model_name, hp, user_choices)
        res = st.session_state.get("training_results")
        if res and "labels" in res:
            if not res.get("ok"):
                st.error(res.get("error"))
            else:
                c1, c2, c3 = st.columns(3)
                c1.markdown(metric_card("Clusters", str(res["n_clusters"])), unsafe_allow_html=True)
                sil = res.get("silhouette")
                c2.markdown(metric_card("Silhouette", f"{sil:.3f}" if sil is not None else "—",
                                        "−1 to 1, higher is better"), unsafe_allow_html=True)
                c3.markdown(metric_card("Noise points", str(res.get("n_noise", 0))), unsafe_allow_html=True)
                if res.get("sil_samples"):
                    st.plotly_chart(make_silhouette_plot(res["sil_samples"], res["sil_sample_labels"]),
                                    use_container_width=True)
                if res.get("pca_x"):
                    st.plotly_chart(make_cluster_scatter(res["pca_x"], res["pca_y"], res["labels"]),
                                    use_container_width=True)
                if res.get("elbow_k"):
                    st.plotly_chart(make_elbow_plot(res["elbow_k"], res["elbow_inertias"]),
                                    use_container_width=True)
    with tab_guide:
        st.markdown(generate_modeling_guide(profile, target, problem_type))


# ══════════════════════════════════════════════════════════════════════════════
#  Time series
# ══════════════════════════════════════════════════════════════════════════════
elif problem_type == "timeseries":
    tab_fc, tab_guide = st.tabs(["📈 Forecast", "📘 Guide & AI Chat"])
    with tab_fc:
        st.markdown("### Time-series forecasting")
        dt_cols = list(profile.get("datetime_analysis", {}).keys())
        if not dt_cols:
            dt_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") == "datetime"]
        num_cols = [c for c, v in columns_profile.items() if v.get("dtype_class") == "numerical"]
        if not dt_cols:
            st.info("No datetime column detected for forecasting.")
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
                order = (int(oc[0].number_input("p (AR)", 0, 5, 1)),
                         int(oc[1].number_input("d (diff)", 0, 2, 1)),
                         int(oc[2].number_input("q (MA)", 0, 5, 1)))
            horizon = st.slider("Forecast horizon (steps)", 7, 180, 30, 1)
            if st.button("🔮 Forecast", type="primary"):
                with st.spinner("Fitting and forecasting…"):
                    st.session_state["training_results"] = train_timeseries(
                        data, date_col, value_col, order, horizon, engine)
            res = st.session_state.get("training_results")
            if res and ("forecast_y" in res or res.get("error")):
                if not res.get("ok"):
                    st.error(res.get("error"))
                else:
                    st.markdown(metric_card(res["metric_label"], str(res["metric_value"]),
                                            f"engine: {res['engine']}"), unsafe_allow_html=True)
                    st.plotly_chart(
                        make_forecast_plot(res["history_x"], res["history_y"], res["forecast_x"],
                                           res["forecast_y"], res.get("lower"), res.get("upper"),
                                           title=f"{value_col} forecast (+{horizon} steps)"),
                        use_container_width=True)
    with tab_guide:
        st.markdown(generate_modeling_guide(profile, target, problem_type))
