"""modules/flow_engine.py — Logic for the Orange-style drag-and-drop node canvas.

Pure logic (no Streamlit / no streamlit_flow imports) so it stays testable. The page
converts the visual graph into a plain dict and calls `execute_workflow`.

A "graph" here is:
    {"nodes": [{"id": str, "stage": str, "cfg": dict}, ...],
     "edges": [{"source": str, "target": str}, ...]}
"""
from __future__ import annotations

from modules.model_trainer import test_and_score, rank_features, RANK_METHODS


# Stage catalogue — the "widgets" you can place on the canvas.
STAGE_INFO = {
    "data":        {"label": "Data",         "color": "#475569", "kind": "source"},
    "preprocess":  {"label": "Preprocess",   "color": "#0D9488", "kind": "mid"},
    "model":       {"label": "Model",        "color": "#7C3AED", "kind": "mid"},
    "test_score":  {"label": "Test & Score", "color": "#2563EB", "kind": "mid"},
    "predictions": {"label": "Predictions",  "color": "#0891B2", "kind": "sink"},
    "evaluate":    {"label": "Evaluate",     "color": "#DB2777", "kind": "sink"},
    "rank":        {"label": "Rank",         "color": "#D97706", "kind": "sink"},
}

# Stages you can add from the palette (Data is created once, automatically).
ADDABLE_STAGES = ["preprocess", "model", "test_score", "predictions", "evaluate", "rank"]


def node_label(stage: str, cfg: dict) -> str:
    """Short label shown inside a canvas node."""
    info = STAGE_INFO.get(stage, {"label": stage})
    title = info["label"]
    cfg = cfg or {}
    if stage == "data":
        tgt = cfg.get("target")
        sub = f"target: {tgt}" if tgt else "no target"
    elif stage == "model":
        sub = cfg.get("model", "— pick model —")
    elif stage == "test_score":
        s = cfg.get("sampling", {})
        m = s.get("method", "cross_validation")
        sub = {"cross_validation": f"CV · {s.get('k', 5)} folds",
               "random_sampling": f"random · {int(s.get('test_pct', 0.3) * 100)}%",
               "leave_one_out": "leave-one-out",
               "test_on_train": "test on train"}.get(m, m)
    elif stage == "rank":
        sub = cfg.get("method", "auto")
    else:
        sub = ""
    return f"**{title}**\n\n{sub}" if sub else f"**{title}**"


def default_model_names(problem_type: str) -> tuple[str, str]:
    if problem_type == "classification":
        return "Logistic Regression", "Random Forest Classifier"
    return "Linear Regression", "Random Forest Regressor"


def default_pipeline(problem_type: str, target):
    """Return (nodes, edges) for a starter workflow.

    nodes: list of (stage, cfg, (x, y))
    edges: list of (src_index, tgt_index)
    """
    m1, m2 = default_model_names(problem_type)
    nodes = [
        ("data",        {"target": target}, (60, 30)),                                   # 0
        ("preprocess",  {}, (60, 150)),                                                   # 1
        ("model",       {"model": m1, "hp": {}}, (10, 270)),                              # 2
        ("model",       {"model": m2, "hp": {}}, (300, 270)),                             # 3
        ("test_score",  {"sampling": {"method": "cross_validation", "k": 5}}, (150, 390)),# 4
        ("evaluate",    {}, (40, 510)),                                                    # 5
        ("predictions", {}, (300, 510)),                                                   # 6
    ]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6)]
    return nodes, edges


def _reachable(start: str, adj: dict) -> set:
    seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):  # neighbours
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen


def execute_workflow(graph: dict, data, target, profile, user_choices, problem_type) -> dict:
    """Interpret the wired graph and run the corresponding engine calls.

    Returns {ok, messages, target, test_score?, want_predictions?, want_evaluate?, rank?}.
    Lenient: if the user hasn't wired anything yet, model nodes are still picked up so the
    canvas does something useful, with a hint to connect them.
    """
    messages: list[str] = []
    out: dict = {"ok": False, "messages": messages}

    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    by_stage: dict[str, list] = {}
    for n in graph.get("nodes", []):
        by_stage.setdefault(n["stage"], []).append(n)

    edges = graph.get("edges", [])
    has_edges = len(edges) > 0
    fwd, rev = {}, {}
    for e in edges:
        fwd.setdefault(e["source"], set()).add(e["target"])
        rev.setdefault(e["target"], set()).add(e["source"])

    # Resolve target (Data node overrides the page selector)
    tgt = target
    for dn in by_stage.get("data", []):
        if dn["cfg"].get("target"):
            tgt = dn["cfg"]["target"]
    out["target"] = tgt

    # ── Test & Score ──────────────────────────────────────────────────────────
    ts_nodes = by_stage.get("test_score", [])
    if not ts_nodes:
        messages.append("Add a **Test & Score** node and connect Model node(s) into it.")
    else:
        T = ts_nodes[0]["id"]
        upstream = _reachable(T, rev)
        model_nodes = [n for n in by_stage.get("model", [])
                       if (n["id"] in upstream) or (not has_edges)]
        if not by_stage.get("model"):
            messages.append("Add at least one **Model** node.")
        elif not model_nodes:
            messages.append("Wire your **Model** node(s) into **Test & Score** (drag handle to handle).")
        elif tgt is None:
            messages.append("Set a **target** column (Data node, or the selector above).")
        else:
            selected, hp_map = [], {}
            for n in model_nodes:
                mn = n["cfg"].get("model")
                if mn and mn not in selected:
                    selected.append(mn)
                    hp_map[mn] = n["cfg"].get("hp", {})
            sampling = nodes_by_id[T]["cfg"].get("sampling", {"method": "cross_validation", "k": 5})
            res = test_and_score(data, tgt, problem_type, selected, hp_map, sampling,
                                 profile, user_choices)
            out["test_score"] = res
            out["ok"] = bool(res.get("ok"))
            if not res.get("ok"):
                messages.append(res.get("error", "Test & Score failed."))
            else:
                downstream = _reachable(T, fwd) if has_edges else set(nodes_by_id)
                out["want_predictions"] = any(
                    (n["id"] in downstream) or (not has_edges) for n in by_stage.get("predictions", []))
                out["want_evaluate"] = any(
                    (n["id"] in downstream) or (not has_edges) for n in by_stage.get("evaluate", []))

    # ── Rank (independent of Test & Score) ────────────────────────────────────
    rank_nodes = by_stage.get("rank", [])
    if rank_nodes:
        if tgt is None:
            messages.append("Rank needs a **target** column.")
        else:
            method = rank_nodes[0]["cfg"].get("method") or RANK_METHODS.get(problem_type, ["?"])[0]
            out["rank"] = {"method": method,
                           "ranked": rank_features(data, tgt, problem_type, method, profile)}

    return out
