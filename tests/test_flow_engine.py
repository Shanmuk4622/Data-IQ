"""tests/test_flow_engine.py — graph-execution logic for the canvas."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from modules.profiler import profile_columns
from modules import flow_engine as fe


def _clf_df(n=200):
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = ((x1 + x2 + (cat == "a") * 1.5) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": y})


def _profile(df):
    return {"columns": profile_columns(df), "preprocessing_recommendations": {},
            "meta": {"rows": len(df), "columns": df.shape[1]}}


def test_default_pipeline_shape():
    nodes, edges = fe.default_pipeline("classification", "target")
    stages = [s for s, _, _ in nodes]
    assert "data" in stages and "test_score" in stages and "model" in stages
    # every edge references valid node indices
    for s, t in edges:
        assert 0 <= s < len(nodes) and 0 <= t < len(nodes)


def test_node_label():
    assert "Test & Score" in fe.node_label("test_score", {"sampling": {"method": "cross_validation", "k": 5}})
    assert "Logistic Regression" in fe.node_label("model", {"model": "Logistic Regression"})


def _graph_from_default(problem_type, target):
    """Build a plain {nodes, edges} graph from the default pipeline (ids = stage-i)."""
    nodes_spec, edge_pairs = fe.default_pipeline(problem_type, target)
    ids = [f"{stage}-{i}" for i, (stage, _, _) in enumerate(nodes_spec)]
    nodes = [{"id": ids[i], "stage": stage, "cfg": cfg}
             for i, (stage, cfg, _) in enumerate(nodes_spec)]
    edges = [{"source": ids[s], "target": ids[t]} for s, t in edge_pairs]
    return {"nodes": nodes, "edges": edges}


def test_execute_workflow_runs_test_score():
    df = _clf_df()
    profile = _profile(df)
    graph = _graph_from_default("classification", "target")
    out = fe.execute_workflow(graph, df, "target", profile, {}, "classification")
    assert out["ok"], out.get("messages")
    ts = out["test_score"]
    assert ts["ok"]
    # both default models evaluated
    assert len(ts["models"]) >= 2
    assert out.get("want_evaluate") and out.get("want_predictions")


def test_execute_workflow_needs_target():
    df = _clf_df()
    graph = _graph_from_default("classification", None)
    # blank the data-node target too
    for n in graph["nodes"]:
        if n["stage"] == "data":
            n["cfg"] = {"target": None}
    out = fe.execute_workflow(graph, df, None, _profile(df), {}, "classification")
    assert out["ok"] is False
    assert any("target" in m.lower() for m in out["messages"])


def test_execute_workflow_rank_node():
    df = _clf_df()
    graph = _graph_from_default("classification", "target")
    graph["nodes"].append({"id": "rank-9", "stage": "rank", "cfg": {"method": "ANOVA"}})
    out = fe.execute_workflow(graph, df, "target", _profile(df), {}, "classification")
    assert out.get("rank") and out["rank"]["ranked"]
