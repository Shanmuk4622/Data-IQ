"""modules/target_detector.py — Auto-detect the likely target column."""
from __future__ import annotations
import pandas as pd


def detect_target_column(
    df: pd.DataFrame, columns_profile: dict
) -> tuple[str | None, float]:
    """
    Returns (best_target_column_name, confidence_score).
    Requires minimum confidence of 0.3 to auto-select.
    """
    if not columns_profile:
        return None, 0.0

    scores = {
        col: info.get("target_candidate_score", 0.0)
        for col, info in columns_profile.items()
    }

    if not scores:
        return None, 0.0

    best_col = max(scores, key=scores.get)
    confidence = scores[best_col]

    if confidence < 0.3:
        return None, confidence

    return best_col, confidence


def get_top_candidates(columns_profile: dict, top_n: int = 3) -> list[dict]:
    """Return top N target column candidates ranked by score."""
    scored = [
        {
            "col": col,
            "score": info.get("target_candidate_score", 0.0),
            "dtype": info.get("dtype_class", "unknown"),
            "unique_count": info.get("unique_count", 0),
            "null_pct": info.get("null_pct", 0.0),
        }
        for col, info in columns_profile.items()
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]
