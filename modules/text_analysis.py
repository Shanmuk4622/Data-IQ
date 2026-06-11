"""modules/text_analysis.py — Word frequency, n-grams, NLP detection."""
from __future__ import annotations
import re
from collections import Counter
import pandas as pd
import numpy as np


STOPWORDS = {
    "the", "a", "an", "is", "in", "it", "of", "and", "to", "was", "for",
    "on", "are", "as", "at", "be", "by", "from", "or", "that", "this",
    "with", "but", "not", "have", "had", "he", "she", "they", "we", "you",
    "i", "my", "our", "his", "her", "its", "their", "will", "would", "can",
    "could", "do", "did", "does", "been", "has", "have", "so", "if", "then",
    "than", "what", "which", "who", "how", "when", "where", "all", "each",
    "more", "also", "after", "before", "into", "up", "out", "about", "over",
}


def _get_ngrams(words: list[str], n: int) -> dict[str, int]:
    grams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    return dict(Counter(grams).most_common(15))


def analyze_text_column(series: pd.Series) -> dict:
    """Full text analysis for a text column."""
    try:
        texts = series.dropna().astype(str)
        if len(texts) == 0:
            return {}

        lengths = texts.str.len()
        words_all = []
        for text in texts:
            words_all.extend(re.findall(r'\b[a-z]{3,}\b', text.lower()))

        filtered_words = [w for w in words_all if w not in STOPWORDS]

        return {
            "avg_length": round(float(lengths.mean()), 1),
            "max_length": int(lengths.max()),
            "min_length": int(lengths.min()),
            "total_words": len(filtered_words),
            "vocabulary_size": len(set(filtered_words)),
            "top_words": dict(Counter(filtered_words).most_common(20)),
            "bigrams": _get_ngrams(filtered_words, 2),
            "trigrams": _get_ngrams(filtered_words, 3),
            "is_nlp_candidate": True,
        }
    except Exception:
        return {}


def analyze_text(df: pd.DataFrame, columns_profile: dict) -> dict:
    """Analyse all text columns."""
    result = {}
    for col, info in columns_profile.items():
        if info.get("dtype_class") == "text":
            analysis = analyze_text_column(df[col])
            if analysis:
                result[col] = analysis
    return result
