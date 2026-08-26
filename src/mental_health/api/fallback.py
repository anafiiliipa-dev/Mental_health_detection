"""
Deterministic, keyword-based fallback prediction — used ONLY when no real
model is available (see ``model_loader.load_production_model``).

This is explicitly NOT a clinical model. It exists so the API stays usable
in degraded mode (e.g. local dev before anything is promoted, or a
Registry outage) instead of hard-failing every request. ``main.py`` always
tags responses produced by this module with ``is_demo_fallback=True`` so
no caller can mistake it for a real prediction.
"""
from __future__ import annotations

from mental_health.config.paths import CLASS_LABELS

# Small, illustrative keyword sets — good enough to make demo mode behave
# sensibly, not a substitute for the trained model. Deliberately simple.
_KEYWORDS: dict[str, list[str]] = {
    "ADHD": ["adhd", "hyperactive", "hyperactivity", "distracted", "can't focus", "inattentive"],
    "Anxiety": ["anxious", "anxiety", "panic", "worry", "worried", "on edge"],
    "Autism": ["autism", "autistic", "spectrum", "asperger"],
    "Bipolar": ["bipolar", "manic", "mania", "mood swings"],
    "BPD": ["borderline", "bpd", "abandonment"],
    "Depression": ["depress", "hopeless", "empty", "worthless", "no motivation"],
    "Schizophrenia": ["schizophrenia", "psychosis", "hallucinat", "hearing voices", "paranoid"],
}

# Deliberately low, so a real model is never outperformed on paper by the fallback.
_MATCH_CONFIDENCE = 0.35
_NO_MATCH_CONFIDENCE = 1.0 / len(CLASS_LABELS)


def fallback_demo_prediction(text: str) -> tuple[str, float, dict[str, float]]:
    """
    Very simple keyword-count heuristic over ``CLASS_LABELS``.

    Returns (label, confidence, probabilities) with the same shape as the
    real model's output, so ``main.py`` can build a ``PredictResponse``
    identically either way.
    """
    lowered = text.lower()
    scores = {label: sum(lowered.count(kw) for kw in keywords) for label, keywords in _KEYWORDS.items()}
    total_hits = sum(scores.values())

    if total_hits == 0:
        probabilities = dict.fromkeys(CLASS_LABELS, _NO_MATCH_CONFIDENCE)
        return "Depression", _NO_MATCH_CONFIDENCE, probabilities

    label = max(scores, key=scores.get)
    probabilities = {lbl: round(count / total_hits, 4) for lbl, count in scores.items()}
    confidence = max(probabilities[label], _MATCH_CONFIDENCE)
    return label, confidence, probabilities
