"""
Model loading and prediction service.

Centralises all model I/O so that app.py and notebooks
never call joblib directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import pandas as pd

from src.config.paths import CLASS_LABELS, MODEL_CANDIDATES


# ============================================================
# Model loading
# ============================================================

def find_model_path(model_name: str) -> Optional[Path]:
    """Return the first existing path for *model_name*, or None."""
    for candidate in MODEL_CANDIDATES.get(model_name, []):
        if candidate.exists():
            return candidate
    return None


def load_model(model_name: str) -> Tuple[Optional[Any], Optional[Path], Optional[str]]:
    """
    Load a joblib model by name.

    Returns
    -------
    (model, path, error_message)
        model is None when loading fails; error_message is None on success.
    """
    model_path = find_model_path(model_name)

    if model_path is None:
        return None, None, f"No local file found for '{model_name}'. Running in demo mode."

    try:
        model = joblib.load(model_path)
        return model, model_path, None
    except Exception as exc:
        return None, model_path, f"Model found but could not be loaded: {exc}"


# ============================================================
# Prediction helpers
# ============================================================

def predict_with_model(
    model: Any,
    text: str,
) -> Tuple[str, Optional[float], pd.DataFrame]:
    """
    Run inference with a loaded sklearn-compatible model.

    Returns
    -------
    (predicted_label, confidence, probability_dataframe)
        confidence is None when the model does not expose predict_proba.
    """
    prediction: str = str(model.predict([text])[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        class_names = list(model.classes_) if hasattr(model, "classes_") else CLASS_LABELS
        prob_df = (
            pd.DataFrame({"Class": class_names, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        confidence = float(prob_df.iloc[0]["Probability"])
        return prediction, confidence, prob_df

    prob_df = pd.DataFrame({"Class": [prediction], "Probability": [1.0]})
    return prediction, None, prob_df


def fake_demo_prediction(text: str) -> Tuple[str, float, pd.DataFrame]:
    """
    Heuristic-based demo prediction used when no real model is available.

    This is clearly labelled as demo mode in the UI.
    """
    _HEURISTICS = [
        ("Schizophrenia", ["voices", "watching me", "paranoid", "they are after me", "hallucination"]),
        ("Depression",    ["hopeless", "empty", "worthless", "sad", "don't want to live"]),
        ("Anxiety",       ["panic", "nervous", "can't breathe", "worry", "anxious"]),
        ("ADHD",          ["can't focus", "distracted", "concentrate", "restless", "forget"]),
        ("Bipolar",       ["extremely energetic", "no sleep", "unstoppable", "racing thoughts"]),
        ("Autism",        ["overstimulated", "social cues", "sensory", "routine"]),
        ("BPD",           ["abandoned", "intense emotions", "empty inside", "unstable relationships"]),
    ]

    _DEFAULT_LABEL = "Anxiety"
    _DEFAULT_CONFIDENCE = 0.62
    _MATCH_CONFIDENCE = 0.87
    _BASE_SCORE = 0.03

    text_lower = text.lower()
    predicted_label = _DEFAULT_LABEL
    confidence = _DEFAULT_CONFIDENCE

    for label, keywords in _HEURISTICS:
        if any(kw in text_lower for kw in keywords):
            predicted_label = label
            confidence = _MATCH_CONFIDENCE
            break

    rows = [
        {"Class": label, "Probability": confidence if label == predicted_label else _BASE_SCORE}
        for label in CLASS_LABELS
    ]
    prob_df = pd.DataFrame(rows)
    prob_df["Probability"] = (prob_df["Probability"] / prob_df["Probability"].sum()).round(4)
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    return str(prob_df.loc[0, "Class"]), float(prob_df.loc[0, "Probability"]), prob_df
