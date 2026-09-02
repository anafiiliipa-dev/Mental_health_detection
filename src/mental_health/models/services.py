"""
Service de chargement de modèle et de prédiction.

Centralise tous les I/O de modèle afin que app.py et les notebooks
n'appellent jamais joblib directement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mental_health.config.paths import CLASS_LABELS, MODEL_CANDIDATES

# ============================================================
# Chargement du modèle
# ============================================================

def find_model_path(model_name: str) -> Path | None:
    """Retourne le premier chemin existant pour *model_name*, ou None."""
    for candidate in MODEL_CANDIDATES.get(model_name, []):
        if candidate.exists():
            return candidate
    return None


def load_model(model_name: str) -> tuple[Any | None, Path | None, str | None]:
    """
    Charge un modèle joblib par son nom.

    Returns
    -------
    (model, path, error_message)
        model est None en cas d'échec de chargement ; error_message est None en cas de succès.
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
# Fonctions d'aide à la prédiction
# ============================================================

def predict_with_model(
    model: Any,
    text: str,
) -> tuple[str, float | None, pd.DataFrame]:
    """
    Exécute l'inférence avec un modèle chargé compatible sklearn.

    Returns
    -------
    (predicted_label, confidence, probability_dataframe)
        confidence est None quand le modèle n'expose pas predict_proba.
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


def fallback_demo_prediction(text: str) -> tuple[str, float, pd.DataFrame]:
    """
    Prédiction de démo basée sur des heuristiques, utilisée quand aucun modèle réel n'est disponible.

    Ceci est clairement signalé comme mode démo dans l'UI.
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

# ============================================================
# Chargement des artefacts d'évaluation
# ============================================================

def load_csv_with_fallback(relative_path: str) -> tuple[pd.DataFrame | None, Path | None, bool]:
    """
    Charge un CSV d'évaluation depuis le vrai dossier de rapports.

    Si l'artefact réel n'existe pas, recours à docs/sample_outputs/
    en utilisant le même nom de fichier.

    Returns
    -------
    (dataframe, path_used, is_sample)
        dataframe est None quand aucun fichier n'est trouvé.
        is_sample est True quand le fichier d'exemple de recours est utilisé.
    """
    base_dir = Path(__file__).resolve().parents[2]

    primary_path = base_dir / relative_path
    fallback_path = base_dir / "docs" / "sample_outputs" / Path(relative_path).name

    if primary_path.exists():
        return pd.read_csv(primary_path), primary_path, False

    if fallback_path.exists():
        return pd.read_csv(fallback_path), fallback_path, True

    return None, None, False
