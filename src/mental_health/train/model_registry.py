"""
Définitions des modèles candidats et calcul des poids de classe pour le
benchmark de ML classique.

Extrait et corrigé de ``notebooks/02_classical_ml.ipynb`` (cellules
"PHASE 3 — COMPUTE CUSTOM CLASS WEIGHTS" et "PHASE 4 — MODEL REGISTRY").

Correction par rapport au notebook original : le boost manuel des poids de classe
vérifiait le label ``"schizophrenia"`` (en minuscules) dans le dictionnaire
de poids. Comme ``src/mental_health/data/cleaning.py`` normalise désormais ce
label en ``"Schizophrenia"`` (voir l'audit pour le bug d'origine), cette
vérification en minuscules serait silencieusement un no-op -- le boost ne s'appliquerait jamais. Ce
module fait dépendre le boost de ``mental_health.config.paths.CRITICAL_LABELS``
plutôt que d'une chaîne codée en dur, afin qu'il y ait une seule source de vérité pour
l'orthographe du label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

from mental_health.config.paths import CRITICAL_LABELS
from mental_health.train.embedding_wrapper import EmbeddingVectorizer
from mental_health.train.xgb_wrapper import XGBTextClassifier

RANDOM_STATE = 42

# Configuration TF-IDF partagée par chaque pipeline candidat (inchangée par rapport à
# le notebook original).
TFIDF_KWARGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 50_000,
    "strip_accents": "unicode",
}

# Boost manuel appliqué en plus des poids de classe "balanced" pour les deux
# labels cliniquement critiques -- même logique et mêmes facteurs que le
# notebook original (schizophrenia est la classe la plus rare et cliniquement la plus
# prioritaire ; bipolar est le second label critique).
CLASS_WEIGHT_BOOST: dict[str, float] = {
    "Schizophrenia": 1.3,
    "Bipolar": 1.2,
}

# Vérification de cohérence au moment de l'import : le boost ne doit jamais cibler que des labels
# effectivement marqués comme cliniquement critiques dans paths.py. Cela transforme une
# incohérence future silencieuse (par ex. quelqu'un renomme un label) en une
# erreur au moment de l'import plutôt qu'un boost sans effet.
_unknown_boost_labels = set(CLASS_WEIGHT_BOOST) - set(CRITICAL_LABELS)
if _unknown_boost_labels:
    raise ValueError(
        f"CLASS_WEIGHT_BOOST references labels not in CRITICAL_LABELS: {_unknown_boost_labels}"
    )


def compute_boosted_class_weights(y_train: pd.Series) -> dict[str, float]:
    """
    Calcule les poids de classe "balanced" de sklearn sur ``y_train``, puis applique le
    boost manuel de ``CLASS_WEIGHT_BOOST`` en plus.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights, strict=True))

    for label, factor in CLASS_WEIGHT_BOOST.items():
        if label in class_weight_dict:
            class_weight_dict[label] *= factor

    return class_weight_dict


def build_model_registry(
    class_weight_dict: dict[str, float], embedding_cache: dict[str, object] | None = None
) -> dict[str, dict]:
    """
    Construit le registre des modèles candidats : pipelines TF-IDF + classifieur,
    chacun avec une petite grille d'hyperparamètres (les 5 d'origine du
    ``MODEL_REGISTRY`` du notebook, plus XGBoost/LightGBM ajoutés en
    Phase 11).

    ``embedding_cache``, lorsqu'il est fourni, ajoute en plus deux
    candidats basés sur des sentence-embeddings (``Embedding_LogReg``,
    ``Embedding_SVM`` -- la "solution intermédiaire" de la Phase 11 entre
    le TF-IDF et un fine-tuning complet de transformer). Omis lorsque
    ``embedding_cache`` vaut ``None`` (valeur par défaut) afin que tout
    appelant existant continue de fonctionner sans changement, et pour que
    l'import/l'utilisation de cette fonction ne nécessite jamais l'extra
    ``embedding_models`` (``sentence-transformers``) sauf si un benchmark basé
    sur des embeddings est réellement demandé.
    """
    registry = {
        "LinearSVC_balanced": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", LinearSVC(class_weight=class_weight_dict, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0, 5.0]},
        },
        "LinearSVC_plain": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", LinearSVC(random_state=RANDOM_STATE)),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0, 5.0]},
        },
        "LogReg_balanced": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", LogisticRegression(
                    solver="saga",
                    class_weight=class_weight_dict,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0, 5.0]},
        },
        "LogReg_plain": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", LogisticRegression(solver="saga", max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0, 5.0]},
        },
        "MultinomialNB": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", MultinomialNB()),
            ]),
            "param_grid": {"clf__alpha": [0.5, 1.0]},
        },
        # Phase 11 : XGBoost/LightGBM publiés sur les mêmes features TF-IDF
        # que le registre linéaire ci-dessus, à la demande de l'audit
        # (mentionné dans architecture.md mais jamais réellement benchmarké).
        # class_weight_dict est la même pondération balanced-boostée que celle
        # déjà utilisée par les candidats linéaires "_balanced".
        "XGBoost_balanced": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", XGBTextClassifier(random_state=RANDOM_STATE, class_weight=class_weight_dict)),
            ]),
            "param_grid": {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 6]},
        },
        "LightGBM_balanced": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", LGBMClassifier(
                    class_weight=class_weight_dict, random_state=RANDOM_STATE, verbose=-1
                )),
            ]),
            "param_grid": {"clf__n_estimators": [100, 200], "clf__num_leaves": [15, 31]},
        },
    }

    if embedding_cache is not None:
        registry["Embedding_LogReg"] = {
            "pipeline": Pipeline([
                ("embed", EmbeddingVectorizer(embedding_cache=embedding_cache)),
                ("clf", LogisticRegression(
                    solver="saga", class_weight=class_weight_dict, max_iter=2000, random_state=RANDOM_STATE
                )),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0]},
        }
        registry["Embedding_SVM"] = {
            "pipeline": Pipeline([
                ("embed", EmbeddingVectorizer(embedding_cache=embedding_cache)),
                ("clf", LinearSVC(class_weight=class_weight_dict, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0]},
        }

    return registry
