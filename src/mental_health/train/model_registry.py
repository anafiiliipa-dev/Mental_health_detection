"""
Candidate model definitions and class-weight computation for the classical
ML benchmark.

Extracted and corrected from ``notebooks/02_classical_ml.ipynb`` (cells
"PHASE 3 — COMPUTE CUSTOM CLASS WEIGHTS" and "PHASE 4 — MODEL REGISTRY").

Correction relative to the original notebook: the manual class-weight boost
checked for the label ``"schizophrenia"`` (lowercase) in the weights
dictionary. Since ``src/mental_health/data/cleaning.py`` now normalises this
label to ``"Schizophrenia"`` (see the audit for the original bug), that
lowercase check would silently no-op — the boost would never apply. This
module keys the boost off ``mental_health.config.paths.CRITICAL_LABELS``
instead of a hardcoded string, so there is a single source of truth for the
label spelling.
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

# TF-IDF configuration shared by every candidate pipeline (unchanged from
# the original notebook).
TFIDF_KWARGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 50_000,
    "strip_accents": "unicode",
}

# Manual boost applied on top of the "balanced" class weights for the two
# clinically critical labels — same rationale and factors as the original
# notebook (schizophrenia is the rarest and clinically highest-priority
# class; bipolar is the second critical label).
CLASS_WEIGHT_BOOST: dict[str, float] = {
    "Schizophrenia": 1.3,
    "Bipolar": 1.2,
}

# Sanity check at import time: the boost must only ever target labels that
# are actually flagged as clinically critical in paths.py. This turns a
# silent future mismatch (e.g. someone renames a label) into an import-time
# error instead of a no-op boost.
_unknown_boost_labels = set(CLASS_WEIGHT_BOOST) - set(CRITICAL_LABELS)
if _unknown_boost_labels:
    raise ValueError(
        f"CLASS_WEIGHT_BOOST references labels not in CRITICAL_LABELS: {_unknown_boost_labels}"
    )


def compute_boosted_class_weights(y_train: pd.Series) -> dict[str, float]:
    """
    Compute sklearn "balanced" class weights on ``y_train``, then apply the
    manual boost from ``CLASS_WEIGHT_BOOST`` on top.
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
    Build the candidate model registry: TF-IDF + classifier pipelines,
    each with a small hyperparameter grid (the original 5 from the
    notebook's ``MODEL_REGISTRY``, plus XGBoost/LightGBM added in
    Phase 11).

    ``embedding_cache``, when provided, additionally adds two
    sentence-embedding candidates (``Embedding_LogReg``,
    ``Embedding_SVM`` -- Phase 11's "solution intermediaire" between
    TF-IDF and a full transformer fine-tune). Left out when
    ``embedding_cache`` is ``None`` (the default) so every existing
    caller keeps working unchanged, and so importing/using this function
    never requires the ``embedding_models`` extra (``sentence-transformers``)
    to be installed unless an embedding-based benchmark is actually
    requested.
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
        # Phase 11: XGBoost/LightGBM published on the same TF-IDF features
        # as the linear registry above, on request from the audit
        # (mentioned in architecture.md but never actually benchmarked).
        # class_weight_dict is the same boosted-balanced weighting the
        # "_balanced" linear candidates already use.
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
