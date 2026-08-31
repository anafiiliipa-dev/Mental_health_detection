"""
Champion selection, final training and final evaluation.

Extracted from ``notebooks/02_classical_ml.ipynb`` (cells "PHASE 10 —
CHAMPION MODEL SELECTION" through "PHASE 17 — CONFUSION MATRIX"). No
behavioural corrections needed here — this part of the notebook already
used the corrected ``nested_cv_summary`` (produced by
``benchmark.run_nested_cv_benchmark``, itself already fixed in the 3b step)
and consistent label casing throughout.
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from mental_health.train.benchmark import critical_recall_score
from mental_health.train.evaluation_metrics import (
    compute_brier_score,
    compute_ece,
    compute_mcc,
    compute_pr_auc_per_class,
    get_ranking_scores,
)


def select_champion_config(nested_summary: pd.DataFrame) -> dict:
    """
    Pick the champion (model, text_variant) pair: the top-ranked row of the
    nested CV summary, ranked by robust_score (ties broken by
    critical_recall_mean then f1_macro_mean).
    """
    ranked = nested_summary.sort_values(
        by=["robust_score", "critical_recall_mean", "f1_macro_mean"],
        ascending=False,
    ).reset_index(drop=True)

    champion_row = ranked.iloc[0].to_dict()
    return {
        "model_name": champion_row["model"],
        "text_variant": champion_row["text_variant"],
    }


def select_runner_up_config(nested_summary: pd.DataFrame) -> dict | None:
    """
    Second-ranked (model, text_variant) pair from the nested CV summary —
    the comparison point for the Phase 11 paired bootstrap significance
    test (see ``evaluation_metrics.paired_bootstrap_test``): "is the
    champion actually better than the next-best candidate, or just a
    lucky split?". Returns ``None`` if the summary has fewer than two
    ranked rows (e.g. a benchmark run over a single candidate).
    """
    ranked = nested_summary.sort_values(
        by=["robust_score", "critical_recall_mean", "f1_macro_mean"],
        ascending=False,
    ).reset_index(drop=True)
    if len(ranked) < 2:
        return None

    runner_up_row = ranked.iloc[1].to_dict()
    return {"model_name": runner_up_row["model"], "text_variant": runner_up_row["text_variant"]}


def select_champion_params(nested_best_params: dict, text_variant: str, model_name: str) -> dict:
    """
    Pick the champion's hyperparameters: the mode (most frequent value)
    across the nested CV's outer folds for this (model, text_variant).

    Using the mode rather than e.g. the best single fold avoids overfitting
    the final hyperparameters to one lucky outer split.
    """
    params_list = nested_best_params[text_variant]["best_params"][model_name]
    extracted_params = [item["best_params"] for item in params_list if item["best_params"] is not None]

    if not extracted_params:
        raise ValueError(f"No nested params found for {model_name} / {text_variant}")

    param_counter = Counter(json.dumps(p, sort_keys=True) for p in extracted_params)
    most_common_params_str = param_counter.most_common(1)[0][0]
    return json.loads(most_common_params_str)


def train_final_model(
    model_registry: dict,
    model_name: str,
    params: dict,
    X_train,  # noqa: N803
    y_train,
    calibrate: bool = False,
    calibration_cv: int = 5,
    calibration_method: str = "sigmoid",
):
    """
    Fit the champion pipeline (architecture from the registry, tuned
    params) on the full training set.

    ``calibrate=True`` (Phase 11) wraps the fitted-and-cloned pipeline in
    ``CalibratedClassifierCV`` (Platt scaling by default) before fitting --
    this is what turns LinearSVC's raw ``decision_function`` output into
    real, well-calibrated probabilities (``predict_proba``), which is a
    prerequisite for Brier score / ECE meaning anything, and also gives
    the API real confidence scores instead of the current softmax
    approximation over ``decision_function``. Off by default so every
    other caller (e.g. the runner-up trained only for the significance
    test) keeps the original, uncalibrated behaviour.
    """
    base_model = clone(model_registry[model_name]["pipeline"])
    base_model.set_params(**params)

    final_model = (
        CalibratedClassifierCV(base_model, cv=calibration_cv, method=calibration_method) if calibrate else base_model
    )
    final_model.fit(X_train, y_train)
    return final_model


def evaluate_final_model(model, X_test, y_test) -> dict:  # noqa: N803
    """
    Evaluate the trained champion on the held-out test set.

    Returns a dict with the three headline metrics (unchanged — these are
    what ``promote.py`` gates on), plus two Phase 11 additions that add
    evaluation rigor without touching that gate:

    - ``mcc``: Matthews Correlation Coefficient, robust to class imbalance.
    - ``pr_auc_per_class``: average precision per class from the model's
      raw ranking scores (``predict_proba``/``decision_function``) — more
      informative than ROC-AUC on the rare critical classes. ``None`` if
      the model exposes neither (shouldn't happen for this project's
      registry, but evaluate_final_model must never raise over it).

    Also returns the full per-class classification report and the
    confusion matrix — everything the Fase 3d MLflow step needs to log as
    metrics/artifacts.
    """
    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    classification_report_df = pd.DataFrame(report_dict).transpose().reset_index()
    classification_report_df = classification_report_df.rename(columns={"index": "label"})

    labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    confusion_matrix_df = pd.DataFrame(cm, index=labels, columns=labels)

    ranking_scores = get_ranking_scores(model, X_test)
    pr_auc_per_class = compute_pr_auc_per_class(y_test, ranking_scores, labels) if ranking_scores is not None else None

    # Brier score / ECE need REAL probabilities, not raw decision_function
    # scores — only computed when the model actually has predict_proba
    # (i.e. it went through calibration; see train_final_model(calibrate=True)).
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    brier_score = compute_brier_score(y_test, proba, labels) if proba is not None else None
    ece = compute_ece(y_test, proba, labels) if proba is not None else None

    return {
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "critical_recall": critical_recall_score(y_test, y_pred),
        "mcc": compute_mcc(y_test, y_pred),
        "pr_auc_per_class": pr_auc_per_class,
        "brier_score": brier_score,
        "ece": ece,
        "classification_report": classification_report_df,
        "confusion_matrix": confusion_matrix_df,
        "y_pred": y_pred,
    }
