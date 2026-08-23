"""
Cross-validation benchmark utilities: light CV and nested CV.

Extracted and corrected from ``notebooks/02_classical_ml.ipynb`` (cells
"PHASE 5 — CLINICAL SCORING", "sample_param_grid", "PHASE 6 —
run_light_cv_benchmark" and "run_light_nested_cv_benchmark").

Corrections relative to the original notebook:

1. ``critical_recall_score`` used a hardcoded ``CRITICAL_LABELS = ["Bipolar",
   "schizophrenia"]`` (lowercase schizophrenia — the same casing bug found
   elsewhere in the audit). It now imports ``CRITICAL_LABELS`` from
   ``mental_health.config.paths`` instead, matching the corrected label
   casing produced by ``cleaning.py``.
2. The "robust score" formula was NOT consistent between the light CV and
   nested CV sections of the notebook: light CV weighted
   ``0.4 * f1_macro + 0.3 * recall_macro + 0.3 * critical_recall``, while
   nested CV weighted ``0.4 * critical_recall + 0.3 * recall_macro +
   0.3 * f1_macro`` — the F1 and critical-recall weights were swapped. This
   didn't affect the champion selection itself (only ``nested_cv`` results
   are used to pick the champion), but it's a real inconsistency. Both
   benchmarks here now share a single ``compute_robust_score`` function
   using the nested-CV weighting, since that is the one that actually
   drove the champion decision.
"""
from __future__ import annotations

import random
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from mental_health.config.paths import CRITICAL_LABELS
from mental_health.train.model_registry import RANDOM_STATE

# Weighting used to combine the three metrics into a single ranking score.
# critical_recall is weighted highest because missing a critical-class post
# (Bipolar / Schizophrenia) is the costliest error for this project.
ROBUST_SCORE_WEIGHTS = {
    "critical_recall": 0.4,
    "recall_macro": 0.3,
    "f1_macro": 0.3,
}


def critical_recall_score(y_true, y_pred, critical_labels: list[str] = CRITICAL_LABELS) -> float:
    """Mean recall across the clinically critical labels only."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    recalls = [report[label]["recall"] for label in critical_labels if label in report]
    return float(np.mean(recalls)) if recalls else 0.0


def compute_robust_score(f1_macro: float, recall_macro: float, critical_recall: float) -> float:
    """Single weighted score combining macro-F1, macro-recall and critical recall."""
    return (
        ROBUST_SCORE_WEIGHTS["critical_recall"] * critical_recall
        + ROBUST_SCORE_WEIGHTS["recall_macro"] * recall_macro
        + ROBUST_SCORE_WEIGHTS["f1_macro"] * f1_macro
    )


def sample_param_grid(param_grid: dict, max_candidates: int = 4, random_state: int = RANDOM_STATE) -> list[dict]:
    """Sample up to ``max_candidates`` combinations from a sklearn-style param grid."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    all_candidates = [dict(zip(keys, combo, strict=True)) for combo in product(*values)]

    if len(all_candidates) <= max_candidates:
        return all_candidates

    rng = random.Random(random_state)
    return rng.sample(all_candidates, max_candidates)


def _rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.sort_values(
        by=["robust_score", "critical_recall_mean", "f1_macro_mean"],
        ascending=False,
    ).reset_index(drop=True)
    summary["robust_rank"] = np.arange(1, len(summary) + 1)
    return summary


def run_light_cv_benchmark(
    X_train,
    y_train,
    model_registry: dict,
    n_splits: int = 3,
    max_candidates: int = 4,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Quick screening benchmark: for each candidate model, sample a few
    hyperparameter combinations, pick the best by mean F1 on a light CV,
    then report per-fold metrics for that best configuration.
    """
    X_train = pd.Series(X_train).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = []
    best_params_log = {}

    for model_name, model_spec in model_registry.items():
        pipeline = model_spec["pipeline"]
        param_grid = model_spec["param_grid"]
        sampled_candidates = sample_param_grid(param_grid, max_candidates=max_candidates, random_state=random_state)

        best_score = -np.inf
        best_params = None

        for params in sampled_candidates:
            fold_f1_scores = []
            for train_idx, valid_idx in cv.split(X_train, y_train):
                model = clone(pipeline)
                model.set_params(**params)
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                y_pred = model.predict(X_train.iloc[valid_idx])
                fold_f1_scores.append(f1_score(y_train.iloc[valid_idx], y_pred, average="macro", zero_division=0))

            mean_f1 = float(np.mean(fold_f1_scores))
            if mean_f1 > best_score:
                best_score = mean_f1
                best_params = params

        best_params_log[model_name] = best_params

        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
            model = clone(pipeline)
            model.set_params(**best_params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_pred = model.predict(X_train.iloc[valid_idx])
            y_val = y_train.iloc[valid_idx]

            fold_rows.append({
                "model": model_name,
                "fold": fold_id,
                "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
                "critical_recall": critical_recall_score(y_val, y_pred),
            })

    fold_results = pd.DataFrame(fold_rows)

    summary = (
        fold_results.groupby("model")
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            recall_macro_mean=("recall_macro", "mean"),
            critical_recall_mean=("critical_recall", "mean"),
            f1_macro_std=("f1_macro", "std"),
            recall_macro_std=("recall_macro", "std"),
            critical_recall_std=("critical_recall", "std"),
        )
        .reset_index()
    )
    for col in ["f1_macro_std", "recall_macro_std", "critical_recall_std"]:
        summary[col] = summary[col].fillna(0)

    summary["robust_score"] = compute_robust_score(
        summary["f1_macro_mean"], summary["recall_macro_mean"], summary["critical_recall_mean"]
    )
    summary = _rank_summary(summary)

    return fold_results, summary, best_params_log


def run_nested_cv_benchmark(
    X_train,
    y_train,
    model_registry: dict,
    outer_splits: int = 3,
    inner_splits: int = 2,
    max_candidates: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Nested CV: an outer loop estimates generalisation performance, an inner
    loop selects hyperparameters (by ``compute_robust_score``) without ever
    touching the outer test fold. This is the benchmark that actually
    decides the champion model.
    """
    X_train = pd.Series(X_train).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    nested_rows = []
    nested_best_params = {}

    for model_name, model_spec in model_registry.items():
        pipeline = model_spec["pipeline"]
        param_grid = model_spec["param_grid"]
        nested_best_params[model_name] = []

        for outer_fold, (dev_idx, test_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
            X_dev, y_dev = X_train.iloc[dev_idx], y_train.iloc[dev_idx]
            X_outer_test, y_outer_test = X_train.iloc[test_idx], y_train.iloc[test_idx]

            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
            sampled_candidates = sample_param_grid(param_grid, max_candidates=max_candidates, random_state=random_state)

            best_score = -np.inf
            best_params = None

            for params in sampled_candidates:
                inner_scores = []
                for inner_train_idx, inner_valid_idx in inner_cv.split(X_dev, y_dev):
                    model = clone(pipeline)
                    model.set_params(**params)
                    model.fit(X_dev.iloc[inner_train_idx], y_dev.iloc[inner_train_idx])
                    y_pred_inner = model.predict(X_dev.iloc[inner_valid_idx])
                    y_inner_valid = y_dev.iloc[inner_valid_idx]

                    inner_scores.append(compute_robust_score(
                        f1_score(y_inner_valid, y_pred_inner, average="macro", zero_division=0),
                        recall_score(y_inner_valid, y_pred_inner, average="macro", zero_division=0),
                        critical_recall_score(y_inner_valid, y_pred_inner),
                    ))

                mean_inner_score = float(np.mean(inner_scores))
                if mean_inner_score > best_score:
                    best_score = mean_inner_score
                    best_params = params

            nested_best_params[model_name].append({"outer_fold": outer_fold, "best_params": best_params})

            final_model = clone(pipeline)
            final_model.set_params(**best_params)
            final_model.fit(X_dev, y_dev)
            y_outer_pred = final_model.predict(X_outer_test)

            nested_rows.append({
                "model": model_name,
                "outer_fold": outer_fold,
                "f1_macro": f1_score(y_outer_test, y_outer_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_outer_test, y_outer_pred, average="macro", zero_division=0),
                "critical_recall": critical_recall_score(y_outer_test, y_outer_pred),
            })

    nested_fold_results = pd.DataFrame(nested_rows)

    nested_summary = (
        nested_fold_results.groupby("model")
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            recall_macro_mean=("recall_macro", "mean"),
            critical_recall_mean=("critical_recall", "mean"),
            f1_macro_std=("f1_macro", "std"),
            recall_macro_std=("recall_macro", "std"),
            critical_recall_std=("critical_recall", "std"),
        )
        .reset_index()
    )
    for col in ["f1_macro_std", "recall_macro_std", "critical_recall_std"]:
        nested_summary[col] = nested_summary[col].fillna(0)

    nested_summary["robust_score"] = compute_robust_score(
        nested_summary["f1_macro_mean"], nested_summary["recall_macro_mean"], nested_summary["critical_recall_mean"]
    )
    nested_summary = _rank_summary(nested_summary)

    return nested_fold_results, nested_summary, nested_best_params
