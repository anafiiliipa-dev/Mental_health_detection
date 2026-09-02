"""
Utilitaires de benchmark de validation croisée : CV légère et nested CV.

Extrait et corrigé de ``notebooks/02_classical_ml.ipynb`` (cellules
"PHASE 5 — CLINICAL SCORING", "sample_param_grid", "PHASE 6 —
run_light_cv_benchmark" et "run_light_nested_cv_benchmark").

Corrections par rapport au notebook original :

1. ``critical_recall_score`` utilisait un ``CRITICAL_LABELS = ["Bipolar",
   "schizophrenia"]`` codé en dur (schizophrenia en minuscules — le même bug
   de casse trouvé ailleurs dans l'audit). Il importe désormais
   ``CRITICAL_LABELS`` depuis ``mental_health.config.paths``, en accord
   avec la casse de label corrigée produite par ``cleaning.py``.
2. La formule du "robust score" n'était PAS cohérente entre les sections CV
   légère et nested CV du notebook : la CV légère pondérait
   ``0.4 * f1_macro + 0.3 * recall_macro + 0.3 * critical_recall``, tandis que
   la nested CV pondérait ``0.4 * critical_recall + 0.3 * recall_macro +
   0.3 * f1_macro`` — les poids du F1 et du critical-recall étaient inversés.
   Cela n'affectait pas la sélection du champion elle-même (seuls les
   résultats de ``nested_cv`` sont utilisés pour choisir le champion), mais
   c'est une vraie incohérence. Les deux benchmarks ici partagent désormais
   une seule fonction ``compute_robust_score`` utilisant la pondération de
   la nested CV, puisque c'est elle qui a réellement déterminé la décision
   du champion.
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

# Pondération utilisée pour combiner les trois métriques en un seul score de classement.
# critical_recall a le poids le plus élevé car manquer un post d'une classe critique
# (Bipolar / Schizophrenia) est l'erreur la plus coûteuse pour ce projet.
ROBUST_SCORE_WEIGHTS = {
    "critical_recall": 0.4,
    "recall_macro": 0.3,
    "f1_macro": 0.3,
}


def critical_recall_score(y_true, y_pred, critical_labels: list[str] = CRITICAL_LABELS) -> float:
    """Recall moyen sur les seuls labels cliniquement critiques."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    recalls = [report[label]["recall"] for label in critical_labels if label in report]
    return float(np.mean(recalls)) if recalls else 0.0


def compute_robust_score(f1_macro: float, recall_macro: float, critical_recall: float) -> float:
    """Score unique pondéré combinant macro-F1, macro-recall et critical recall."""
    return (
        ROBUST_SCORE_WEIGHTS["critical_recall"] * critical_recall
        + ROBUST_SCORE_WEIGHTS["recall_macro"] * recall_macro
        + ROBUST_SCORE_WEIGHTS["f1_macro"] * f1_macro
    )


def sample_param_grid(param_grid: dict, max_candidates: int = 4, random_state: int = RANDOM_STATE) -> list[dict]:
    """Échantillonne jusqu'à ``max_candidates`` combinaisons depuis une grille de paramètres de style sklearn."""
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
    Benchmark de screening rapide : pour chaque modèle candidat, échantillonne
    quelques combinaisons d'hyperparamètres, choisit la meilleure par F1 moyen
    sur une CV légère, puis rapporte les métriques par fold pour cette
    meilleure configuration.
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
    Nested CV : une boucle externe estime la performance de généralisation, une
    boucle interne sélectionne les hyperparamètres (via ``compute_robust_score``)
    sans jamais toucher au fold de test externe. C'est ce benchmark qui décide
    réellement du modèle champion.
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
