"""
Sélection du champion, entraînement final et évaluation finale.

Extrait de ``notebooks/02_classical_ml.ipynb`` (cellules "PHASE 10 —
CHAMPION MODEL SELECTION" à "PHASE 17 — CONFUSION MATRIX"). Aucune
correction comportementale nécessaire ici — cette partie du notebook
utilisait déjà le ``nested_cv_summary`` corrigé (produit par
``benchmark.run_nested_cv_benchmark``, lui-même déjà corrigé à l'étape 3b)
et une casse de label cohérente partout.
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
    Choisit la paire championne (model, text_variant) : la ligne la mieux
    classée du résumé de nested CV, classée par robust_score (égalités
    départagées par critical_recall_mean puis f1_macro_mean).
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
    Paire (model, text_variant) classée deuxième dans le résumé de nested
    CV — le point de comparaison pour le test de significativité par
    bootstrap apparié de la Phase 11 (voir
    ``evaluation_metrics.paired_bootstrap_test``) : "le champion est-il
    réellement meilleur que le candidat suivant, ou est-ce juste un split
    chanceux ?". Retourne ``None`` si le résumé contient moins de deux
    lignes classées (par ex. un benchmark exécuté sur un seul candidat).
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
    Choisit les hyperparamètres du champion : le mode (valeur la plus
    fréquente) parmi les folds externes de la nested CV pour ce couple
    (model, text_variant).

    Utiliser le mode plutôt que, par exemple, le meilleur fold unique évite
    de surajuster les hyperparamètres finaux à un split externe chanceux.
    """
    params_list = nested_best_params[text_variant]["best_params"][model_name]
    extracted_params = [item["best_params"] for item in params_list if item["best_params"] is not None]

    if not extracted_params:
        raise ValueError(f"No nested params found for {model_name} / {text_variant}")

    param_counter = Counter(json.dumps(p, sort_keys=True) for p in extracted_params)
    most_common_params_str = param_counter.most_common(1)[0][0]
    return json.loads(most_common_params_str)


def _strip_embedding_caches(model) -> None:
    """
    Best-effort : après le fit, retire le embedding_cache d'entraînement de
    tout EmbeddingVectorizer (qui peut contenir des milliers de vecteurs) du
    modèle entraîné avant qu'il ne soit sérialisé/enregistré -- au moment de
    l'inférence, un cache neuf et minuscule est construit à chaque appel,
    donc rien n'est perdu, seule la masse désormais inutile du cache
    d'entraînement l'est.
    """
    candidates = [model]
    if hasattr(model, "calibrated_classifiers_"):
        for calibrated in model.calibrated_classifiers_:
            inner = getattr(calibrated, "estimator", None)
            if inner is not None:
                candidates.append(inner)

    for candidate in candidates:
        steps = getattr(candidate, "steps", None)
        if not steps:
            continue
        for _, step in steps:
            if hasattr(step, "embedding_cache"):
                step.embedding_cache = None


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
    Entraîne le pipeline champion (architecture depuis le registry, params
    ajustés) sur l'intégralité du training set.

    ``calibrate=True`` (Phase 11) enveloppe le pipeline cloné-puis-entraîné
    dans ``CalibratedClassifierCV`` (Platt scaling par défaut) avant le fit --
    c'est ce qui transforme la sortie brute ``decision_function`` de
    LinearSVC en de vraies probabilités bien calibrées (``predict_proba``),
    ce qui est un prérequis pour que le Brier score / ECE aient un sens, et
    qui donne aussi à l'API de vrais scores de confiance au lieu de
    l'approximation softmax actuelle sur ``decision_function``. Désactivé
    par défaut pour que tout autre appelant (par ex. le runner-up entraîné
    uniquement pour le test de significativité) conserve le comportement
    original, non calibré.
    """
    base_model = clone(model_registry[model_name]["pipeline"])
    base_model.set_params(**params)

    final_model = (
        CalibratedClassifierCV(base_model, cv=calibration_cv, method=calibration_method) if calibrate else base_model
    )
    final_model.fit(X_train, y_train)
    # NOTE : les caches d'embedding ne sont volontairement PAS retirés ici --
    # l'appelant (run_champion_stage) a encore besoin d'un cache vivant pour
    # exécuter model.predict(X_test) dans evaluate_final_model juste après.
    # Le retrait se fait explicitement, plus tard, uniquement pour le modèle
    # qui est effectivement sur le point d'être sérialisé/enregistré dans
    # MLflow -- voir le site d'appel de _strip_embedding_caches dans train.py.
    return final_model


def evaluate_final_model(model, X_test, y_test) -> dict:  # noqa: N803
    """
    Évalue le champion entraîné sur le test set mis de côté.

    Retourne un dict avec les trois métriques principales (inchangées — ce
    sont elles sur lesquelles ``promote.py`` se base pour son gate), plus
    deux ajouts de la Phase 11 qui renforcent la rigueur d'évaluation sans
    toucher à ce gate :

    - ``mcc`` : Matthews Correlation Coefficient, robuste au déséquilibre
      des classes.
    - ``pr_auc_per_class`` : average precision par classe à partir des
      scores de classement bruts du modèle (``predict_proba``/
      ``decision_function``) — plus informatif que le ROC-AUC sur les
      classes critiques rares. ``None`` si le modèle n'expose ni l'un ni
      l'autre (ne devrait pas arriver avec le registry de ce projet, mais
      evaluate_final_model ne doit jamais lever d'exception pour ça).

    Retourne aussi le rapport de classification complet par classe et la
    matrice de confusion — tout ce dont l'étape MLflow de la Phase 3d a
    besoin pour logger en tant que métriques/artefacts.
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

    # Le Brier score / ECE ont besoin de VRAIES probabilités, pas des scores
    # bruts de decision_function — calculés uniquement quand le modèle a
    # effectivement predict_proba (c.-à-d. qu'il est passé par la
    # calibration ; voir train_final_model(calibrate=True)).
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
