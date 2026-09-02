"""
Métriques d'évaluation supplémentaires et rigueur statistique (Phase 11, première tranche).

Vient s'ajouter à l'évaluation champion existante
(``champion.evaluate_final_model``) sans toucher à ses métriques
principales déjà établies (``f1_macro``, ``recall_macro``, ``critical_recall``) ni
à la porte de promotion dans ``promote.py``, qui ne regarde toujours que ces
trois métriques -- ce module ajoute de la rigueur d'évaluation, il ne change aucune
décision automatisée déjà en production.

- **MCC** (coefficient de corrélation de Matthews) : robuste au déséquilibre des classes,
  ne nécessite que les labels prédits -- aucune calibration requise.
- **PR-AUC par classe** : plus informatif que le ROC-AUC sur les classes rares et
  cliniquement critiques (Bipolaire, Schizophrénie). Calculé à partir des scores bruts
  de décision/probabilité (fonctionne pour LinearSVC via
  ``decision_function``, qui n'a pas de ``predict_proba``) -- c'est une
  métrique de classement, elle ne nécessite donc PAS que le modèle soit calibré.
- **Test de significativité par bootstrap apparié** : rééchantillonne l'ensemble de test mis de côté
  (mêmes lignes pour les deux modèles) pour estimer si un écart de métrique observé
  entre deux modèles est probablement réel ou juste du bruit -- répond à "est-ce
  vraiment un meilleur modèle, ou a-t-il simplement eu un split chanceux ?".

La calibration elle-même (Platt scaling / ``CalibratedClassifierCV``) et les
métriques qui en dépendent (Brier score, Expected Calibration Error) constituent une
tranche ultérieure et distincte de la Phase 11 -- les calculer sur un
softmax de decision_function non calibré (comme le fait déjà, de manière informelle,
le score de confiance de l'API) serait trompeur, donc elles ne sont délibérément
PAS incluses ici.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize


def compute_mcc(y_true, y_pred) -> float:
    """Coefficient de corrélation de Matthews -- robuste au déséquilibre des classes, contrairement à l'accuracy ou au F1 seul."""
    return float(matthews_corrcoef(y_true, y_pred))


def get_ranking_scores(model, X):  # noqa: N803
    """
    Scores de classement par classe, au mieux, pour le PR-AUC : ``predict_proba`` si
    le modèle le possède (LogisticRegression, MultinomialNB), sinon
    ``decision_function`` (LinearSVC). Retourne ``None`` si aucun des deux n'existe,
    afin que les appelants puissent ignorer le PR-AUC proprement plutôt que de lever une exception.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_pr_auc_per_class(y_true, scores: np.ndarray, labels: list[str]) -> dict[str, float]:
    """
    Précision moyenne (aire sous la courbe précision-rappel) par classe,
    à partir de la sortie de ``get_ranking_scores``. Une métrique de classement -- valide sur des scores
    decision_function bruts et non calibrés, contrairement au Brier score/ECE.

    Gère explicitement le cas binaire : ``label_binarize`` réduit deux
    classes à une seule colonne (la classe positive = ``labels[1]``),
    contrairement à la forme une-colonne-par-classe qu'il retourne pour 3 classes ou plus.
    """
    y_true_binarized = label_binarize(y_true, classes=labels)
    scores = np.asarray(scores)

    if len(labels) == 2:
        positive_scores = scores[:, 1] if scores.ndim == 2 else scores
        positive_ap = float(average_precision_score(y_true_binarized[:, 0], positive_scores))
        negative_ap = float(average_precision_score(1 - y_true_binarized[:, 0], -positive_scores))
        return {labels[0]: negative_ap, labels[1]: positive_ap}

    return {
        label: float(average_precision_score(y_true_binarized[:, i], scores[:, i]))
        for i, label in enumerate(labels)
    }


def compute_brier_score(y_true, proba: np.ndarray, labels: list[str]) -> float:
    """
    Score de Brier multiclasse : distance quadratique moyenne entre le vecteur de
    probabilité prédit et le label réel en one-hot, moyennée sur les échantillons.
    Plus bas est meilleur (0 = parfait). Nécessite de VRAIES probabilités
    (``predict_proba``) -- dénué de sens sur des scores decision_function bruts,
    ce qui explique précisément pourquoi ce projet ne le calcule qu'après calibration.
    """
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    one_hot = pd.get_dummies(y_true_series).reindex(columns=labels, fill_value=0).to_numpy(dtype=float)
    proba = np.asarray(proba)
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def compute_ece(y_true, proba: np.ndarray, labels: list[str], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (top-label, bins de largeur égale) : regroupe les
    prédictions par la confiance de leur classe la plus probable, et moyenne
    |accuracy - confiance| au sein de chaque bin, pondéré par la taille du bin. Plus bas
    est meilleur (0 = parfaitement calibré). Nécessite également de vraies
    probabilités, même mise en garde que ``compute_brier_score``.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    confidences = proba.max(axis=1)
    predictions = np.array(labels)[proba.argmax(axis=1)]
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _default_metric(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def paired_bootstrap_test(
    y_true,
    y_pred_a,
    y_pred_b,
    metric_fn=_default_metric,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Test de significativité par bootstrap apparié sur les MÊMES lignes mises de côté pour deux
    modèles (A, B). Rééchantillonne les indices de lignes avec remise ``n_bootstrap``
    fois, recalcule ``metric_fn`` pour les deux modèles sur chaque rééchantillon, et
    rapporte un IC à 95 % + une p-value bilatérale sur la différence observée
    (A - B) -- c.-à-d. si un écart de métrique observé est probablement réel ou juste
    du bruit provenant du split de test particulier.

    ``y_true``, ``y_pred_a``, ``y_pred_b`` doivent déjà être alignés sur les
    mêmes lignes (par ex. champion vs. dauphin, tous deux évalués sur le même
    X_test).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    if not (len(y_pred_a) == n and len(y_pred_b) == n):
        raise ValueError("y_true, y_pred_a and y_pred_b must have the same length (paired rows).")

    observed_diff = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)

    rng = np.random.RandomState(random_state)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])

    ci_lower, ci_upper = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))

    # Test de significativité bootstrap bilatéral : deux fois la proportion de rééchantillons qui tombent
    # sur (ou au-delà) le côté opposé de zéro par rapport à la différence observée.
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }
