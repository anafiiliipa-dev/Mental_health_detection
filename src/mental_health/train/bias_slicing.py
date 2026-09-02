"""
Bias slicing : évaluation par sous-groupe (Phase 11, slice restante :
"évaluation par sous-groupes (bias slicing)").

Ce dataset (``cleaning.py`` : ``body`` + ``category``, scrapé depuis des
forums publics de santé mentale) ne porte aucun attribut démographique
(âge, genre, locale, ...) sur lequel slicer — il n'y a rien à fabriquer ici,
et inventer un proxy pour un attribut protégé que les données ne contiennent
pas réellement serait pire que de ne pas slicer du tout. Le slicing se fait
plutôt sur des attributs que les données possèdent réellement et qui sont
des modes d'échec connus pour les classifieurs de texte :

- **performance par classe (label)** : le modèle échoue-t-il discrètement
  sur une ou plusieurs des sept catégories de diagnostic pendant que les
  métriques macro globales semblent correctes ? C'est la slice la plus
  importante cliniquement, puisque les deux labels critiques (Bipolar,
  Schizophrenia) sont aussi parmi les plus rares.
- **longueur du texte** : les posts courts portent moins de signal que les
  longs — un modèle qui ne fonctionne que sur des posts verbeux est un
  véritable mode d'échec, auparavant non documenté, pour un outil de
  triage censé fonctionner sur ce qu'un utilisateur tape réellement.

Comme ``robustness.py`` et le reste des ajouts d'évaluation de la Phase 11,
ceci est purement du reporting/diagnostic — cela n'alimente pas la sélection
du champion ni le gate de promotion.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score, recall_score

from mental_health.config.paths import CRITICAL_LABELS

# ============================================================
# Attribution des slices
# ============================================================

DEFAULT_LENGTH_BIN_LABELS = ["short", "medium", "long"]


def assign_length_slices(texts: pd.Series, n_bins: int = 3, labels: list[str] | None = None) -> pd.Series:
    """
    Range chaque texte dans une slice de longueur à fréquence égale
    (quantile) selon le nombre de mots — par exemple des terciles
    ``["short", "medium", "long"]`` par défaut. Le binning par quantile
    (et non à largeur égale) est utilisé pour que chaque slice ait une
    taille d'échantillon comparable, ce qui rend une comparaison de recall
    par slice pertinente plutôt que bruitée sur un bin quasi vide.
    """
    labels = labels if labels is not None else DEFAULT_LENGTH_BIN_LABELS[:n_bins]
    if len(labels) != n_bins:
        raise ValueError(f"Expected {n_bins} labels, got {len(labels)}: {labels}")

    word_counts = pd.Series(texts).reset_index(drop=True).map(lambda t: len(str(t).split()))
    try:
        return pd.qcut(word_counts, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        # Entrée dégénérée (par ex. tous les textes de même longueur, ou trop
        # peu de lignes) — qcut ne peut pas former n_bins bornes distinctes.
        # On retombe sur une slice unique plutôt que de lever une exception,
        # puisque c'est un outil de diagnostic, pas une étape d'entraînement
        # qui doit être stricte.
        return pd.Series([labels[0]] * len(word_counts))


# ============================================================
# Évaluation par slice
# ============================================================


def _slice_metrics(y_true, y_pred) -> dict:
    # IMPORTANT : labels est fixé aux classes réellement présentes dans le
    # y_true de cette slice, et non laissé au défaut de sklearn (l'union de
    # y_true et y_pred). Laissé au défaut, une slice de classe -- où y_true
    # est un SEUL label par construction -- ferait une macro-moyenne sur
    # tous les autres labels que le modèle a (à tort) prédits dans cette
    # slice aussi, chacun noté 0 en recall par zero_division puisqu'il n'a
    # aucune instance vraie ici. Cela tire silencieusement le score de
    # chaque slice vers un quasi-bruit, indépendamment de la performance
    # réelle du modèle sur son vrai label -- ce n'est plus du tout une
    # mesure de performance de sous-groupe, juste un artefact de à quel
    # point les suppositions hors-label du modèle étaient fausses.
    labels = sorted(pd.Series(y_true).unique())
    return {
        "support": len(y_true),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def evaluate_slices(y_true, y_pred, slice_labels, slice_column_name: str = "slice") -> pd.DataFrame:
    """
    Évaluation générique par slice : regroupe ``(y_true, y_pred)`` par
    ``slice_labels`` (n'importe quelle clé de regroupement de même longueur
    — un label de classe, un bucket de longueur, une variante de texte, ...)
    et rapporte le support, le f1_macro et le recall_macro par slice, triés
    du pire f1_macro au meilleur pour que le sous-groupe le plus faible soit
    immédiatement visible.
    """
    df = pd.DataFrame({
        slice_column_name: pd.Series(slice_labels).reset_index(drop=True),
        "y_true": pd.Series(y_true).reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
    })

    rows = [
        {slice_column_name: slice_value, **_slice_metrics(group["y_true"], group["y_pred"])}
        for slice_value, group in df.groupby(slice_column_name, observed=True)
    ]
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=True).reset_index(drop=True)


def evaluate_class_slices(y_true, y_pred) -> pd.DataFrame:
    """
    Recall par label (catégorie de diagnostic) — la vue de bias-slicing la
    plus directe pour ce projet : une classe donnée, critique ou non, est-elle
    sous-servie par rapport aux autres ? Signale explicitement les labels
    cliniquement critiques (``CRITICAL_LABELS``) pour qu'un relecteur n'ait
    pas à croiser avec une liste séparée.
    """
    report = evaluate_slices(y_true, y_pred, y_true, slice_column_name="label")
    report = report.rename(columns={"f1_macro": "f1", "recall_macro": "recall"})
    report["is_critical"] = report["label"].isin(CRITICAL_LABELS)
    return report[["label", "is_critical", "support", "recall", "f1"]]


def evaluate_length_slices(texts, y_true, y_pred, n_bins: int = 3) -> pd.DataFrame:
    """Métriques macro par bucket de longueur (short/medium/long, selon le nombre de mots)."""
    length_slices = assign_length_slices(texts, n_bins=n_bins)
    return evaluate_slices(y_true, y_pred, length_slices, slice_column_name="length_bucket")


def summarize_fairness_gap(slice_report: pd.DataFrame, metric_col: str = "recall") -> dict:
    """
    Le chiffre de fairness principal pour un rapport de slice : l'écart entre
    la slice la plus performante et la moins performante sur ``metric_col``,
    plus quelle slice est la pire — c'est le chiffre qui devrait attirer
    l'attention même quand la métrique macro globale semble correcte,
    puisqu'un grand écart peut se cacher derrière une moyenne saine.
    """
    if slice_report.empty or metric_col not in slice_report.columns:
        return {"gap": 0.0, "worst_slice": None, "best_slice": None}

    worst_row = slice_report.loc[slice_report[metric_col].idxmin()]
    best_row = slice_report.loc[slice_report[metric_col].idxmax()]
    slice_col = next(c for c in slice_report.columns if c not in {"support", "recall", "f1", "is_critical"})

    return {
        "gap": float(best_row[metric_col] - worst_row[metric_col]),
        "worst_slice": worst_row[slice_col],
        "worst_value": float(worst_row[metric_col]),
        "best_slice": best_row[slice_col],
        "best_value": float(best_row[metric_col]),
    }


def run() -> dict[str, pd.DataFrame]:
    """
    Charge le champion actuel (alias "production" enregistré, avec repli sur
    "staging"), reconstruit son split de test, et écrit les rapports de
    slice par classe et par bucket de longueur dans
    ``paths.BIAS_SLICING_REPORT_PATH`` (concaténés, avec une colonne
    ``slice_type`` distinguant les deux).
    """
    import logging

    import mlflow
    from dotenv import load_dotenv

    load_dotenv()

    from mental_health.config.mlflow_config import (
        MLFLOW_REGISTERED_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        PRODUCTION_ALIAS,
        STAGING_ALIAS,
    )
    from mental_health.config.paths import BIAS_SLICING_REPORT_PATH, DEFAULT_CLEAN_DATA_PATH
    from mental_health.train.train import build_splits

    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        version = client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        alias_used = PRODUCTION_ALIAS
    except mlflow.exceptions.MlflowException:
        version = client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, STAGING_ALIAS)
        alias_used = STAGING_ALIAS

    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{alias_used}")
    logger.info("Loaded '%s' v%s (alias=%s) for bias slicing", MLFLOW_REGISTERED_MODEL_NAME, version.version, alias_used)

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    splits = build_splits(df)
    X_test, y_test = splits["raw"]["X_test"], splits["raw"]["y_test"]
    y_pred = model.predict(X_test)

    class_report = evaluate_class_slices(y_test, y_pred)
    length_report = evaluate_length_slices(X_test, y_test, y_pred)

    class_gap = summarize_fairness_gap(class_report.rename(columns={"label": "slice"}))
    length_gap = summarize_fairness_gap(length_report.rename(columns={"length_bucket": "slice", "recall_macro": "recall"}))
    logger.info("Fairness gap by class: %s", class_gap)
    logger.info("Fairness gap by text length: %s", length_gap)

    class_report.insert(0, "slice_type", "class")
    length_report.insert(0, "slice_type", "length")
    combined = pd.concat([class_report, length_report], ignore_index=True)
    BIAS_SLICING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BIAS_SLICING_REPORT_PATH, index=False)
    logger.info("Bias slicing report written to %s:\n%s", BIAS_SLICING_REPORT_PATH, combined.to_string(index=False))

    return {"class_slices": class_report, "length_slices": length_report}


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
