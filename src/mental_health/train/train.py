"""
Point d'entrée d'entraînement de bout en bout : benchmarker tous les candidats, sélectionner le
champion, l'entraîner, l'évaluer, et tout logger dans MLflow.

Extrait et corrigé de ``notebooks/02_classical_ml.ipynb`` (l'orchestration
globale : de PHASE 1 train/test split jusqu'au résumé de PHASE 20),
en assemblant ``model_registry.py``, ``benchmark.py`` et
``champion.py``. C'est le livrable de la Phase 3 de la roadmap :
"instrumenter le script de training avec MLflow".

Choix de conception par rapport au notebook original, rendus explicites plutôt
que portés silencieusement :

1. **Seul le benchmark nested CV est exécuté ici, pas la CV légère.** Dans le
   notebook, la CV légère est une étape rapide de pré-sélection dont les résultats
   ne sont en réalité jamais utilisés par la sélection du champion (``select_champion_config``
   ne lit que ``nested_cv_summary``). L'exécuter en production doublerait
   à peu près le temps d'entraînement pour aucun effet sur le résultat, donc ce
   script n'exécute que la nested CV — le seul benchmark qui décide réellement.
   ``run_light_cv_benchmark`` reste disponible dans ``benchmark.py`` pour une
   exploration ponctuelle dans un notebook si utile.
2. **Store de tracking MLflow** : backend SQLite local
   (``<project_root>/mlflow.db``) avec les artefacts stockés sous
   ``<project_root>/mlruns`` (convenu avec Ana — pas de dépendance Postgres/S3
   avant l'arrivée de Docker en Phase 7). SQLite plutôt que le simple store
   filesystem car MLflow 3.x place le backend filestore brut en mode
   maintenance et recommande un backend base de données même pour un usage local ;
   SQLite signifie aussi que l'éventuelle migration vers Postgres (Phase 7) sera un
   simple changement de chaîne de connexion, pas une migration de données. Migrer plus tard
   signifiera simplement pointer ``mlflow.set_tracking_uri`` vers le nouveau backend ; rien
   d'autre ne change.
3. **Un run MLflow par variante de texte** pendant le benchmark nested CV
   (``nested_cv_raw``, ``nested_cv_masked``), plus un run final
   ``champion_final`` qui logue le modèle entraîné lui-même via
   ``mlflow.sklearn.log_model``. Cela reflète la façon dont le notebook sépare
   conceptuellement "comparer les candidats" de "entraîner le gagnant".
4. **Hash du dataset loggé comme paramètre MLflow** (SHA256, 16 premiers caractères) —
   la recommandation de l'Étape F de l'audit, offrant une traçabilité de base
   des versions de dataset sans introduire DVC.
5. **Model Registry (Phase 5)** : le champion est enregistré sous
   ``MLFLOW_REGISTERED_MODEL_NAME`` et aliasé ``"staging"`` à chaque run —
   jamais ``"production"`` directement. La promotion est une décision séparée
   et explicite prise par ``promote.py`` selon des seuils documentés, pas une
   action automatique ou manuelle dans l'UI (selon la recommandation de gouvernance
   de l'audit : "qui a le droit de promouvoir un modèle ?"). Les stages
   (l'ancien mécanisme de MLflow) sont dépréciés depuis MLflow 2.9 au profit des
   alias, donc ceci utilise directement les alias plutôt que de s'appuyer sur une
   API déjà programmée pour être supprimée.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Doit s'exécuter avant que l'import mlflow_config ci-dessous ne lise MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT depuis l'environnement (par ex. un backend d'équipe partagé
# au lieu du SQLite local par défaut) — voir la docstring de mlflow_config.py.
load_dotenv()

from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    STAGING_ALIAS,
)
from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH, MODEL_COMPARISON_PATH  # noqa: E402
from mental_health.data.cleaning import MASKED_COL, TARGET_COL, TEXT_COL  # noqa: E402
from mental_health.train.benchmark import run_nested_cv_benchmark  # noqa: E402
from mental_health.train.champion import (  # noqa: E402
    _strip_embedding_caches,
    evaluate_final_model,
    select_champion_config,
    select_champion_params,
    select_runner_up_config,
    train_final_model,
)
from mental_health.train.embedding_wrapper import precompute_dataset_embeddings  # noqa: E402
from mental_health.train.evaluation_metrics import paired_bootstrap_test  # noqa: E402
from mental_health.train.model_registry import (  # noqa: E402
    RANDOM_STATE,
    build_model_registry,
    compute_boosted_class_weights,
)

logger = logging.getLogger(__name__)

# MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT, MLFLOW_EXPERIMENT_NAME et
# MLFLOW_REGISTERED_MODEL_NAME vivent dans mental_health.config.mlflow_config —
# la source de vérité unique partagée avec promote.py et le service FastAPI
# (Phase 6), afin que le code de service n'ait pas besoin d'importer ce module d'entraînement.
TEST_SIZE = 0.2
TEXT_VARIANTS = ["raw"]  # temporairement réduit pour la rapidité de la démo -- restaurer ["raw", "masked"] pour le benchmark officiel
VARIANT_COLUMNS = {"raw": TEXT_COL, "masked": MASKED_COL}


def _get_or_create_experiment(name: str, artifact_location: str) -> str:
    """Récupère l'expérience par son nom, la crée (avec un emplacement d'artefact explicite) si elle n'existe pas."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)


def compute_dataset_hash(path: Path) -> str:
    """16 premiers caractères hexadécimaux du SHA256 du fichier de dataset — un marqueur de version léger et sans DVC."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_splits(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> dict:
    """
    Construit UN seul split train/test stratifié sur les indices de lignes, puis l'applique
    aux deux variantes de texte (raw, masked) — garantissant que les deux variantes sont
    comparées exactement sur les mêmes lignes.
    """
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state
    )

    splits = {}
    for variant, column in VARIANT_COLUMNS.items():
        splits[variant] = {
            "X_train": df.loc[train_idx, column].reset_index(drop=True),
            "X_test": df.loc[test_idx, column].reset_index(drop=True),
            "y_train": df.loc[train_idx, TARGET_COL].reset_index(drop=True),
            "y_test": df.loc[test_idx, TARGET_COL].reset_index(drop=True),
        }
    return splits


def run_benchmark_stage(
    splits: dict, dataset_hash: str, embedding_cache: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Exécute le benchmark nested CV pour chaque variante de texte, en loggant un run
    MLflow par variante. Retourne le résumé combiné (toutes variantes, tous
    modèles) et la sortie brute des meilleurs paramètres nécessaire à la sélection du champion.

    ``embedding_cache``, lorsqu'il est fourni, n'est passé au registre que pour la
    variante de texte "raw" (Phase 11 : les candidats à base d'embeddings sont limités au
    texte raw uniquement, laissant la comparaison de la variante masked inchangée).
    """
    summaries = []
    nested_outputs = {}

    for text_variant in TEXT_VARIANTS:
        X_train = splits[text_variant]["X_train"]
        y_train = splits[text_variant]["y_train"]

        class_weights = compute_boosted_class_weights(y_train)
        variant_embedding_cache = embedding_cache if text_variant == "raw" else None
        registry = build_model_registry(class_weights, embedding_cache=variant_embedding_cache)

        with mlflow.start_run(run_name=f"nested_cv_{text_variant}"):
            mlflow.log_param("stage", "nested_cv_benchmark")
            mlflow.log_param("text_variant", text_variant)
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("candidate_models", list(registry.keys()))

            logger.info("Running nested CV benchmark — variant=%s", text_variant)
            _, summary, best_params = run_nested_cv_benchmark(
                X_train, y_train, registry, outer_splits=2, inner_splits=2, max_candidates=2
            )  # temporairement réduit pour la rapidité de la démo -- restaurer les valeurs par défaut pour le benchmark officiel
            summary = summary.copy()
            summary["text_variant"] = text_variant

            for _, row in summary.iterrows():
                model_name = row["model"]
                mlflow.log_metric(f"{model_name}__f1_macro", row["f1_macro_mean"])
                mlflow.log_metric(f"{model_name}__recall_macro", row["recall_macro_mean"])
                mlflow.log_metric(f"{model_name}__critical_recall", row["critical_recall_mean"])
                mlflow.log_metric(f"{model_name}__robust_score", row["robust_score"])

            with tempfile.TemporaryDirectory() as tmp_dir:
                summary_path = Path(tmp_dir) / f"nested_cv_summary_{text_variant}.csv"
                summary.to_csv(summary_path, index=False)
                mlflow.log_artifact(str(summary_path))

        summaries.append(summary)
        nested_outputs[text_variant] = {"best_params": best_params}
        logger.info("Nested CV done — variant=%s, best robust_score=%.4f", text_variant, summary["robust_score"].max())

    nested_summary = pd.concat(summaries, ignore_index=True)
    return nested_summary, nested_outputs


def _train_and_evaluate(
    registry: dict, model_name: str, params: dict, splits: dict, text_variant: str, calibrate: bool = False
) -> tuple:
    """
    Entraîne un candidat (model_name/text_variant) sur son split d'entraînement
    complet et l'évalue sur son propre split de test.

    ``calibrate=True`` le fait passer par ``CalibratedClassifierCV`` (Platt
    scaling) — utilisé uniquement pour le champion (voir ``run_champion_stage``),
    puisque c'est le modèle qui finit réellement par être servi ; le dauphin
    entraîné ici uniquement pour le test de significativité reste non calibré.
    """
    X_train = splits[text_variant]["X_train"]
    y_train = splits[text_variant]["y_train"]
    X_test = splits[text_variant]["X_test"]
    y_test = splits[text_variant]["y_test"]

    model = train_final_model(registry, model_name, params, X_train, y_train, calibrate=calibrate)
    eval_result = evaluate_final_model(model, X_test, y_test)
    return model, eval_result, X_train, X_test, y_test


def evaluate_all_candidates(
    nested_summary: pd.DataFrame, nested_outputs: dict, splits: dict, embedding_cache: dict | None = None
) -> pd.DataFrame:
    """
    Phase 11 : entraîne + évalue TOUS les candidats (model, text_variant) issus
    du benchmark nested CV sur leur propre split de test mis de côté -- pas seulement le
    champion et le dauphin -- avec la suite complète de métriques de rigueur
    (f1_macro, recall_macro, critical_recall, mcc, brier_score, ece).

    Toutes les lignes sont non calibrées (contrairement au champion effectivement
    enregistré par run_champion_stage), afin que chaque candidat soit comparé sur un
    même pied d'égalité ; les chiffres calibrés propres au champion sont loggés
    séparément dans le run MLflow champion_final. C'est le "tableau de
    résultats" complet de comparaison sur tous les candidats que le registre
    contient actuellement -- y compris les futurs ajouts (embeddings, DistilBERT) une fois
    câblés dans build_model_registry / l'étape de benchmark de la même
    façon que XGBoost/LightGBM l'ont été.
    """
    rows = []
    for _, summary_row in nested_summary.iterrows():
        model_name = summary_row["model"]
        text_variant = summary_row["text_variant"]
        params = select_champion_params(nested_outputs, text_variant, model_name)
        class_weights = compute_boosted_class_weights(splits[text_variant]["y_train"])
        variant_embedding_cache = embedding_cache if text_variant == "raw" else None
        registry = build_model_registry(class_weights, embedding_cache=variant_embedding_cache)

        _, eval_result, _, _, _ = _train_and_evaluate(registry, model_name, params, splits, text_variant)
        rows.append(
            {
                "model": model_name,
                "text_variant": text_variant,
                "f1_macro": eval_result["f1_macro"],
                "recall_macro": eval_result["recall_macro"],
                "critical_recall": eval_result["critical_recall"],
                "mcc": eval_result["mcc"],
                "brier_score": eval_result["brier_score"],
                "ece": eval_result["ece"],
            }
        )

    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def run_champion_stage(
    nested_summary: pd.DataFrame,
    nested_outputs: dict,
    splits: dict,
    dataset_hash: str,
    embedding_cache: dict | None = None,
):
    """
    Sélectionne le champion (modèle + variante de texte), l'entraîne sur l'ensemble
    d'entraînement complet, l'évalue sur l'ensemble de test mis de côté, et logue tout
    — y compris l'artefact du modèle — comme un seul run MLflow.

    Ajout de la Phase 11 : entraîne aussi le candidat dauphin (deuxième au classement)
    sur son propre ensemble de test mis de côté et exécute un test de significativité par
    bootstrap apparié contre le champion (mêmes lignes de test, puisque ``build_splits`` utilise
    un seul index de split stratifié partagé entre les variantes de texte) — "le
    champion est-il réellement meilleur, ou juste un split chanceux ?". Ceci est du reporting
    uniquement : cela ne change PAS quel modèle est enregistré/aliasé
    "staging" — c'est toujours exactement la configuration la mieux classée du
    benchmark nested CV, inchangée par cet ajout.
    """
    champion_config = select_champion_config(nested_summary)
    model_name = champion_config["model_name"]
    text_variant = champion_config["text_variant"]
    champion_params = select_champion_params(nested_outputs, text_variant, model_name)

    class_weights = compute_boosted_class_weights(splits[text_variant]["y_train"])
    champion_embedding_cache = embedding_cache if text_variant == "raw" else None
    registry = build_model_registry(class_weights, embedding_cache=champion_embedding_cache)

    logger.info("Champion selected: %s / %s, params=%s", model_name, text_variant, champion_params)
    # Phase 11 : le champion est le modèle qui est effectivement servi, donc il
    # est calibré (Platt scaling) ici — transforme le decision_function brut de
    # LinearSVC en vraies probabilités (predict_proba), un
    # prérequis pour que brier_score/ece aient un sens et pour que le score de
    # confiance de l'API cesse d'être une approximation softmax.
    final_model, eval_result, X_train, X_test, y_test = _train_and_evaluate(
        registry, model_name, champion_params, splits, text_variant, calibrate=True
    )

    runner_up_config = select_runner_up_config(nested_summary)
    significance_result = None
    if runner_up_config is not None:
        ru_model_name = runner_up_config["model_name"]
        ru_text_variant = runner_up_config["text_variant"]
        ru_params = select_champion_params(nested_outputs, ru_text_variant, ru_model_name)
        ru_class_weights = compute_boosted_class_weights(splits[ru_text_variant]["y_train"])
        ru_embedding_cache = embedding_cache if ru_text_variant == "raw" else None
        ru_registry = build_model_registry(ru_class_weights, embedding_cache=ru_embedding_cache)

        logger.info("Runner-up for significance test: %s / %s, params=%s", ru_model_name, ru_text_variant, ru_params)
        _, ru_eval_result, _, _, ru_y_test = _train_and_evaluate(
            ru_registry, ru_model_name, ru_params, splits, ru_text_variant
        )

        # Appariement valide : build_splits utilise UN SEUL split d'index stratifié
        # (train_test_split sur les indices de lignes) appliqué de manière identique à chaque
        # variante de texte, donc y_test correspond aux mêmes lignes/ordre quelle que soit la
        # variante — une simple vérification d'égalité documente cette hypothèse
        # plutôt que de lui faire confiance silencieusement.
        if list(y_test) == list(ru_y_test):
            significance_result = paired_bootstrap_test(y_test, eval_result["y_pred"], ru_eval_result["y_pred"])
            significance_result["runner_up_model"] = ru_model_name
            significance_result["runner_up_text_variant"] = ru_text_variant
            logger.info(
                "Significance test (champion vs runner-up %s/%s): diff=%.4f, p=%.4f, significant=%s",
                ru_model_name, ru_text_variant,
                significance_result["observed_diff"], significance_result["p_value"],
                significance_result["significant_at_0.05"],
            )
        else:
            logger.warning(
                "Champion and runner-up test sets did not align on the same rows — skipping significance test."
            )

    with mlflow.start_run(run_name="champion_final"):
        mlflow.log_param("stage", "champion_final")
        mlflow.log_param("champion_model", model_name)
        mlflow.log_param("text_variant", text_variant)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.log_params({f"hp__{k}": v for k, v in champion_params.items()})

        mlflow.log_metric("f1_macro", eval_result["f1_macro"])
        mlflow.log_metric("recall_macro", eval_result["recall_macro"])
        mlflow.log_metric("critical_recall", eval_result["critical_recall"])
        mlflow.log_metric("mcc", eval_result["mcc"])
        if eval_result["pr_auc_per_class"]:
            for label, value in eval_result["pr_auc_per_class"].items():
                mlflow.log_metric(f"pr_auc__{label}", value)
        if eval_result["brier_score"] is not None:
            mlflow.log_metric("brier_score", eval_result["brier_score"])
        if eval_result["ece"] is not None:
            mlflow.log_metric("ece", eval_result["ece"])

        if significance_result is not None:
            mlflow.log_param("significance_runner_up_model", significance_result["runner_up_model"])
            mlflow.log_param("significance_runner_up_text_variant", significance_result["runner_up_text_variant"])
            mlflow.log_metric("significance_observed_diff", significance_result["observed_diff"])
            mlflow.log_metric("significance_p_value", significance_result["p_value"])
            mlflow.log_metric("significance_significant", int(significance_result["significant_at_0.05"]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            eval_result["classification_report"].to_csv(tmp_dir / "classification_report.csv", index=False)
            eval_result["confusion_matrix"].to_csv(tmp_dir / "confusion_matrix.csv")
            mlflow.log_artifact(str(tmp_dir / "classification_report.csv"))
            mlflow.log_artifact(str(tmp_dir / "confusion_matrix.csv"))
            if significance_result is not None:
                (tmp_dir / "significance_test.json").write_text(json.dumps(significance_result, indent=2))
                mlflow.log_artifact(str(tmp_dir / "significance_test.json"))

        # Phase 11 : supprime le cache volumineux de temps d'entraînement de tout
        # EmbeddingVectorizer juste avant la sérialisation -- evaluate_final_model (ci-dessus)
        # l'a déjà utilisé pour model.predict(X_test), donc il n'est plus
        # nécessaire ; au moment de l'inférence réelle, l'API construit à la place un
        # petit cache par appel. Best-effort / duck-typed : un no-op pour tout
        # champion non basé sur des embeddings.
        _strip_embedding_caches(final_model)

        model_info = mlflow.sklearn.log_model(
            final_model,
            name="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            # Le sérialiseur sklearn par défaut de MLflow (skops) refuse de
            # (dé)sérialiser les types qu'il ne reconnaît pas comme sûrs, par conception.
            # Le champion est désormais enveloppé dans CalibratedClassifierCV
            # (calibration de la Phase 11), dont les types de calibrateur internes doivent
            # être explicitement déclarés de confiance -- ce sont les internes de calibration
            # propres à sklearn, pas du code arbitraire, donc c'est sans risque.
            skops_trusted_types=[
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration",
                # Phase 11 : notre propre EmbeddingVectorizer -- pas un
                # composant natif de sklearn, donc skops le traite comme non fiable par défaut. C'est
                # notre code, pas du code tiers arbitraire, donc lui faire confiance
                # ici est sans risque -- même logique que les internes de calibration
                # ci-dessus. Pertinent uniquement lorsque le champion est effectivement
                # Embedding_LogReg/Embedding_SVM ; inoffensif de toujours
                # le déclarer sinon.
                "mental_health.train.embedding_wrapper.EmbeddingVectorizer",
            ],
        )

        # Chaque modèle nouvellement entraîné est enregistré et aliasé "staging" —
        # disponible pour revue/comparaison, mais ne sert jamais de trafic tant que
        # promote.py ne le déplace pas explicitement vers "production".
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            MLFLOW_REGISTERED_MODEL_NAME, STAGING_ALIAS, model_info.registered_model_version
        )

        logger.info(
            "Champion final metrics — f1_macro=%.4f, recall_macro=%.4f, critical_recall=%.4f, mcc=%.4f",
            eval_result["f1_macro"], eval_result["recall_macro"], eval_result["critical_recall"], eval_result["mcc"],
        )
        logger.info(
            "Registered as '%s' version %s, aliased 'staging'",
            MLFLOW_REGISTERED_MODEL_NAME, model_info.registered_model_version,
        )

    return final_model, champion_config, eval_result, model_info.registered_model_version


def run(data_path: Path = DEFAULT_CLEAN_DATA_PATH) -> dict:
    """Exécute le pipeline complet de la Phase 3 : benchmark, sélection du champion, entraînement final, logging MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = _get_or_create_experiment(MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_id=experiment_id)

    df = pd.read_csv(data_path)
    dataset_hash = compute_dataset_hash(data_path)
    logger.info("Loaded clean dataset: %d rows, hash=%s", len(df), dataset_hash)

    splits = build_splits(df)

    # Phase 11 : les embeddings sentence-transformer (candidats
    # Embedding_LogReg/SVM) sont précalculés UNE SEULE FOIS pour tout le dataset ici -- un
    # extracteur de features gelé, pré-entraîné et non fit ne pose aucun risque de
    # fuite par fold, seulement un coût de calcul qui serait autrement payé
    # de manière redondante à chaque fold de CV et chaque évaluation de candidat.
    embedding_cache = precompute_dataset_embeddings(df[TEXT_COL])

    nested_summary, nested_outputs = run_benchmark_stage(splits, dataset_hash, embedding_cache=embedding_cache)
    final_model, champion_config, eval_result, registered_version = run_champion_stage(
        nested_summary, nested_outputs, splits, dataset_hash, embedding_cache=embedding_cache
    )

    logger.info(
        "Training complete. Champion: %s / %s | f1_macro=%.4f | critical_recall=%.4f | registry_version=%s",
        champion_config["model_name"], champion_config["text_variant"],
        eval_result["f1_macro"], eval_result["critical_recall"], registered_version,
    )

    # Phase 11 : tableau de comparaison complet sur TOUS les candidats benchmarkés,
    # écrit dans le repo (tableau docs/reports) et loggé comme artefact MLflow
    # de son propre run, afin qu'il survive indépendamment de tout run
    # champion_final unique et soit facile à diffier entre les runs d'entraînement.
    comparison_df = evaluate_all_candidates(nested_summary, nested_outputs, splits, embedding_cache=embedding_cache)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info("Model comparison table written to %s:\n%s", MODEL_COMPARISON_PATH, comparison_df.to_string(index=False))

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("stage", "model_comparison")
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("n_candidates", len(comparison_df))
        mlflow.log_artifact(str(MODEL_COMPARISON_PATH))

    return {
        "champion_config": champion_config,
        "eval_result": eval_result,
        "model": final_model,
        "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
        "registered_version": registered_version,
        "model_comparison": comparison_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
