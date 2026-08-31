"""
End-to-end training entry point: benchmark all candidates, select the
champion, train it, evaluate it, and log everything to MLflow.

Extracted and corrected from ``notebooks/02_classical_ml.ipynb`` (the
overall orchestration: PHASE 1 train/test split through PHASE 20 summary),
wiring together ``model_registry.py``, ``benchmark.py`` and
``champion.py``. This is the Phase 3 deliverable of the roadmap:
"instrumenter le script de training avec MLflow".

Design choices relative to the original notebook, made explicit rather
than silently ported:

1. **Only the nested CV benchmark is run here, not the light CV.** In the
   notebook, light CV is a fast pre-screening step whose results are
   never actually used by the champion selection (``select_champion_config``
   reads only ``nested_cv_summary``). Running it in production would
   roughly double training time for zero effect on the outcome, so this
   script runs nested CV only — the one benchmark that actually decides.
   ``run_light_cv_benchmark`` is still available in ``benchmark.py`` for
   ad-hoc exploration in a notebook if useful.
2. **MLflow tracking store**: local SQLite backend
   (``<project_root>/mlflow.db``) with artifacts stored under
   ``<project_root>/mlruns`` (agreed with Ana — no Postgres/S3 dependency
   until Docker arrives in Phase 7). SQLite rather than the plain
   filesystem store because MLflow 3.x puts the raw filestore backend in
   maintenance mode and recommends a database backend even for local use;
   SQLite also means the eventual migration to Postgres (Phase 7) is a
   connection-string change, not a data migration. Migrating later just
   means pointing ``mlflow.set_tracking_uri`` at the new backend; nothing
   else changes.
3. **One MLflow run per text variant** during the nested CV benchmark
   (``nested_cv_raw``, ``nested_cv_masked``), plus one final
   ``champion_final`` run that logs the trained model itself via
   ``mlflow.sklearn.log_model``. This mirrors how the notebook conceptually
   separates "compare candidates" from "train the winner".
4. **Dataset hash logged as an MLflow param** (SHA256, first 16 chars) —
   the audit's Étape F recommendation, giving basic dataset-version
   traceability without introducing DVC.
5. **Model Registry (Phase 5)**: the champion is registered under
   ``MLFLOW_REGISTERED_MODEL_NAME`` and aliased ``"staging"`` on every run —
   never ``"production"`` directly. Promotion is a separate, explicit
   decision made by ``promote.py`` against documented thresholds, not an
   automatic or manual UI action (per the audit's governance
   recommendation: "qui a le droit de promouvoir un modèle ?"). Stages
   (MLflow's older mechanism) are deprecated as of MLflow 2.9 in favour of
   aliases, so this uses aliases directly rather than building on an
   API already scheduled for removal.
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

# Must run before the mlflow_config import below reads MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT from the environment (e.g. a shared team backend
# instead of the local SQLite default) — see mlflow_config.py's docstring.
load_dotenv()

from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    STAGING_ALIAS,
)
from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH  # noqa: E402
from mental_health.data.cleaning import MASKED_COL, TARGET_COL, TEXT_COL  # noqa: E402
from mental_health.train.benchmark import run_nested_cv_benchmark  # noqa: E402
from mental_health.train.champion import (  # noqa: E402
    evaluate_final_model,
    select_champion_config,
    select_champion_params,
    select_runner_up_config,
    train_final_model,
)
from mental_health.train.evaluation_metrics import paired_bootstrap_test  # noqa: E402
from mental_health.train.model_registry import (  # noqa: E402
    RANDOM_STATE,
    build_model_registry,
    compute_boosted_class_weights,
)

logger = logging.getLogger(__name__)

# MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT, MLFLOW_EXPERIMENT_NAME and
# MLFLOW_REGISTERED_MODEL_NAME live in mental_health.config.mlflow_config —
# the single source of truth shared with promote.py and the FastAPI service
# (Phase 6), so serving code doesn't need to import this training module.
TEST_SIZE = 0.2
TEXT_VARIANTS = ["raw", "masked"]
VARIANT_COLUMNS = {"raw": TEXT_COL, "masked": MASKED_COL}


def _get_or_create_experiment(name: str, artifact_location: str) -> str:
    """Get the experiment by name, creating it (with an explicit artifact location) if missing."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)


def compute_dataset_hash(path: Path) -> str:
    """First 16 hex chars of the dataset file's SHA256 — a lightweight, DVC-free version marker."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_splits(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> dict:
    """
    Build ONE stratified train/test split on row indices, then apply it to
    both text variants (raw, masked) — guaranteeing the two variants are
    compared on the exact same rows.
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


def run_benchmark_stage(splits: dict, dataset_hash: str) -> tuple[pd.DataFrame, dict]:
    """
    Run the nested CV benchmark for every text variant, logging one MLflow
    run per variant. Returns the combined summary (all variants, all
    models) and the raw best-params output needed for champion selection.
    """
    summaries = []
    nested_outputs = {}

    for text_variant in TEXT_VARIANTS:
        X_train = splits[text_variant]["X_train"]
        y_train = splits[text_variant]["y_train"]

        class_weights = compute_boosted_class_weights(y_train)
        registry = build_model_registry(class_weights)

        with mlflow.start_run(run_name=f"nested_cv_{text_variant}"):
            mlflow.log_param("stage", "nested_cv_benchmark")
            mlflow.log_param("text_variant", text_variant)
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("candidate_models", list(registry.keys()))

            logger.info("Running nested CV benchmark — variant=%s", text_variant)
            _, summary, best_params = run_nested_cv_benchmark(X_train, y_train, registry)
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
    Train one candidate (model_name/text_variant) on its full training
    split and evaluate it on its own test split.

    ``calibrate=True`` fits it through ``CalibratedClassifierCV`` (Platt
    scaling) — used only for the champion (see ``run_champion_stage``),
    since that is the model that actually ends up served; the runner-up
    trained here only for the significance test stays uncalibrated.
    """
    X_train = splits[text_variant]["X_train"]
    y_train = splits[text_variant]["y_train"]
    X_test = splits[text_variant]["X_test"]
    y_test = splits[text_variant]["y_test"]

    model = train_final_model(registry, model_name, params, X_train, y_train, calibrate=calibrate)
    eval_result = evaluate_final_model(model, X_test, y_test)
    return model, eval_result, X_train, X_test, y_test


def run_champion_stage(nested_summary: pd.DataFrame, nested_outputs: dict, splits: dict, dataset_hash: str):
    """
    Select the champion (model + text variant), train it on the full
    training set, evaluate it on the held-out test set, and log everything
    — including the model artifact — as one MLflow run.

    Phase 11 addition: also trains the runner-up (second-ranked) candidate
    on its own held-out test set and runs a paired bootstrap significance
    test against the champion (same test rows, since ``build_splits`` uses
    one shared stratified split index across text variants) — "is the
    champion actually better, or just a lucky split?". This is reporting
    only: it does NOT change which model gets registered/aliased
    "staging" — that is still exactly the top-ranked config from the
    nested CV benchmark, unchanged from before this addition.
    """
    champion_config = select_champion_config(nested_summary)
    model_name = champion_config["model_name"]
    text_variant = champion_config["text_variant"]
    champion_params = select_champion_params(nested_outputs, text_variant, model_name)

    class_weights = compute_boosted_class_weights(splits[text_variant]["y_train"])
    registry = build_model_registry(class_weights)

    logger.info("Champion selected: %s / %s, params=%s", model_name, text_variant, champion_params)
    # Phase 11: the champion is the model that actually gets served, so it
    # is calibrated (Platt scaling) here — turns LinearSVC's raw
    # decision_function into real probabilities (predict_proba), a
    # prerequisite for brier_score/ece to mean anything and for the API's
    # confidence score to stop being a softmax approximation.
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
        ru_registry = build_model_registry(ru_class_weights)

        logger.info("Runner-up for significance test: %s / %s, params=%s", ru_model_name, ru_text_variant, ru_params)
        _, ru_eval_result, _, _, ru_y_test = _train_and_evaluate(
            ru_registry, ru_model_name, ru_params, splits, ru_text_variant
        )

        # Valid pairing: build_splits uses ONE stratified index split
        # (train_test_split on row indices) applied identically to every
        # text variant, so y_test is the same rows/order regardless of
        # variant — a plain equality check documents that assumption
        # rather than silently trusting it.
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

        model_info = mlflow.sklearn.log_model(
            final_model,
            name="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            # MLflow's default sklearn serializer (skops) refuses to
            # (de)serialize types it doesn't recognise as safe, by design.
            # The champion is now wrapped in CalibratedClassifierCV
            # (Phase 11 calibration), whose internal calibrator types need
            # to be explicitly declared trusted -- these are sklearn's own
            # calibration internals, not arbitrary code, so this is safe.
            skops_trusted_types=[
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration",
            ],
        )

        # Every newly trained model is registered and aliased "staging" —
        # available for review/comparison, but never serving traffic until
        # promote.py explicitly moves it to "production".
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
    """Run the full Phase 3 pipeline: benchmark, champion selection, final training, MLflow logging."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = _get_or_create_experiment(MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_id=experiment_id)

    df = pd.read_csv(data_path)
    dataset_hash = compute_dataset_hash(data_path)
    logger.info("Loaded clean dataset: %d rows, hash=%s", len(df), dataset_hash)

    splits = build_splits(df)

    nested_summary, nested_outputs = run_benchmark_stage(splits, dataset_hash)
    final_model, champion_config, eval_result, registered_version = run_champion_stage(
        nested_summary, nested_outputs, splits, dataset_hash
    )

    logger.info(
        "Training complete. Champion: %s / %s | f1_macro=%.4f | critical_recall=%.4f | registry_version=%s",
        champion_config["model_name"], champion_config["text_variant"],
        eval_result["f1_macro"], eval_result["critical_recall"], registered_version,
    )

    return {
        "champion_config": champion_config,
        "eval_result": eval_result,
        "model": final_model,
        "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
        "registered_version": registered_version,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
