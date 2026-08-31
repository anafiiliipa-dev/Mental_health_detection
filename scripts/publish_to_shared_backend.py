"""
One-off script: republish the current LOCAL "production" model into the
shared MLflow backend (Neon Postgres + S3-compatible storage) used for the
team's Cloud Run deployment.

Why this exists: the shared backend starts empty — there is no built-in
MLflow way to "copy" a registered model (plus its metrics) between two
different backends. This script does the simplest thing that works: load
the model already serving locally, then log + register + alias it again
against the shared backend, exactly like train.py's ``champion_final`` run
does for the local one.

Usage (see the commented MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_ROOT lines
in .env — "MLflow — backend partagé"): uncomment those two lines, fill in
the new Neon password, then run:

    python scripts/publish_to_shared_backend.py

Safe to re-run: each run creates one new version on the shared backend and
re-points the "production" alias to it — never touches the local store.
"""
from __future__ import annotations

import logging

import mlflow
from dotenv import load_dotenv

# Must run before the mlflow_config import below reads MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT from the environment — nothing else in this project's
# MLflow/train/promote code path loads .env (only the Streamlit app does),
# so this script has to do it itself.
load_dotenv()

from mental_health.api.model_loader import load_production_model  # noqa: E402
from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
)
from mental_health.config.paths import PROJECT_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_or_create_experiment(name: str, artifact_location: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)


def main() -> None:
    if MLFLOW_TRACKING_URI.startswith("sqlite:///"):
        raise SystemExit(
            "MLFLOW_TRACKING_URI is still pointing at the local SQLite store. "
            "Uncomment the shared-backend lines in .env before running this script — "
            "publishing to the local store would be a no-op."
        )

    # Read the source model from the LOCAL store explicitly (not whatever
    # the environment happens to point at), so "what are we republishing"
    # is never ambiguous even once MLFLOW_TRACKING_URI is set to the shared
    # backend for the rest of this script.
    local_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    logger.info("Loading the current LOCAL production model from %s ...", local_uri)
    mlflow.set_tracking_uri(local_uri)
    local_model = load_production_model()
    if not local_model.is_available:
        raise SystemExit(f"No local production model to republish: {local_model.error}")

    logger.info(
        "Loaded local production model v%s (run %s), metrics=%s",
        local_model.version,
        local_model.run_id,
        local_model.metrics,
    )

    logger.info("Switching to the shared backend: %s", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = _get_or_create_experiment(MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name="republish_from_local"):
        mlflow.log_param("stage", "republish_from_local")
        mlflow.log_param("source_local_run_id", local_model.run_id)
        mlflow.log_param("source_local_version", local_model.version)
        mlflow.log_metrics(local_model.metrics)

        model_info = mlflow.sklearn.log_model(
            local_model.model, name="model", registered_model_name=MLFLOW_REGISTERED_MODEL_NAME
        )

    client = mlflow.MlflowClient()
    new_version = model_info.registered_model_version
    client.set_registered_model_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, new_version)
    logger.info(
        "'%s' v%s is now aliased '%s' on the SHARED backend.",
        MLFLOW_REGISTERED_MODEL_NAME,
        new_version,
        PRODUCTION_ALIAS,
    )


if __name__ == "__main__":
    main()
