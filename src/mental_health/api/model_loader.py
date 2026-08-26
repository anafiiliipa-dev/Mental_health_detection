"""
Loads the "production"-aliased model from the MLflow Model Registry for
the FastAPI service.

Design decision (confirmed with the project owner before writing this
file): if no model is aliased "production" — Registry empty, MLflow
unreachable, etc. — the API must still start. ``load_production_model()``
never raises; it returns a ``LoadedModel`` whose ``model`` is ``None`` and
whose ``error`` explains why. Callers (``main.py``) fall back to a clearly
labelled demo heuristic, mirroring the existing graceful-degradation
pattern already used in ``mental_health.models.services`` (``load_model``
returning a ``(model, path, error)`` tuple + ``fallback_demo_prediction``).
An unavailable ML model must never take the whole API down — this is a
triage tool, not the sole gate to care.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException

from mental_health.config.mlflow_config import (
    MLFLOW_REGISTERED_MODEL_NAME,
    PRODUCTION_ALIAS,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Result of attempting to load the production model."""

    model: Any | None
    version: str | None
    run_id: str | None
    metrics: dict
    error: str | None

    @property
    def is_available(self) -> bool:
        return self.model is not None


def load_production_model(model_name: str = MLFLOW_REGISTERED_MODEL_NAME) -> LoadedModel:
    """
    Load the model currently aliased "production" in the MLflow Registry.

    Never raises: any failure (no "production" alias set, MLflow store
    unreachable, corrupt artifact, ...) is captured in the returned
    ``LoadedModel.error`` instead, so the API can start in degraded mode.

    Assumes ``mlflow.set_tracking_uri`` has already been called by the
    entrypoint (``main.py`` at API startup, or a test fixture) — this
    function does not set it itself, so it stays testable against a
    throwaway store without touching global config, the same convention
    already used by ``train.py`` / ``promote.py``.
    """
    client = mlflow.MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException as exc:
        logger.warning("No '%s' model aliased '%s': %s", model_name, PRODUCTION_ALIAS, exc)
        return LoadedModel(model=None, version=None, run_id=None, metrics={}, error=str(exc))

    try:
        model_uri = f"models:/{model_name}@{PRODUCTION_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        run = client.get_run(model_version.run_id)
    except (MlflowException, OSError) as exc:
        logger.error("Found '%s' v%s but failed to load it: %s", model_name, model_version.version, exc)
        return LoadedModel(model=None, version=model_version.version, run_id=model_version.run_id, metrics={}, error=str(exc))

    logger.info("Loaded '%s' v%s (run %s) as the production model", model_name, model_version.version, model_version.run_id)
    return LoadedModel(
        model=model,
        version=model_version.version,
        run_id=model_version.run_id,
        metrics=dict(run.data.metrics),
        error=None,
    )
