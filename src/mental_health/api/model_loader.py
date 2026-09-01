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

Flavor dispatch (added after DistilBERT joined the candidate pool via
``register_distilbert.py``): ``promote.py``'s promotion gate only compares
``f1_macro``/``critical_recall`` — it has no notion of "model type", so
whatever is aliased "production" can be either a classical/embedding
scikit-learn champion (``train.py``, ``mlflow.sklearn.log_model``) or the
fine-tuned DistilBERT (``mlflow.transformers.log_model``). This loader
detects which one it actually got and loads it with the matching MLflow
flavor loader, so a promotion never silently breaks the API (as it did
before this change: ``mlflow.sklearn.load_model`` raising
"Model does not have the sklearn flavor" against a promoted DistilBERT
version, degrading straight to the demo fallback).

``mlflow.transformers`` is imported lazily, inside ``load_production_model``,
and only on the branch that actually needs it — the Docker image only
installs the ``api``/``mlflow`` extras (see ``Dockerfile``), NOT
``transformers`` (torch alone is ~2GB and would balloon every deploy just
to support a candidate that is normally in "staging", not "production").
A module-level ``import mlflow.transformers`` would make importing this
module — and therefore starting the API at all — fail outright wherever
that extra isn't installed, even when serving a plain scikit-learn
champion. Lazy import keeps the common case (sklearn in production)
working unchanged in the slim image; serving a promoted DistilBERT
version for real additionally requires building the image with the
``transformers`` extra (see ``Dockerfile``'s comment).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

from mental_health.config.mlflow_config import (
    MLFLOW_REGISTERED_MODEL_NAME,
    PRODUCTION_ALIAS,
)

logger = logging.getLogger(__name__)

# The two model flavors this project's registry can currently contain --
# every classical/embedding champion (mlflow.sklearn.log_model) and the
# DistilBERT fine-tune (mlflow.transformers.log_model, see
# register_distilbert.py). Checked in this order so a model logged with
# both flavors present (shouldn't happen here, but MLmodel files can in
# principle list more than one) prefers the lighter, already-installed
# sklearn loader.
SUPPORTED_FLAVORS = ("sklearn", "transformers")


@dataclass
class LoadedModel:
    """Result of attempting to load the production model."""

    model: Any | None
    flavor: str | None
    version: str | None
    run_id: str | None
    metrics: dict
    error: str | None

    @property
    def is_available(self) -> bool:
        return self.model is not None


def _detect_flavor(model_uri: str) -> str:
    """
    Which of ``SUPPORTED_FLAVORS`` this model version was logged with.

    Raises ``ValueError`` if it's neither — ``load_production_model``
    catches this and turns it into a ``LoadedModel.error`` instead of
    letting the API crash on an unexpected/future flavor.
    """
    info = mlflow.models.get_model_info(model_uri)
    for flavor in SUPPORTED_FLAVORS:
        if flavor in info.flavors:
            return flavor
    raise ValueError(f"Unsupported model flavor(s) {list(info.flavors)} -- expected one of {SUPPORTED_FLAVORS}")


def load_production_model(model_name: str = MLFLOW_REGISTERED_MODEL_NAME) -> LoadedModel:
    """
    Load the model currently aliased "production" in the MLflow Registry.

    Never raises: any failure (no "production" alias set, MLflow store
    unreachable, corrupt artifact, unsupported/undetectable flavor, ...) is
    captured in the returned ``LoadedModel.error`` instead, so the API can
    start in degraded mode.

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
        return LoadedModel(model=None, flavor=None, version=None, run_id=None, metrics={}, error=str(exc))

    model_uri = f"models:/{model_name}@{PRODUCTION_ALIAS}"
    flavor: str | None = None
    try:
        flavor = _detect_flavor(model_uri)
        if flavor == "sklearn":
            model = mlflow.sklearn.load_model(model_uri)
        else:
            # Deliberately lazy, and imported under an alias rather than
            # `import mlflow.transformers` -- a plain submodule import here
            # would make Python treat the outer `mlflow` name as local to
            # this whole function (shadowing the module-level `import
            # mlflow` used a few lines above for `mlflow.MlflowClient()`),
            # which is a real UnboundLocalError risk, not just a style
            # nit. See module docstring for why this import is lazy at all.
            from mlflow import transformers as mlflow_transformers

            model = mlflow_transformers.load_model(model_uri)
        run = client.get_run(model_version.run_id)
    except (MlflowException, OSError, ValueError, ImportError) as exc:
        logger.error(
            "Found '%s' v%s (flavor=%s) but failed to load it: %s", model_name, model_version.version, flavor, exc
        )
        return LoadedModel(
            model=None, flavor=None, version=str(model_version.version), run_id=model_version.run_id,
            metrics={}, error=str(exc),
        )

    logger.info(
        "Loaded '%s' v%s (run %s, flavor=%s) as the production model",
        model_name, model_version.version, model_version.run_id, flavor,
    )
    return LoadedModel(
        model=model,
        flavor=flavor,
        version=str(model_version.version),
        run_id=model_version.run_id,
        metrics=dict(run.data.metrics),
        error=None,
    )
