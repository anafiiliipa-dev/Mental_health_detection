"""
FastAPI service (Phase 6): serves the "production"-aliased MLflow model
over HTTP.

Endpoints
---------
GET  /health       liveness only — never touches the model or MLflow.
GET  /model-info    which model version is serving, its metrics, or why
                    none is available.
POST /predict       classify a piece of text.

Degraded mode (confirmed with the project owner before writing this):
if no model is aliased "production" in the Registry, the API still
starts and /predict answers with a clearly-labelled heuristic fallback
(`fallback.py`, `is_demo_fallback=True`) instead of failing hard — this
mirrors the graceful-degradation pattern already used elsewhere in the
project (`mental_health.models.services`).

Privacy (per the original audit): the raw request text is never logged.
Only a one-way hash + length are logged per /predict call, and the
response never echoes the submitted text back (enforced structurally —
see `schemas.PredictResponse`, which has no text field at all).
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI

from mental_health.api.fallback import fallback_demo_prediction
from mental_health.api.model_loader import LoadedModel, load_production_model
from mental_health.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from mental_health.config.mlflow_config import MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

# Loaded once at startup (see `lifespan`), not per-request — re-loading the
# model on every call would be needless latency and MLflow Registry load.
_STATE: dict[str, LoadedModel] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> Iterator[None]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    loaded = load_production_model()
    if loaded.is_available:
        logger.info("Startup: serving '%s' v%s", MLFLOW_REGISTERED_MODEL_NAME, loaded.version)
    else:
        logger.warning(
            "Startup: no production model available (%s) — /predict will use the demo fallback.", loaded.error
        )
    _STATE["model"] = loaded
    yield
    _STATE.clear()


app = FastAPI(
    title="Mental Health Intelligence API",
    description="Serves the MLflow-registered mental-health text triage classifier.",
    version="0.1.0",
    lifespan=lifespan,
)


def _text_fingerprint(text: str) -> str:
    """Non-reversible fingerprint for logs — never the raw text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _predict_with_real_model(loaded: LoadedModel, text: str) -> PredictResponse:
    model = loaded.model
    prediction = model.predict([text])[0]

    probabilities: dict[str, float] | None = None
    confidence = 1.0

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        probabilities = dict(zip((str(c) for c in model.classes_), (float(p) for p in proba), strict=True))
        confidence = float(max(proba))
    elif hasattr(model, "decision_function"):
        # The current champion (LinearSVC) has no predict_proba. Approximate a
        # confidence score with a softmax over its decision_function — informative,
        # not a calibrated probability.
        scores = model.decision_function([text])[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        probabilities = dict(zip((str(c) for c in model.classes_), (float(p) for p in proba), strict=True))
        confidence = float(max(proba))

    return PredictResponse(
        label=str(prediction), confidence=confidence, probabilities=probabilities, is_demo_fallback=False
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    loaded: LoadedModel = _STATE["model"]
    return ModelInfoResponse(
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        model_available=loaded.is_available,
        version=loaded.version,
        run_id=loaded.run_id,
        metrics=loaded.metrics,
        error=loaded.error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    loaded: LoadedModel = _STATE["model"]

    # Never log request.text — a hash + length is enough to debug volume/traffic
    # without ever persisting raw mental-health text in application logs.
    logger.info(
        "predict request: fingerprint=%s length=%d demo_fallback=%s",
        _text_fingerprint(request.text),
        len(request.text),
        not loaded.is_available,
    )

    if not loaded.is_available:
        label, confidence, probabilities = fallback_demo_prediction(request.text)
        return PredictResponse(label=label, confidence=confidence, probabilities=probabilities, is_demo_fallback=True)

    return _predict_with_real_model(loaded, request.text)
