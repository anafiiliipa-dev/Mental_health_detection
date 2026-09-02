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
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI

# Must run before the mlflow_config import below reads MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT from the environment (e.g. a shared team backend
# instead of the local SQLite default) — see mlflow_config.py's docstring.
# A no-op in Docker/Cloud Run, where these are set directly as container
# env vars and there is no .env file to find.
load_dotenv()

from mental_health.api.fallback import fallback_demo_prediction  # noqa: E402
from mental_health.api.logging_config import configure_logging  # noqa: E402
from mental_health.api.model_loader import LoadedModel, load_production_model  # noqa: E402
from mental_health.api.schemas import (  # noqa: E402
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

configure_logging()
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


def _predict_with_sklearn_model(model, text: str) -> PredictResponse:
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


def _predict_with_transformers_model(pipeline, text: str) -> PredictResponse:
    """
    ``pipeline`` is a HF ``text-classification`` pipeline (see
    ``register_distilbert.py``) with real label strings (not "LABEL_0")
    because the fine-tuned model's config carries the ``id2label`` mapping
    baked in at training time (``distilbert_finetune.py``).

    ``top_k=None`` is passed explicitly on every call, NOT relied upon as
    the pipeline's baked-in default: a pipeline built with ``top_k=None``
    and then round-tripped through ``mlflow.transformers.log_model`` /
    ``load_model`` does not necessarily keep that setting (observed in
    practice — the reloaded pipeline silently fell back to ``top_k=1``,
    returning one ``{"label", "score"}`` dict per input instead of a list
    covering every class, which crashed the dict comprehension below with
    a confusing "string indices must be integers" — iterating a dict
    yields its string keys). Passing ``top_k=None`` here every time is
    what actually determines the output shape, regardless of what MLflow
    preserved from registration.

    ``truncation=True`` is passed for the same reason: the pipeline was
    never built with a truncation setting (see ``register_distilbert.py``),
    so a request text longer than the model's 512-token limit crashes with
    a raw ``RuntimeError`` from inside PyTorch's position-embedding
    addition instead of failing gracefully — first observed when the
    monitoring job (``drift_check.py``) scored real, longer training
    texts. Truncating to the model's max length is the standard way to
    handle this in a triage tool, where cutting off the tail of an
    overlong submission is an acceptable trade-off against a hard 500.
    """
    scores = pipeline([text], top_k=None, truncation=True)[0]
    probabilities = {item["label"]: float(item["score"]) for item in scores}
    best = max(scores, key=lambda item: item["score"])

    return PredictResponse(
        label=str(best["label"]), confidence=float(best["score"]), probabilities=probabilities, is_demo_fallback=False
    )


def _predict_with_real_model(loaded: LoadedModel, text: str) -> PredictResponse:
    """Dispatch on the loaded model's flavor (see model_loader.py) — the
    two shapes of "real model" this API can currently serve."""
    if loaded.flavor == "transformers":
        return _predict_with_transformers_model(loaded.model, text)
    return _predict_with_sklearn_model(loaded.model, text)


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
    started = time.perf_counter()

    if not loaded.is_available:
        label, confidence, probabilities = fallback_demo_prediction(request.text)
        response = PredictResponse(label=label, confidence=confidence, probabilities=probabilities, is_demo_fallback=True)
    else:
        response = _predict_with_real_model(loaded, request.text)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    # Never log request.text — a hash + length is enough to debug volume/traffic
    # without ever persisting raw mental-health text in application logs.
    #
    # probabilities IS included (unlike the text): it never reveals
    # anything about the submitted text beyond what predicted_label
    # already does (the predicted class), and having the full per-class
    # distribution in the log -- not just the top label -- is useful for
    # spotting low-confidence predictions and, later, feeding the
    # Evidently monitoring/drift checks (mental_health.monitoring) without
    # needing to re-run inference.
    logger.info(
        "predict request",
        extra={
            "fingerprint": _text_fingerprint(request.text),
            "text_length": len(request.text),
            "is_demo_fallback": response.is_demo_fallback,
            "model_version": loaded.version,
            "predicted_label": response.label,
            "probabilities": response.probabilities,
            "latency_ms": latency_ms,
        },
    )

    return response
