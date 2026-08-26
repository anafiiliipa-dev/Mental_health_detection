"""
Pydantic request/response models for the FastAPI service.

Kept deliberately separate from ``main.py`` so the contract (what a
client sends/receives) is readable and testable on its own, and so
``model_loader.py`` never needs to import FastAPI at all.

Privacy note (per the original audit): ``PredictResponse`` never echoes
the submitted text back to the caller. The API is not a place to persist
or reflect raw mental-health text unnecessarily — the response carries
only the prediction.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mental_health.config.paths import CLASS_LABELS

# Hard cap on request text length — not a modelling choice, a basic abuse/DoS
# guard for a public-ish HTTP endpoint. Generous for a forum-style post.
MAX_TEXT_LENGTH = 10_000


class PredictRequest(BaseModel):
    """Body of POST /predict."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Free-text post/message to classify. Never logged in clear text.",
    )


class PredictResponse(BaseModel):
    """Body returned by POST /predict. Never includes the submitted text."""

    label: str = Field(description="Predicted class label.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the predicted label.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="Per-class probabilities, when the underlying model exposes them."
    )
    is_demo_fallback: bool = Field(
        description="True if no real model was available and a heuristic demo prediction was used instead."
    )


class ModelInfoResponse(BaseModel):
    """Body returned by GET /model-info."""

    registered_model_name: str
    model_available: bool = Field(description="False when running in demo-fallback mode.")
    version: str | None = None
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = Field(
        default=None, description="Why the production model could not be loaded, if model_available is False."
    )


class HealthResponse(BaseModel):
    """Body returned by GET /health. Deliberately does not touch the model."""

    status: str = "ok"


# Exposed for reuse by main.py / tests without re-deriving the label list.
VALID_LABELS: list[str] = list(CLASS_LABELS)
