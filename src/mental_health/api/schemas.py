"""
Modèles Pydantic de requête/réponse pour le service FastAPI.

Volontairement séparés de ``main.py`` afin que le contrat (ce qu'un
client envoie/reçoit) soit lisible et testable de façon autonome, et afin que
``model_loader.py`` n'ait jamais besoin d'importer FastAPI.

Note de confidentialité (selon l'audit initial) : ``PredictResponse`` ne renvoie
jamais le texte soumis à l'appelant. L'API n'est pas un endroit où persister
ou refléter du texte brut de santé mentale inutilement — la réponse ne porte
que la prédiction.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mental_health.config.paths import CLASS_LABELS

# Plafond strict sur la longueur du texte de la requête — pas un choix de
# modélisation, une simple protection basique contre l'abus/DoS pour un
# endpoint HTTP quasi public. Généreux pour un post de type forum.
MAX_TEXT_LENGTH = 10_000


class PredictRequest(BaseModel):
    """Corps de POST /predict."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Free-text post/message to classify. Never logged in clear text.",
    )


class PredictResponse(BaseModel):
    """Corps retourné par POST /predict. N'inclut jamais le texte soumis."""

    label: str = Field(description="Predicted class label.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the predicted label.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="Per-class probabilities, when the underlying model exposes them."
    )
    is_demo_fallback: bool = Field(
        description="True if no real model was available and a heuristic demo prediction was used instead."
    )


class ModelInfoResponse(BaseModel):
    """Corps retourné par GET /model-info."""

    registered_model_name: str
    model_available: bool = Field(description="False when running in demo-fallback mode.")
    version: str | None = None
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = Field(
        default=None, description="Why the production model could not be loaded, if model_available is False."
    )


class HealthResponse(BaseModel):
    """Corps retourné par GET /health. Ne touche volontairement pas au modèle."""

    status: str = "ok"


# Exposé pour être réutilisé par main.py / les tests sans re-dériver la liste de labels.
VALID_LABELS: list[str] = list(CLASS_LABELS)
