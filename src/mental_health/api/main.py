"""
Service FastAPI (Phase 6) : sert le modèle MLflow avec l'alias "production"
via HTTP.

Endpoints
---------
GET  /health       liveness uniquement — ne touche jamais au modèle ni à MLflow.
GET  /model-info    quelle version de modèle est servie, ses métriques, ou pourquoi
                    aucune n'est disponible.
POST /predict       classifie un texte.

Mode dégradé (confirmé avec le propriétaire du projet avant d'écrire ce code) :
si aucun modèle n'a l'alias "production" dans le Registry, l'API démarre
quand même et /predict répond avec un fallback heuristique clairement
étiqueté (`fallback.py`, `is_demo_fallback=True`) plutôt que d'échouer
brutalement — cela reprend le pattern de dégradation gracieuse déjà utilisé
ailleurs dans le projet (`mental_health.models.services`).

Confidentialité (selon l'audit initial) : le texte brut de la requête n'est jamais
loggé. Seuls un hash à sens unique et sa longueur sont loggés par appel /predict, et
la réponse ne renvoie jamais le texte soumis (imposé structurellement —
voir `schemas.PredictResponse`, qui n'a aucun champ texte).
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

# Doit s'exécuter avant que l'import mlflow_config ci-dessous ne lise MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT depuis l'environnement (par ex. un backend d'équipe partagé
# au lieu du SQLite local par défaut) — voir le docstring de mlflow_config.py.
# Un no-op sous Docker/Cloud Run, où ces variables sont définies directement comme
# variables d'environnement du conteneur et où il n'y a pas de fichier .env à trouver.
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

# Chargé une seule fois au démarrage (voir `lifespan`), pas à chaque requête — recharger
# le modèle à chaque appel serait une latence inutile et une charge inutile sur le MLflow Registry.
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
    """Empreinte non réversible pour les logs — jamais le texte brut lui-même."""
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
        # Le champion actuel (LinearSVC) n'a pas de predict_proba. On approxime un
        # score de confiance avec un softmax sur son decision_function — informatif,
        # mais pas une probabilité calibrée.
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
    ``pipeline`` est un pipeline HF ``text-classification`` (voir
    ``register_distilbert.py``) avec de vraies chaînes de labels (pas "LABEL_0")
    car la config du modèle fine-tuné embarque le mapping ``id2label``
    figé au moment de l'entraînement (``distilbert_finetune.py``).

    ``top_k=None`` est passé explicitement à chaque appel, SANS compter sur
    le réglage par défaut figé dans le pipeline : un pipeline construit avec
    ``top_k=None`` puis passé par un aller-retour via ``mlflow.transformers.log_model`` /
    ``load_model`` ne conserve pas nécessairement ce réglage (observé en
    pratique — le pipeline rechargé retombait silencieusement sur ``top_k=1``,
    renvoyant un seul dict ``{"label", "score"}`` par entrée au lieu d'une liste
    couvrant chaque classe, ce qui faisait planter la compréhension de dict
    ci-dessous avec un message confus "string indices must be integers" —
    itérer sur un dict renvoie ses clés sous forme de chaînes). Passer
    ``top_k=None`` ici à chaque fois est ce qui détermine réellement la forme
    de la sortie, indépendamment de ce que MLflow a conservé lors de l'enregistrement.

    ``truncation=True`` est passé pour la même raison : le pipeline n'a jamais
    été construit avec un réglage de troncature (voir ``register_distilbert.py``),
    donc un texte de requête plus long que la limite de 512 tokens du modèle
    plante avec une ``RuntimeError`` brute venant de l'addition des
    position-embeddings dans PyTorch, au lieu d'échouer proprement — observé
    pour la première fois lorsque le job de monitoring (``drift_check.py``) a
    scoré des textes d'entraînement réels, plus longs. Tronquer à la longueur
    maximale du modèle est la façon standard de gérer ceci dans un outil de
    triage, où couper la fin d'une soumission trop longue est un compromis
    acceptable face à une erreur 500 brutale.
    """
    scores = pipeline([text], top_k=None, truncation=True)[0]
    probabilities = {item["label"]: float(item["score"]) for item in scores}
    best = max(scores, key=lambda item: item["score"])

    return PredictResponse(
        label=str(best["label"]), confidence=float(best["score"]), probabilities=probabilities, is_demo_fallback=False
    )


def _predict_with_real_model(loaded: LoadedModel, text: str) -> PredictResponse:
    """Dispatche selon le flavor du modèle chargé (voir model_loader.py) — les
    deux formes de "vrai modèle" que cette API peut actuellement servir."""
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

    # Ne jamais logger request.text — un hash + une longueur suffisent pour déboguer
    # le volume/trafic sans jamais persister de texte brut de santé mentale dans les logs
    # applicatifs.
    #
    # probabilities EST inclus (contrairement au texte) : cela ne révèle
    # jamais rien sur le texte soumis au-delà de ce que predicted_label
    # révèle déjà (la classe prédite), et avoir la distribution complète
    # par classe dans le log — pas seulement le label le plus probable — est
    # utile pour repérer les prédictions à faible confiance et, plus tard,
    # alimenter les vérifications de monitoring/drift Evidently
    # (mental_health.monitoring) sans avoir besoin de relancer l'inférence.
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
