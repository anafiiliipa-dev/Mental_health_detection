"""
Charge le modèle avec l'alias "production" depuis le MLflow Model Registry pour
le service FastAPI.

Décision de conception (confirmée avec le propriétaire du projet avant d'écrire ce
fichier) : si aucun modèle n'a l'alias "production" — Registry vide, MLflow
injoignable, etc. — l'API doit quand même démarrer. ``load_production_model()``
ne lève jamais d'exception ; elle retourne un ``LoadedModel`` dont ``model`` vaut
``None`` et dont ``error`` explique pourquoi. Les appelants (``main.py``) se
rabattent sur une heuristique de démo clairement étiquetée, reprenant le pattern
de dégradation gracieuse déjà utilisé dans ``mental_health.models.services``
(``load_model`` renvoyant un tuple ``(model, path, error)`` + ``fallback_demo_prediction``).
Un modèle ML indisponible ne doit jamais faire tomber toute l'API — c'est un
outil de triage, pas l'unique porte d'accès aux soins.

Dispatch par flavor (ajouté après l'arrivée de DistilBERT dans le pool de
candidats via ``register_distilbert.py``) : le portail de promotion de
``promote.py`` compare uniquement ``f1_macro``/``critical_recall`` — il n'a
aucune notion de "type de modèle", donc ce qui a l'alias "production" peut
être soit un champion classique/embedding scikit-learn (``train.py``,
``mlflow.sklearn.log_model``), soit le DistilBERT fine-tuné
(``mlflow.transformers.log_model``). Ce loader détecte lequel des deux il a
réellement reçu et le charge avec le loader MLflow du flavor correspondant,
afin qu'une promotion ne casse jamais silencieusement l'API (comme c'était
le cas avant ce changement : ``mlflow.sklearn.load_model`` levait
"Model does not have the sklearn flavor" face à une version DistilBERT
promue, dégradant directement vers le fallback de démo).

``mlflow.transformers`` est importé de façon paresseuse (lazy), à l'intérieur de
``load_production_model``, et uniquement sur la branche qui en a réellement
besoin — l'image Docker installe uniquement les extras ``api``/``mlflow``
(voir ``Dockerfile``), PAS ``transformers`` (torch à lui seul pèse ~2 Go et
gonflerait chaque déploiement juste pour supporter un candidat qui est
normalement en "staging", pas en "production"). Un ``import mlflow.transformers``
au niveau module ferait échouer purement et simplement l'import de ce module
— et donc le démarrage de l'API — partout où cet extra n'est pas installé,
même en servant un simple champion scikit-learn. L'import paresseux garde
le cas courant (sklearn en production) fonctionnel sans changement dans
l'image allégée ; servir réellement une version DistilBERT promue nécessite
en plus de construire l'image avec l'extra ``transformers`` (voir le
commentaire du ``Dockerfile``).
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

# Les deux flavors de modèle que le registry de ce projet peut actuellement
# contenir -- tout champion classique/embedding (mlflow.sklearn.log_model) et
# le fine-tune DistilBERT (mlflow.transformers.log_model, voir
# register_distilbert.py). Vérifiés dans cet ordre afin qu'un modèle loggé
# avec les deux flavors présents (ne devrait pas arriver ici, mais un fichier
# MLmodel peut en principe en lister plusieurs) préfère le loader sklearn,
# plus léger et déjà installé.
SUPPORTED_FLAVORS = ("sklearn", "transformers")


@dataclass
class LoadedModel:
    """Résultat d'une tentative de chargement du modèle de production."""

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
    Lequel des ``SUPPORTED_FLAVORS`` a été utilisé pour logger cette version
    de modèle.

    Lève ``ValueError`` si ce n'est ni l'un ni l'autre — ``load_production_model``
    intercepte cette exception et la transforme en ``LoadedModel.error``
    plutôt que de laisser l'API planter sur un flavor inattendu/futur.
    """
    info = mlflow.models.get_model_info(model_uri)
    for flavor in SUPPORTED_FLAVORS:
        if flavor in info.flavors:
            return flavor
    raise ValueError(f"Unsupported model flavor(s) {list(info.flavors)} -- expected one of {SUPPORTED_FLAVORS}")


def load_production_model(model_name: str = MLFLOW_REGISTERED_MODEL_NAME) -> LoadedModel:
    """
    Charge le modèle actuellement aliasé "production" dans le MLflow Registry.

    Ne lève jamais d'exception : tout échec (pas d'alias "production" défini,
    store MLflow injoignable, artefact corrompu, flavor non supporté/non
    détectable, ...) est capturé dans le ``LoadedModel.error`` retourné à la
    place, afin que l'API puisse démarrer en mode dégradé.

    Suppose que ``mlflow.set_tracking_uri`` a déjà été appelé par le point
    d'entrée (``main.py`` au démarrage de l'API, ou une fixture de test) —
    cette fonction ne le définit pas elle-même, ce qui la garde testable
    contre un store jetable sans toucher à la config globale, la même
    convention déjà utilisée par ``train.py`` / ``promote.py``.
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
            # Volontairement paresseux (lazy), et importé sous un alias plutôt que
            # `import mlflow.transformers` -- un simple import de sous-module ici
            # ferait que Python traiterait le nom externe `mlflow` comme local à
            # toute cette fonction (masquant l'import `mlflow` au niveau module
            # utilisé quelques lignes plus haut pour `mlflow.MlflowClient()`),
            # ce qui est un vrai risque d'UnboundLocalError, pas seulement une
            # question de style. Voir le docstring du module pour comprendre
            # pourquoi cet import est paresseux du tout.
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
