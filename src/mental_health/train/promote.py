"""
Promotion explicite et pilotée par script de "staging" vers "production" dans le
MLflow Model Registry.

Répond à une lacune de gouvernance signalée dans l'audit initial (Étape F —
"blind spots") : MLflow répond à "quel modèle est en production ?", mais pas
à la question symétrique "qui a le droit de promouvoir un modèle ?". Pour un
projet solo, la réponse adoptée ici est : **seul ce script, avec des
seuils explicites et documentés — jamais un clic manuel dans l'UI MLflow.**

Règle de promotion (délibérément simple, non sur-conçue) :

- S'il n'y a pas de version "production" actuelle, le candidat "staging" est
  promu automatiquement (cas de bootstrap — il n'y a rien contre quoi régresser).
- Sinon, le candidat n'est promu que s'il ne régresse sur AUCUNE
  des deux métriques principales par rapport au modèle de production actuel :
  ``f1_macro`` et ``critical_recall`` doivent tous deux être >= aux
  valeurs du modèle de production. ``critical_recall`` est vérifié car c'est la
  métrique cliniquement critique (Bipolar/Schizophrenia) — un modèle qui sacrifie
  le rappel critique pour un meilleur F1 global n'est pas un compromis acceptable pour ce
  projet, selon les priorités documentées dans ``benchmark.py``.

Ceci n'est délibérément PAS un test de significativité statistique (par ex. une
comparaison par bootstrap) — c'est signalé comme un SHOULD-HAVE dans l'audit, pas requis
pour clore cette phase. La logique de seuil ici est une porte simple et
auditable qui pourra être remplacée par une plus stricte plus tard sans changer la façon
dont elle est invoquée.
"""
from __future__ import annotations

import logging

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException

# Doit s'exécuter avant que l'import mlflow_config ci-dessous ne lise MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT depuis l'environnement (par ex. un backend d'équipe partagé
# au lieu du SQLite local par défaut) — voir la docstring de mlflow_config.py.
load_dotenv()

from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    STAGING_ALIAS,
)

logger = logging.getLogger(__name__)

GATED_METRICS = ["f1_macro", "critical_recall"]


def get_metrics_for_version(client: mlflow.MlflowClient, model_name: str, version: str) -> dict:
    """Récupère les métriques enregistrées sur le run MLflow ayant produit cette version du modèle."""
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    return dict(run.data.metrics)


def evaluate_promotion(candidate_metrics: dict, production_metrics: dict | None) -> tuple[bool, str]:
    """
    Décide si ``candidate_metrics`` doit être promu par rapport à
    ``production_metrics``. Retourne (should_promote, human_readable_reason).
    """
    if production_metrics is None:
        return True, "No current production model — bootstrap promotion."

    regressions = [
        f"{metric} regressed ({candidate_metrics.get(metric):.4f} < {production_metrics.get(metric):.4f})"
        for metric in GATED_METRICS
        if candidate_metrics.get(metric, 0.0) < production_metrics.get(metric, 0.0)
    ]

    if regressions:
        return False, "Not promoted: " + "; ".join(regressions)

    return True, "Promoted: no regression on " + " or ".join(GATED_METRICS)


def promote_staging_to_production(model_name: str = MLFLOW_REGISTERED_MODEL_NAME) -> dict:
    """
    Compare la version de modèle "staging" actuelle à la version
    "production" actuelle (le cas échéant) et promeut staging vers production si elle
    passe ``evaluate_promotion``.
    """
    client = mlflow.MlflowClient()

    staging_version = client.get_model_version_by_alias(model_name, STAGING_ALIAS)
    candidate_metrics = get_metrics_for_version(client, model_name, staging_version.version)

    try:
        production_version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        production_metrics = get_metrics_for_version(client, model_name, production_version.version)
    except MlflowException:
        production_version = None
        production_metrics = None

    should_promote, reason = evaluate_promotion(candidate_metrics, production_metrics)
    logger.info("Promotion decision for %s v%s: %s", model_name, staging_version.version, reason)

    if should_promote:
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, staging_version.version)
        logger.info("'%s' v%s is now aliased 'production'", model_name, staging_version.version)

    return {
        "promoted": should_promote,
        "reason": reason,
        "candidate_version": staging_version.version,
        "candidate_metrics": candidate_metrics,
        "previous_production_version": production_version.version if production_version else None,
        "previous_production_metrics": production_metrics,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    result = promote_staging_to_production()
    logger.info("Result: %s", result)
