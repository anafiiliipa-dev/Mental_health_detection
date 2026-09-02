"""
Script ponctuel : republie le modèle "production" LOCAL actuel dans le
backend MLflow partagé (Neon Postgres + stockage compatible S3) utilisé pour
le déploiement Cloud Run de l'équipe.

Pourquoi cela existe : le backend partagé démarre vide — il n'existe pas de moyen
MLflow natif pour "copier" un modèle enregistré (plus ses métriques) entre deux
backends différents. Ce script fait la chose la plus simple qui fonctionne : charger
le modèle qui sert déjà localement, puis le logger + l'enregistrer + l'aliaser à nouveau
contre le backend partagé, exactement comme le fait le run ``champion_final`` de
train.py pour le backend local.

Utilisation (voir les lignes MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_ROOT commentées
dans .env — "MLflow — backend partagé") : décommenter ces deux lignes, renseigner
le nouveau mot de passe Neon, puis exécuter :

    python scripts/publish_to_shared_backend.py

Sûr à relancer : chaque exécution crée une nouvelle version sur le backend partagé et
repointe l'alias "production" vers elle — ne touche jamais au store local.
"""
from __future__ import annotations

import logging

import mlflow
from dotenv import load_dotenv

# Doit s'exécuter avant que l'import mlflow_config ci-dessous ne lise MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT depuis l'environnement — rien d'autre dans le chemin de code
# MLflow/train/promote de ce projet ne charge .env (seule l'app Streamlit le fait),
# donc ce script doit le faire lui-même.
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

    # Lire le modèle source depuis le store LOCAL explicitement (pas
    # ce vers quoi pointe l'environnement), afin que "qu'est-ce qu'on est en train de
    # republier" ne soit jamais ambigu même une fois MLFLOW_TRACKING_URI défini sur le backend
    # partagé pour le reste de ce script.
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
