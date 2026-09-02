"""
Configuration MLflow partagée — emplacement du tracking store, nom de
l'expérience, et nom du modèle enregistré.

Extraite de ``mental_health.train.train`` afin que les consommateurs légers
(le service FastAPI en particulier) puissent lire "où se trouve MLflow et
comment s'appelle le modèle" sans importer toute la pile d'entraînement
(model registry scikit-learn, benchmark, sélection du champion). Le code
d'entraînement et le code de serving doivent partager cette unique source
de vérité, pas chacun la redéfinir de son côté.

L'emplacement du tracking store est surchargeable via l'environnement
(Cloud Run / toute cible de déploiement qui ne peut pas compter sur le
disque local) : si ``MLFLOW_TRACKING_URI`` / ``MLFLOW_ARTIFACT_ROOT`` sont
définis dans l'environnement, ils l'emportent purement et simplement —
par exemple un backend Postgres partagé + une racine d'artefacts
``s3://...`` pour un déploiement d'équipe. Laissés non définis, les deux
retombent sur le fichier SQLite local / le dossier ``mlruns/`` local
utilisés pour le développement en solo et la configuration Docker Compose
existante — donc rien ne change pour quiconque ne définit pas ces deux
variables.
"""
from __future__ import annotations

import os

from mental_health.config.paths import PROJECT_ROOT

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", f"file:{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = "mental_health_classical_ml"

# Model Registry : chaque champion entraîné par train.py est enregistré sous
# ce nom et aliasé "staging" d'abord, puis "production" uniquement via les
# critères de promotion explicites de promote.py.
MLFLOW_REGISTERED_MODEL_NAME = "mental_health_classifier"

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"
