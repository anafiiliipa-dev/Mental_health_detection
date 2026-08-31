"""
Shared MLflow configuration — tracking store location, experiment name,
and the registered model name.

Extracted out of ``mental_health.train.train`` so that lightweight
consumers (the FastAPI service in particular) can read "where is MLflow
and what's the model called" without importing the full training stack
(scikit-learn model registry, benchmark, champion selection). Training
code and serving code should share this one source of truth, not each
redefine it.

Tracking store location is environment-overridable (Cloud Run / any
deployment target that can't rely on local disk): if ``MLFLOW_TRACKING_URI``
/ ``MLFLOW_ARTIFACT_ROOT`` are set in the environment, they win outright —
e.g. a shared Postgres backend + an ``s3://...`` artifact root for a team
deployment. Left unset, both fall back to the local SQLite file / local
``mlruns/`` folder used for solo dev and the existing Docker Compose setup —
so nothing changes for anyone who doesn't set these two variables.
"""
from __future__ import annotations

import os

from mental_health.config.paths import PROJECT_ROOT

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", f"file:{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = "mental_health_classical_ml"

# Model Registry: every champion trained by train.py is registered under
# this name and aliased "staging" first, then "production" only via the
# explicit promotion criteria in promote.py.
MLFLOW_REGISTERED_MODEL_NAME = "mental_health_classifier"

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"
