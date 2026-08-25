"""
Shared MLflow configuration — tracking store location, experiment name,
and the registered model name.

Extracted out of ``mental_health.train.train`` so that lightweight
consumers (the FastAPI service in particular) can read "where is MLflow
and what's the model called" without importing the full training stack
(scikit-learn model registry, benchmark, champion selection). Training
code and serving code should share this one source of truth, not each
redefine it.
"""
from __future__ import annotations

from mental_health.config.paths import PROJECT_ROOT

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = f"file:{PROJECT_ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT_NAME = "mental_health_classical_ml"

# Model Registry: every champion trained by train.py is registered under
# this name and aliased "staging" first, then "production" only via the
# explicit promotion criteria in promote.py.
MLFLOW_REGISTERED_MODEL_NAME = "mental_health_classifier"

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"
