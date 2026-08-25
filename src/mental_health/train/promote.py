"""
Explicit, script-driven promotion from "staging" to "production" in the
MLflow Model Registry.

Addresses a governance gap flagged in the initial audit (Étape F —
"blind spots"): MLflow answers "which model is in production?", but not
the symmetric question "who has the right to promote a model?". For a
solo project, the answer adopted here is: **only this script, with
explicit, documented thresholds — never a manual click in the MLflow UI.**

Promotion rule (deliberately simple, not over-engineered):

- If there is no current "production" version, the "staging" candidate is
  promoted automatically (bootstrap case — there's nothing to regress
  against).
- Otherwise, the candidate is promoted only if it does not regress on
  EITHER headline metric versus the current production model:
  ``f1_macro`` and ``critical_recall`` must both be >= the production
  model's values. ``critical_recall`` is checked because it's the
  clinically critical metric (Bipolar/Schizophrenia) — a model that trades
  critical recall for overall F1 is not an acceptable trade for this
  project, per the priorities documented in ``benchmark.py``.

This is intentionally NOT a statistical significance test (e.g. bootstrap
comparison) — that's flagged as a SHOULD-HAVE in the audit, not required
to close this phase. The threshold logic here is a simple, auditable
gate that can be replaced with a stricter one later without changing how
it's invoked.
"""
from __future__ import annotations

import logging

import mlflow
from mlflow.exceptions import MlflowException

from mental_health.config.mlflow_config import (
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    STAGING_ALIAS,
)

logger = logging.getLogger(__name__)

GATED_METRICS = ["f1_macro", "critical_recall"]


def get_metrics_for_version(client: mlflow.MlflowClient, model_name: str, version: str) -> dict:
    """Fetch the metrics logged on the MLflow run that produced this model version."""
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    return dict(run.data.metrics)


def evaluate_promotion(candidate_metrics: dict, production_metrics: dict | None) -> tuple[bool, str]:
    """
    Decide whether ``candidate_metrics`` should be promoted over
    ``production_metrics``. Returns (should_promote, human_readable_reason).
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
    Compare the current "staging" model version against the current
    "production" one (if any) and promote staging to production if it
    passes ``evaluate_promotion``.
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
