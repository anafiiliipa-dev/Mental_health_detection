"""
Weekly drift-monitoring job (Phase 10 completion).

Scores a sampled batch of "new" messages (see ``mock_stream.py``) with the
current "production" model, compares the distribution of predicted labels
and text length against the training reference set using Evidently, and:

- writes an HTML drift report (uploaded as a GitHub Actions run artifact
  by the calling workflow),
- logs a summary run (drift metrics + the report itself) to the shared
  MLflow backend, under the ``mental_health_monitoring`` experiment.

Deliberately does NOT trigger retraining — that decision is explicitly
deferred to Phase 12 (automated retraining trigger). This script only
observes and reports.

Usage:
    python -m mental_health.monitoring.drift_check
    python -m mental_health.monitoring.drift_check --batch-size 100 --report-path drift.html
"""
from __future__ import annotations

import argparse
import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

# Must run before the mlflow_config import below reads MLFLOW_TRACKING_URI /
# MLFLOW_ARTIFACT_ROOT from the environment — same convention as
# train.py / promote.py / main.py. A no-op in CI, where these are set
# directly as GitHub Actions secrets/env vars.
load_dotenv()

from evidently.metric_preset import DataDriftPreset  # noqa: E402
from evidently.report import Report  # noqa: E402

from mental_health.api.model_loader import load_production_model  # noqa: E402
from mental_health.config.mlflow_config import (  # noqa: E402
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
)
from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH  # noqa: E402
from mental_health.data.cleaning import TEXT_COL  # noqa: E402
from mental_health.monitoring.mock_stream import (  # noqa: E402
    build_reference_and_holdout,
    sample_mock_batch,
)

logger = logging.getLogger(__name__)

MONITORING_EXPERIMENT_NAME = "mental_health_monitoring"
DEFAULT_BATCH_SIZE = 50


def _get_or_create_experiment(name: str, artifact_location: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)


def _score(model, texts: pd.Series) -> list[str]:
    return [str(p) for p in model.predict(list(texts))]


def _confidence(model, texts: pd.Series) -> list[float] | None:
    """
    Top-class predicted probability per row (``max(predict_proba(x))``) —
    ``None`` when the model has no ``predict_proba`` at all (defensive
    only: since Phase 11 the served champion is always calibrated via
    CalibratedClassifierCV, so this should always be available in
    practice, but drift_check.py must not hard-crash if a future change
    ever registers an uncalibrated model).

    Requested by a teammate reviewing the monitoring setup: label/length
    drift alone can miss a real shift — a model can keep predicting the
    same labels while becoming steadily less sure of them. Tracking
    confidence as its own numerical column lets Evidently flag that too.
    """
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(list(texts))
    return [float(row.max()) for row in proba]


def build_drift_frames(model, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the columns Evidently compares between the reference set and the
    current sampled batch: ``text_length`` and ``prediction`` (categorical/
    target drift) always, plus ``prediction_confidence`` (numerical drift)
    whenever the model exposes real probabilities.
    """
    reference_confidence = _confidence(model, reference_df[TEXT_COL])
    current_confidence = _confidence(model, current_df[TEXT_COL])

    reference = pd.DataFrame(
        {
            "text_length": reference_df[TEXT_COL].str.len(),
            "prediction": _score(model, reference_df[TEXT_COL]),
        }
    )
    current = pd.DataFrame(
        {
            "text_length": current_df[TEXT_COL].str.len(),
            "prediction": _score(model, current_df[TEXT_COL]),
        }
    )

    if reference_confidence is not None and current_confidence is not None:
        reference["prediction_confidence"] = reference_confidence
        current["prediction_confidence"] = current_confidence

    return reference, current


def run_drift_check(
    batch_size: int = DEFAULT_BATCH_SIZE, report_path: str = "drift_report.html", simulate_drift: bool = False
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    loaded = load_production_model()
    if not loaded.is_available:
        raise SystemExit(f"No production model available — cannot run drift check: {loaded.error}")

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    reference_df, holdout_pool = build_reference_and_holdout(df)
    current_df = sample_mock_batch(holdout_pool, n=batch_size, simulate_drift=simulate_drift)

    reference, current = build_drift_frames(loaded.model, reference_df, current_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(report_path)

    drift_result = report.as_dict()["metrics"][0]["result"]
    dataset_drift = bool(drift_result.get("dataset_drift", False))
    n_drifted = int(drift_result.get("number_of_drifted_columns", 0))

    logger.info(
        "Drift check: model v%s, batch_size=%d, dataset_drift=%s, drifted_columns=%d",
        loaded.version,
        len(current_df),
        dataset_drift,
        n_drifted,
    )

    experiment_id = _get_or_create_experiment(MONITORING_EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_param("model_version", loaded.version)
        mlflow.log_param("batch_size", len(current_df))
        mlflow.log_param("simulate_drift", simulate_drift)
        mlflow.log_metric("dataset_drift", int(dataset_drift))
        mlflow.log_metric("n_drifted_columns", n_drifted)
        mlflow.log_artifact(report_path)

    return {"dataset_drift": dataset_drift, "n_drifted_columns": n_drifted, "report_path": report_path}


def _write_github_output(result: dict) -> None:
    """
    When running inside a GitHub Actions step, expose the drift outcome as
    step outputs (``dataset_drift``, ``n_drifted_columns``) so the calling
    workflow can decide whether to open an alert issue. A no-op outside CI.
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as fh:
        fh.write(f"dataset_drift={str(result['dataset_drift']).lower()}\n")
        fh.write(f"n_drifted_columns={result['n_drifted_columns']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the drift-monitoring check against the production model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--report-path", type=str, default="drift_report.html")
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help=(
            "Deliberately sample a skewed batch instead of an honest holdout "
            "sample, so a drift alert reliably fires. For exercising the "
            "detection -> alert -> retrain loop on a predictable cadence "
            "while there is no live production traffic yet -- NOT a "
            "realistic traffic simulation."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_drift_check(batch_size=args.batch_size, report_path=args.report_path, simulate_drift=args.simulate_drift)
    logger.info("Drift check complete: %s", result)
    _write_github_output(result)


if __name__ == "__main__":
    main()
