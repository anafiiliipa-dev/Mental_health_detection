"""Unit/integration tests for src/mental_health/train/promote.py."""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.promote import (
    GATED_METRICS,
    evaluate_promotion,
    get_metrics_for_version,
    promote_staging_to_production,
)

# ============================================================
# evaluate_promotion (pure logic, no MLflow needed)
# ============================================================

class TestEvaluatePromotion:
    def test_bootstrap_case_always_promotes(self):
        should_promote, reason = evaluate_promotion({"f1_macro": 0.5, "critical_recall": 0.4}, None)
        assert should_promote is True
        assert "bootstrap" in reason.lower() or "no current production" in reason.lower()

    def test_promotes_when_candidate_strictly_better(self):
        candidate = {"f1_macro": 0.80, "critical_recall": 0.70}
        production = {"f1_macro": 0.75, "critical_recall": 0.65}
        should_promote, _ = evaluate_promotion(candidate, production)
        assert should_promote is True

    def test_promotes_when_candidate_exactly_equal(self):
        # Equal is not a regression — must still promote (fresher model, same quality).
        metrics = {"f1_macro": 0.75, "critical_recall": 0.65}
        should_promote, _ = evaluate_promotion(dict(metrics), dict(metrics))
        assert should_promote is True

    def test_rejects_when_f1_macro_regresses(self):
        candidate = {"f1_macro": 0.70, "critical_recall": 0.70}
        production = {"f1_macro": 0.75, "critical_recall": 0.65}
        should_promote, reason = evaluate_promotion(candidate, production)
        assert should_promote is False
        assert "f1_macro" in reason

    def test_rejects_when_critical_recall_regresses_even_if_f1_improves(self):
        # This is the whole point of gating on critical_recall separately:
        # a model must not trade critical-class recall for overall F1.
        candidate = {"f1_macro": 0.90, "critical_recall": 0.50}
        production = {"f1_macro": 0.75, "critical_recall": 0.65}
        should_promote, reason = evaluate_promotion(candidate, production)
        assert should_promote is False
        assert "critical_recall" in reason

    def test_gated_metrics_are_the_two_headline_metrics(self):
        assert set(GATED_METRICS) == {"f1_macro", "critical_recall"}


# ============================================================
# get_metrics_for_version / promote_staging_to_production
# (integration-style: real (temp) MLflow tracking store)
# ============================================================

@pytest.fixture
def mlflow_tmp_registry(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite store + artifact root for this test only."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test_promote_exp", artifact_location=f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test_promote_exp")
    return tmp_path


def _log_and_register_version(model_name: str, metrics: dict) -> str:
    """Train a trivial model, log it with the given metrics, register it, return the version."""
    X = np.random.rand(20, 3)
    y = (np.random.rand(20) > 0.5).astype(int)
    model = LogisticRegression().fit(X, y)

    with mlflow.start_run():
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        model_info = mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)

    return model_info.registered_model_version


class TestGetMetricsForVersion:
    def test_returns_the_logged_metrics(self, mlflow_tmp_registry):
        version = _log_and_register_version("m1", {"f1_macro": 0.81, "critical_recall": 0.66})
        client = mlflow.MlflowClient()
        metrics = get_metrics_for_version(client, "m1", version)
        assert metrics["f1_macro"] == pytest.approx(0.81)
        assert metrics["critical_recall"] == pytest.approx(0.66)


class TestPromoteStagingToProduction:
    def test_bootstrap_promotion_when_no_production_exists(self, mlflow_tmp_registry):
        client = mlflow.MlflowClient()
        version = _log_and_register_version("m2", {"f1_macro": 0.80, "critical_recall": 0.70})
        client.set_registered_model_alias("m2", "staging", version)

        result = promote_staging_to_production("m2")

        assert result["promoted"] is True
        assert result["previous_production_version"] is None
        prod = client.get_model_version_by_alias("m2", "production")
        assert prod.version == version

    def test_does_not_promote_a_regression(self, mlflow_tmp_registry):
        client = mlflow.MlflowClient()

        prod_version = _log_and_register_version("m3", {"f1_macro": 0.85, "critical_recall": 0.75})
        client.set_registered_model_alias("m3", "production", prod_version)

        staging_version = _log_and_register_version("m3", {"f1_macro": 0.70, "critical_recall": 0.60})
        client.set_registered_model_alias("m3", "staging", staging_version)

        result = promote_staging_to_production("m3")

        assert result["promoted"] is False
        current_prod = client.get_model_version_by_alias("m3", "production")
        assert current_prod.version == prod_version, "production alias must not move on a rejected promotion"

    def test_promotes_an_improvement_over_current_production(self, mlflow_tmp_registry):
        client = mlflow.MlflowClient()

        prod_version = _log_and_register_version("m4", {"f1_macro": 0.75, "critical_recall": 0.65})
        client.set_registered_model_alias("m4", "production", prod_version)

        staging_version = _log_and_register_version("m4", {"f1_macro": 0.80, "critical_recall": 0.70})
        client.set_registered_model_alias("m4", "staging", staging_version)

        result = promote_staging_to_production("m4")

        assert result["promoted"] is True
        current_prod = client.get_model_version_by_alias("m4", "production")
        assert current_prod.version == staging_version
