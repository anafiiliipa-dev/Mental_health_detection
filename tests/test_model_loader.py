"""Unit/integration tests for src/mental_health/api/model_loader.py."""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.api.model_loader import load_production_model


@pytest.fixture
def mlflow_tmp_registry(tmp_path):
    """Point MLflow at a throwaway SQLite store + artifact root for this test only."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test_model_loader_exp", artifact_location=f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test_model_loader_exp")
    return tmp_path


def _log_and_register_version(model_name: str, metrics: dict) -> tuple[str, str]:
    """Train a trivial model, log it with the given metrics, register it, return (version, run_id)."""
    X = np.random.rand(20, 3)
    y = (np.random.rand(20) > 0.5).astype(int)
    model = LogisticRegression().fit(X, y)

    with mlflow.start_run() as run:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        model_info = mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)

    return model_info.registered_model_version, run.info.run_id


class TestLoadProductionModel:
    def test_loads_the_model_aliased_production(self, mlflow_tmp_registry):
        client = mlflow.MlflowClient()
        version, run_id = _log_and_register_version("loader_m1", {"f1_macro": 0.8, "critical_recall": 0.7})
        client.set_registered_model_alias("loader_m1", "production", version)

        result = load_production_model("loader_m1")

        assert result.is_available is True
        assert result.error is None
        assert result.version == version
        assert result.run_id == run_id
        assert result.metrics["f1_macro"] == pytest.approx(0.8)
        # Sanity check it's a real usable estimator, not a stub.
        assert hasattr(result.model, "predict")

    def test_no_production_alias_returns_unavailable_not_raise(self, mlflow_tmp_registry):
        # Registry has no model at all registered under this name.
        result = load_production_model("does_not_exist_model")

        assert result.is_available is False
        assert result.model is None
        assert result.error is not None

    def test_unavailable_when_alias_exists_but_only_staging(self, mlflow_tmp_registry):
        client = mlflow.MlflowClient()
        version, _ = _log_and_register_version("loader_m2", {"f1_macro": 0.8, "critical_recall": 0.7})
        client.set_registered_model_alias("loader_m2", "staging", version)  # not "production"

        result = load_production_model("loader_m2")

        assert result.is_available is False
        assert result.error is not None
