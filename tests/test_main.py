"""Integration tests for src/mental_health/api/main.py (FastAPI TestClient)."""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mental_health.api.main as main_module
from mental_health.api.schemas import VALID_LABELS


@pytest.fixture
def mlflow_tmp_registry(tmp_path, monkeypatch):
    """Point the API's MLflow tracking URI at a throwaway store for this test only."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(main_module, "MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("test_main_exp", artifact_location=f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test_main_exp")
    return tmp_path


def _register_production_model() -> None:
    """Register a trivial-but-real text classifier aliased 'production'."""
    texts = ["I feel so anxious and worried all the time", "I am hopeless and empty, nothing motivates me"] * 5
    labels = ["Anxiety", "Depression"] * 5
    model = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression())])
    model.fit(texts, labels)

    with mlflow.start_run():
        mlflow.log_metric("f1_macro", 0.9)
        mlflow.log_metric("critical_recall", 0.8)
        model_info = mlflow.sklearn.log_model(model, name="model", registered_model_name="mental_health_classifier")

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        "mental_health_classifier", "production", model_info.registered_model_version
    )


class TestHealth:
    def test_health_ok_even_without_any_model(self, mlflow_tmp_registry):
        with TestClient(main_module.app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelInfoDegraded:
    def test_reports_unavailable_when_nothing_registered(self, mlflow_tmp_registry):
        with TestClient(main_module.app) as client:
            resp = client.get("/model-info")
        body = resp.json()
        assert resp.status_code == 200
        assert body["model_available"] is False
        assert body["version"] is None
        assert body["error"] is not None


class TestPredictDegraded:
    def test_falls_back_to_demo_prediction(self, mlflow_tmp_registry):
        with TestClient(main_module.app) as client:
            resp = client.post("/predict", json={"text": "I feel so anxious and panicked"})
        body = resp.json()
        assert resp.status_code == 200
        assert body["is_demo_fallback"] is True
        assert body["label"] in VALID_LABELS

    def test_rejects_empty_text(self, mlflow_tmp_registry):
        with TestClient(main_module.app) as client:
            resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_never_echoes_submitted_text(self, mlflow_tmp_registry):
        secret_text = "this exact sentence must never appear in the response"
        with TestClient(main_module.app) as client:
            resp = client.post("/predict", json={"text": secret_text})
        assert secret_text not in resp.text


class TestModelInfoAndPredictWithRealModel:
    def test_model_info_reports_the_registered_version(self, mlflow_tmp_registry):
        _register_production_model()
        with TestClient(main_module.app) as client:
            resp = client.get("/model-info")
        body = resp.json()
        assert body["model_available"] is True
        assert body["version"] == "1"
        assert body["metrics"]["f1_macro"] == pytest.approx(0.9)

    def test_predict_uses_the_real_model(self, mlflow_tmp_registry):
        _register_production_model()
        with TestClient(main_module.app) as client:
            resp = client.post("/predict", json={"text": "I feel so anxious and worried"})
        body = resp.json()
        assert resp.status_code == 200
        assert body["is_demo_fallback"] is False
        assert body["label"] in {"Anxiety", "Depression"}
        assert body["probabilities"] is not None
        assert np.isclose(sum(body["probabilities"].values()), 1.0, atol=1e-3)
