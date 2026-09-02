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


class _FakeTransformersPipeline:
    """
    Stands in for a HF text-classification pipeline -- returns real label
    strings + softmax-normalised scores for every class, the shape
    ``_predict_with_transformers_model`` expects when it passes
    ``top_k=None`` explicitly on every call (see that function's
    docstring for why it doesn't rely on the pipeline's own default).
    Avoids requiring the heavy transformers/torch extra just to test the
    dispatch/parsing logic on the main.py side.
    """

    def __call__(self, texts: list[str], top_k: int | None = 1):
        assert top_k is None, "must request every class explicitly, not rely on the pipeline's own top_k default"
        return [[
            {"label": "Anxiety", "score": 0.82},
            {"label": "Depression", "score": 0.10},
            {"label": "ADHD", "score": 0.08},
        ]]


class TestPredictWithTransformersFlavor:
    """
    A promoted DistilBERT candidate must be servable through /predict too
    -- not just loadable (see test_model_loader.py's TestFlavorDispatch).
    Bypasses load_production_model entirely (monkeypatched) so this test
    doesn't need a real transformers-flavored MLflow model or the
    transformers/torch extra installed.
    """

    def test_predict_uses_the_transformers_pipeline_when_flavor_is_transformers(self, mlflow_tmp_registry, mocker):
        from mental_health.api.model_loader import LoadedModel

        fake_loaded = LoadedModel(
            model=_FakeTransformersPipeline(),
            flavor="transformers",
            version="7",
            run_id="fake_run_id",
            metrics={"f1_macro": 0.95, "critical_recall": 0.90},
            error=None,
        )
        mocker.patch.object(main_module, "load_production_model", return_value=fake_loaded)

        with TestClient(main_module.app) as client:
            resp = client.post("/predict", json={"text": "I feel really anxious lately"})

        body = resp.json()
        assert resp.status_code == 200
        assert body["is_demo_fallback"] is False
        assert body["label"] == "Anxiety"
        assert body["confidence"] == pytest.approx(0.82)
        assert body["probabilities"] == {"Anxiety": pytest.approx(0.82), "Depression": pytest.approx(0.10), "ADHD": pytest.approx(0.08)}
        assert np.isclose(sum(body["probabilities"].values()), 1.0, atol=1e-3)


class TestPredictLogging:
    """
    The structured log for every /predict call must include the full
    per-class probabilities (not just predicted_label) -- useful for
    spotting low-confidence predictions and feeding monitoring/drift
    checks later -- while still never logging the submitted text itself.
    """

    def test_predict_request_log_includes_probabilities_but_never_the_text(self, mlflow_tmp_registry, caplog):
        _register_production_model()
        secret_text = "I feel so anxious and worried"
        with caplog.at_level("INFO", logger="mental_health.api.main"):
            with TestClient(main_module.app) as client:
                client.post("/predict", json={"text": secret_text})

        records = [r for r in caplog.records if r.getMessage() == "predict request"]
        assert len(records) == 1
        record = records[0]

        assert record.probabilities is not None
        assert set(record.probabilities.keys()) == {"Anxiety", "Depression"}
        assert np.isclose(sum(record.probabilities.values()), 1.0, atol=1e-3)
        assert secret_text not in str(record.__dict__)
