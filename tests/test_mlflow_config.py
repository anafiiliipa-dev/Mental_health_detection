"""Unit tests for src/mental_health/config/mlflow_config.py.

The module's constants are computed once at import time from the
environment, so these tests reload it under monkeypatched env vars rather
than asserting on the already-imported module (which would only reflect
whatever the environment happened to be when the test session started).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mental_health.config.mlflow_config as mlflow_config


def _reload():
    return importlib.reload(mlflow_config)


class TestLocalDefaults:
    def test_tracking_uri_defaults_to_local_sqlite_when_unset(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
        module = _reload()
        assert module.MLFLOW_TRACKING_URI.startswith("sqlite:///")
        assert module.MLFLOW_TRACKING_URI.endswith("mlflow.db")

    def test_artifact_root_defaults_to_local_folder_when_unset(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
        module = _reload()
        assert module.MLFLOW_ARTIFACT_ROOT.startswith("file:")
        assert module.MLFLOW_ARTIFACT_ROOT.endswith("mlruns")


class TestEnvironmentOverride:
    def test_tracking_uri_env_var_wins_when_set(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "postgresql://user:pw@host/mlflow")
        monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
        module = _reload()
        assert module.MLFLOW_TRACKING_URI == "postgresql://user:pw@host/mlflow"

    def test_artifact_root_env_var_wins_when_set(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", "s3://team-bucket/mlflow-artifacts")
        module = _reload()
        assert module.MLFLOW_ARTIFACT_ROOT == "s3://team-bucket/mlflow-artifacts"

    def test_unrelated_constants_are_unaffected_by_the_override(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "postgresql://user:pw@host/mlflow")
        monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", "s3://team-bucket/mlflow-artifacts")
        module = _reload()
        assert module.MLFLOW_EXPERIMENT_NAME == "mental_health_classical_ml"
        assert module.MLFLOW_REGISTERED_MODEL_NAME == "mental_health_classifier"
        assert module.STAGING_ALIAS == "staging"
        assert module.PRODUCTION_ALIAS == "production"


def teardown_module():
    # Leave the module in its default (env-unset) state for any test that
    # imports it after this file runs.
    import os

    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_ARTIFACT_ROOT", None)
    _reload()
