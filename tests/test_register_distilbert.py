"""
Unit tests for the torch/transformers-independent parts of
src/mental_health/train/register_distilbert.py.

``register_distilbert_candidate`` (the actual MLflow registration path)
needs the heavy ``transformers``/``torch`` extra plus a real fine-tuned
model directory on disk -- intentionally NOT exercised here, same
rationale as ``test_distilbert_finetune.py`` never running the real
fine-tune. ``load_distilbert_metrics`` has no such dependency (it only
reads a CSV), so it is fully unit tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.register_distilbert import EXPECTED_METRICS, load_distilbert_metrics

# ============================================================
# load_distilbert_metrics
# ============================================================


class TestLoadDistilbertMetrics:
    def test_reads_the_four_headline_metrics(self, tmp_path):
        path = tmp_path / "distilbert_metrics.csv"
        pd.DataFrame([{
            "model": "DistilBERT_finetuned",
            "text_variant": "raw",
            "f1_macro": 0.767,
            "recall_macro": 0.766,
            "critical_recall": 0.689,
            "mcc": 0.743,
        }]).to_csv(path, index=False)

        metrics = load_distilbert_metrics(path)

        assert metrics == {
            "f1_macro": pytest.approx(0.767),
            "recall_macro": pytest.approx(0.766),
            "critical_recall": pytest.approx(0.689),
            "mcc": pytest.approx(0.743),
        }

    def test_ignores_extra_columns_like_model_and_text_variant(self, tmp_path):
        path = tmp_path / "distilbert_metrics.csv"
        pd.DataFrame([{
            "model": "DistilBERT_finetuned",
            "text_variant": "raw",
            "f1_macro": 0.5,
            "recall_macro": 0.5,
            "critical_recall": 0.5,
            "mcc": 0.5,
        }]).to_csv(path, index=False)

        metrics = load_distilbert_metrics(path)

        assert set(metrics.keys()) == set(EXPECTED_METRICS)

    def test_raises_on_more_than_one_row(self, tmp_path):
        path = tmp_path / "distilbert_metrics.csv"
        pd.DataFrame([
            {"f1_macro": 0.5, "recall_macro": 0.5, "critical_recall": 0.5, "mcc": 0.5},
            {"f1_macro": 0.6, "recall_macro": 0.6, "critical_recall": 0.6, "mcc": 0.6},
        ]).to_csv(path, index=False)

        with pytest.raises(ValueError, match="exactly one row"):
            load_distilbert_metrics(path)

    def test_raises_on_missing_metric_column(self, tmp_path):
        path = tmp_path / "distilbert_metrics.csv"
        pd.DataFrame([{"f1_macro": 0.5, "recall_macro": 0.5, "critical_recall": 0.5}]).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing expected metric"):
            load_distilbert_metrics(path)
