"""
Unit tests for the torch/transformers-independent parts of
src/mental_health/train/distilbert_finetune.py.

The real fine-tuning path (``finetune_distilbert``/``run``) needs the heavy
``transformers``/``torch`` extra and realistically a GPU -- it is
intentionally NOT exercised here, same rationale as
``test_embedding_wrapper.py`` never downloading the real sentence-
transformers model. This module's import is lazy specifically so these
pure-Python helpers (label mapping, the metrics callback) can be unit
tested without that dependency installed at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.distilbert_finetune import _build_label_maps, _compute_metrics_fn

# ============================================================
# _build_label_maps
# ============================================================


class TestBuildLabelMaps:
    def test_maps_are_inverse_of_each_other(self):
        labels = ["Bipolar", "ADHD", "Bipolar", "Depression"]
        label_to_id, id_to_label = _build_label_maps(labels)

        for label, idx in label_to_id.items():
            assert id_to_label[idx] == label

    def test_ids_are_dense_and_zero_indexed(self):
        labels = ["Bipolar", "ADHD", "Depression"]
        label_to_id, _ = _build_label_maps(labels)
        assert set(label_to_id.values()) == set(range(len(set(labels))))

    def test_deterministic_ordering_from_sorted_labels(self):
        # Sorted, not insertion order -- so re-running on the same label
        # set always assigns the same ids regardless of row order.
        a, _ = _build_label_maps(["Bipolar", "ADHD"])
        b, _ = _build_label_maps(["ADHD", "Bipolar"])
        assert a == b


# ============================================================
# _compute_metrics_fn
# ============================================================


class TestComputeMetricsFn:
    def test_perfect_predictions_score_one_on_every_metric(self):
        id_to_label = {0: "ADHD", 1: "Bipolar"}
        compute = _compute_metrics_fn(id_to_label)

        logits = np.array([[10.0, 0.0], [0.0, 10.0]])  # argmax -> [0, 1]
        label_ids = np.array([0, 1])

        metrics = compute((logits, label_ids))

        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["recall_macro"] == pytest.approx(1.0)
        assert metrics["critical_recall"] == pytest.approx(1.0)
        assert metrics["mcc"] == pytest.approx(1.0)

    def test_returns_the_four_expected_keys(self):
        id_to_label = {0: "ADHD", 1: "Depression"}
        compute = _compute_metrics_fn(id_to_label)
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        label_ids = np.array([0, 0])

        metrics = compute((logits, label_ids))

        assert set(metrics.keys()) == {"f1_macro", "recall_macro", "critical_recall", "mcc"}
