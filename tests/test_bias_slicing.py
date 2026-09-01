"""Unit tests for src/mental_health/train/bias_slicing.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.bias_slicing import (
    assign_length_slices,
    evaluate_class_slices,
    evaluate_length_slices,
    evaluate_slices,
    summarize_fairness_gap,
)

# ============================================================
# assign_length_slices
# ============================================================


class TestAssignLengthSlices:
    def test_returns_one_label_per_row(self):
        texts = pd.Series(["a " * n for n in range(1, 13)])
        slices = assign_length_slices(texts, n_bins=3)
        assert len(slices) == len(texts)

    def test_short_texts_land_in_the_short_bucket(self):
        texts = pd.Series(["a"] * 4 + ["a " * 5] * 4 + ["a " * 20] * 4)
        slices = assign_length_slices(texts, n_bins=3)
        assert slices.iloc[0] == "short"
        assert slices.iloc[-1] == "long"

    def test_custom_labels(self):
        texts = pd.Series(["a " * n for n in range(1, 13)])
        slices = assign_length_slices(texts, n_bins=2, labels=["lo", "hi"])
        assert set(slices.unique()) <= {"lo", "hi"}

    def test_mismatched_label_count_raises(self):
        with pytest.raises(ValueError):
            assign_length_slices(pd.Series(["a", "b", "c"]), n_bins=3, labels=["only_one"])

    def test_degenerate_input_falls_back_instead_of_raising(self):
        # Every text has the same word count -- qcut can't form 3 distinct
        # quantile edges from this.
        texts = pd.Series(["same length text"] * 6)
        slices = assign_length_slices(texts, n_bins=3)
        assert len(slices) == 6


# ============================================================
# evaluate_slices
# ============================================================


class TestEvaluateSlices:
    def test_one_row_per_distinct_slice_value(self):
        y_true = ["Anxiety", "Anxiety", "Depression", "Depression"]
        y_pred = ["Anxiety", "Depression", "Depression", "Depression"]
        slice_labels = ["A", "A", "B", "B"]

        report = evaluate_slices(y_true, y_pred, slice_labels)

        assert set(report["slice"]) == {"A", "B"}
        assert report["support"].sum() == 4

    def test_sorted_worst_f1_first(self):
        y_true = ["Anxiety", "Anxiety", "Depression", "Depression"]
        y_pred = ["Anxiety", "Anxiety", "Anxiety", "Depression"]  # slice B: 1/2 correct, slice A: 2/2 correct
        slice_labels = ["A", "A", "B", "B"]

        report = evaluate_slices(y_true, y_pred, slice_labels)

        assert report.iloc[0]["slice"] == "B"
        assert report["f1_macro"].is_monotonic_increasing


# ============================================================
# evaluate_class_slices / evaluate_length_slices
# ============================================================


class TestEvaluateClassSlices:
    def test_flags_critical_labels(self):
        y_true = ["Bipolar", "Bipolar", "ADHD", "ADHD"]
        y_pred = ["Bipolar", "ADHD", "ADHD", "ADHD"]

        report = evaluate_class_slices(y_true, y_pred)

        bipolar_row = report[report["label"] == "Bipolar"].iloc[0]
        adhd_row = report[report["label"] == "ADHD"].iloc[0]
        assert bool(bipolar_row["is_critical"]) is True
        assert bool(adhd_row["is_critical"]) is False

    def test_recall_column_present(self):
        y_true = ["Bipolar", "ADHD"]
        y_pred = ["Bipolar", "ADHD"]
        report = evaluate_class_slices(y_true, y_pred)
        assert "recall" in report.columns
        assert "f1" in report.columns


class TestEvaluateLengthSlices:
    def test_returns_a_row_per_bucket_present(self):
        texts = pd.Series(["a"] * 3 + ["a " * 5] * 3 + ["a " * 20] * 3)
        y_true = pd.Series(["Anxiety"] * 9)
        y_pred = pd.Series(["Anxiety"] * 9)

        report = evaluate_length_slices(texts, y_true, y_pred, n_bins=3)

        assert set(report["length_bucket"]) <= {"short", "medium", "long"}


# ============================================================
# summarize_fairness_gap
# ============================================================


class TestSummarizeFairnessGap:
    def test_computes_gap_between_best_and_worst(self):
        report = pd.DataFrame([
            {"label": "Bipolar", "recall": 0.5, "f1": 0.5, "support": 10, "is_critical": True},
            {"label": "ADHD", "recall": 0.9, "f1": 0.9, "support": 10, "is_critical": False},
        ])
        summary = summarize_fairness_gap(report.rename(columns={"label": "slice"}))
        assert summary["gap"] == pytest.approx(0.4)
        assert summary["worst_slice"] == "Bipolar"
        assert summary["best_slice"] == "ADHD"

    def test_empty_report_returns_zero_gap(self):
        summary = summarize_fairness_gap(pd.DataFrame())
        assert summary["gap"] == 0.0
        assert summary["worst_slice"] is None
