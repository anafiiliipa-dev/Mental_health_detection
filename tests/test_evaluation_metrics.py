"""Unit tests for src/mental_health/train/evaluation_metrics.py."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from mental_health.train.evaluation_metrics import (
    compute_brier_score,
    compute_ece,
    compute_mcc,
    compute_pr_auc_per_class,
    get_ranking_scores,
    paired_bootstrap_test,
)


class TestComputeMcc:
    def test_perfect_predictions_score_one(self):
        y_true = ["A", "B", "A", "B", "A"]
        assert compute_mcc(y_true, y_true) == pytest.approx(1.0)

    def test_worse_than_random_can_go_negative(self):
        y_true = ["A", "A", "A", "B", "B", "B"]
        y_pred = ["B", "B", "B", "A", "A", "A"]
        assert compute_mcc(y_true, y_pred) == pytest.approx(-1.0)


class TestGetRankingScores:
    def test_uses_predict_proba_when_available(self):
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 5)
        y = ["A", "B", "A", "B"] * 5
        model = LogisticRegression().fit(X, y)
        scores = get_ranking_scores(model, X)
        assert scores.shape == (len(X), 2)

    def test_falls_back_to_decision_function(self):
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 5)
        y = ["A", "B", "A", "B"] * 5
        model = LinearSVC().fit(X, y)
        assert not hasattr(model, "predict_proba")
        scores = get_ranking_scores(model, X)
        assert scores.shape == (len(X),)  # binary LinearSVC: one score per sample

    def test_returns_none_when_neither_is_available(self):
        class NoScores:
            pass

        assert get_ranking_scores(NoScores(), None) is None


class TestComputePrAucPerClass:
    def test_perfect_separation_scores_near_one(self):
        labels = ["A", "B"]
        y_true = ["A", "A", "B", "B"]
        scores = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])
        result = compute_pr_auc_per_class(y_true, scores, labels)
        assert result["A"] == pytest.approx(1.0)
        assert result["B"] == pytest.approx(1.0)

    def test_returns_one_score_per_label(self):
        labels = ["A", "B", "C"]
        y_true = ["A", "B", "C", "A"]
        scores = np.random.RandomState(0).rand(4, 3)
        result = compute_pr_auc_per_class(y_true, scores, labels)
        assert set(result.keys()) == set(labels)


class TestPairedBootstrapTest:
    def test_identical_predictions_have_zero_observed_diff(self):
        y_true = ["A", "B", "A", "B", "A", "B"] * 5
        y_pred = ["A", "B", "A", "B", "A", "A"] * 5
        result = paired_bootstrap_test(y_true, y_pred, y_pred, n_bootstrap=200)
        assert result["observed_diff"] == pytest.approx(0.0)
        assert result["significant_at_0.05"] is False

    def test_clearly_better_model_is_flagged_significant(self):
        rng = np.random.RandomState(0)
        y_true = rng.choice(["A", "B"], size=200)
        y_pred_perfect = y_true.copy()
        y_pred_random = rng.choice(["A", "B"], size=200)
        result = paired_bootstrap_test(y_true, y_pred_perfect, y_pred_random, n_bootstrap=300, random_state=1)
        assert result["observed_diff"] > 0
        assert result["significant_at_0.05"] is True

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paired_bootstrap_test(["A", "B"], ["A"], ["A", "B"])


class TestComputeBrierScore:
    def test_perfect_confident_predictions_score_zero(self):
        labels = ["A", "B"]
        y_true = ["A", "B", "A", "B"]
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert compute_brier_score(y_true, proba, labels) == pytest.approx(0.0)

    def test_confidently_wrong_predictions_score_high(self):
        labels = ["A", "B"]
        y_true = ["A", "A"]
        proba = np.array([[0.0, 1.0], [0.0, 1.0]])  # fully confident in the wrong class
        assert compute_brier_score(y_true, proba, labels) == pytest.approx(2.0)

    def test_uniform_guessing_lands_between_perfect_and_worst(self):
        labels = ["A", "B"]
        y_true = ["A", "B"]
        proba = np.array([[0.5, 0.5], [0.5, 0.5]])
        score = compute_brier_score(y_true, proba, labels)
        assert 0.0 < score < 2.0


class TestComputeEce:
    def test_perfectly_calibrated_confidence_scores_zero(self):
        # 10 samples all predicted "A" at confidence 0.7 — 7 of them
        # actually are "A" (correct), 3 are "B" (wrong): accuracy 0.7
        # matches confidence 0.7 exactly within the bin.
        labels = ["A", "B"]
        y_true = ["A"] * 7 + ["B"] * 3
        proba = np.array([[0.7, 0.3]] * 10)
        assert compute_ece(y_true, proba, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_overconfident_wrong_predictions_score_high(self):
        labels = ["A", "B"]
        y_true = ["A"] * 10
        proba = np.array([[0.0, 1.0]] * 10)  # 100% confident, always wrong
        assert compute_ece(y_true, proba, labels, n_bins=10) == pytest.approx(1.0)
