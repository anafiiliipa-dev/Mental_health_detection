"""Unit tests for src/mental_health/train/benchmark.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.benchmark import (
    ROBUST_SCORE_WEIGHTS,
    compute_robust_score,
    critical_recall_score,
    run_light_cv_benchmark,
    run_nested_cv_benchmark,
    sample_param_grid,
)
from mental_health.train.model_registry import build_model_registry, compute_boosted_class_weights

# ============================================================
# critical_recall_score
# ============================================================

class TestCriticalRecallScore:
    def test_perfect_predictions_on_critical_labels_score_one(self):
        y_true = ["Bipolar", "Schizophrenia", "ADHD"]
        y_pred = ["Bipolar", "Schizophrenia", "Depression"]  # ADHD wrong, doesn't matter
        assert critical_recall_score(y_true, y_pred) == pytest.approx(1.0)

    def test_missing_all_critical_labels_scores_zero(self):
        y_true = ["Bipolar", "Schizophrenia"]
        y_pred = ["ADHD", "Depression"]
        assert critical_recall_score(y_true, y_pred) == pytest.approx(0.0)

    def test_no_critical_labels_present_returns_zero_not_error(self):
        y_true = ["ADHD", "Depression"]
        y_pred = ["ADHD", "ADHD"]
        assert critical_recall_score(y_true, y_pred) == pytest.approx(0.0)

    def test_uses_correct_casing_by_default(self):
        # Regression test: must match "Schizophrenia" (title case), not the
        # lowercase "schizophrenia" from the original notebook bug.
        y_true = ["Schizophrenia", "Schizophrenia"]
        y_pred = ["Schizophrenia", "ADHD"]
        assert critical_recall_score(y_true, y_pred) == pytest.approx(0.5)


# ============================================================
# compute_robust_score
# ============================================================

class TestComputeRobustScore:
    def test_matches_expected_weighting(self):
        score = compute_robust_score(f1_macro=1.0, recall_macro=1.0, critical_recall=1.0)
        assert score == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        assert sum(ROBUST_SCORE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_critical_recall_has_the_highest_weight(self):
        # Documents the deliberate priority: missing a critical-class post
        # is the costliest error, so critical_recall must dominate.
        assert ROBUST_SCORE_WEIGHTS["critical_recall"] > ROBUST_SCORE_WEIGHTS["f1_macro"]
        assert ROBUST_SCORE_WEIGHTS["critical_recall"] > ROBUST_SCORE_WEIGHTS["recall_macro"]

    def test_zero_metrics_give_zero_score(self):
        assert compute_robust_score(0.0, 0.0, 0.0) == pytest.approx(0.0)


# ============================================================
# sample_param_grid
# ============================================================

class TestSampleParamGrid:
    def test_empty_grid_returns_single_empty_dict(self):
        assert sample_param_grid({}) == [{}]

    def test_returns_all_combinations_when_under_the_cap(self):
        grid = {"clf__C": [0.5, 1.0]}
        candidates = sample_param_grid(grid, max_candidates=10)
        assert len(candidates) == 2
        assert {"clf__C": 0.5} in candidates
        assert {"clf__C": 1.0} in candidates

    def test_caps_at_max_candidates(self):
        grid = {"clf__C": [0.5, 1.0, 2.0, 5.0], "clf__alpha": [0.1, 0.2]}
        candidates = sample_param_grid(grid, max_candidates=3)
        assert len(candidates) == 3

    def test_is_deterministic_for_a_given_random_state(self):
        grid = {"clf__C": [0.5, 1.0, 2.0, 5.0]}
        a = sample_param_grid(grid, max_candidates=2, random_state=7)
        b = sample_param_grid(grid, max_candidates=2, random_state=7)
        assert a == b


# ============================================================
# run_light_cv_benchmark / run_nested_cv_benchmark (smoke tests)
# ============================================================

def _tiny_dataset():
    X = pd.Series(
        [
            "I feel anxious all the time about everything",
            "worry and panic every single day of my life",
            "racing thoughts and no sleep for days now",
            "manic episodes followed by crushing lows",
            "I hear voices when nobody is around me",
            "paranoid thoughts about people watching me",
            "I feel hopeless and empty most days lately",
            "nothing brings me joy anymore these days",
        ]
        * 3
    )
    y = pd.Series(
        ["Anxiety", "Anxiety", "Bipolar", "Bipolar", "Schizophrenia", "Schizophrenia", "Depression", "Depression"]
        * 3
    )
    return X, y


class TestRunLightCvBenchmark:
    def test_returns_one_summary_row_per_model(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        _, summary, best_params_log = run_light_cv_benchmark(X, y, registry, n_splits=2, max_candidates=2)

        assert set(summary["model"]) == set(registry.keys())
        assert set(best_params_log.keys()) == set(registry.keys())

    def test_summary_is_ranked_by_robust_score_descending(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        _, summary, _ = run_light_cv_benchmark(X, y, registry, n_splits=2, max_candidates=2)

        scores = summary["robust_score"].tolist()
        assert scores == sorted(scores, reverse=True)
        assert list(summary["robust_rank"]) == list(range(1, len(summary) + 1))


class TestRunNestedCvBenchmark:
    def test_returns_one_summary_row_per_model(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        _, summary, nested_best_params = run_nested_cv_benchmark(
            X, y, registry, outer_splits=2, inner_splits=2, max_candidates=2
        )

        assert set(summary["model"]) == set(registry.keys())
        assert set(nested_best_params.keys()) == set(registry.keys())

    def test_each_model_has_one_best_params_entry_per_outer_fold(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        _, _, nested_best_params = run_nested_cv_benchmark(
            X, y, registry, outer_splits=2, inner_splits=2, max_candidates=2
        )

        for model_name, entries in nested_best_params.items():
            assert len(entries) == 2, f"{model_name} should have 2 outer-fold entries"
