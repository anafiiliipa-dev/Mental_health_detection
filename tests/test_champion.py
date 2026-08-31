"""Unit tests for src/mental_health/train/champion.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.champion import (
    evaluate_final_model,
    select_champion_config,
    select_champion_params,
    train_final_model,
)
from mental_health.train.model_registry import build_model_registry, compute_boosted_class_weights

# ============================================================
# select_champion_config
# ============================================================

class TestSelectChampionConfig:
    def test_picks_the_top_robust_score_row(self):
        nested_summary = pd.DataFrame([
            {"model": "LinearSVC_balanced", "text_variant": "raw", "robust_score": 0.73, "critical_recall_mean": 0.68, "f1_macro_mean": 0.77},
            {"model": "LogReg_balanced", "text_variant": "raw", "robust_score": 0.71, "critical_recall_mean": 0.70, "f1_macro_mean": 0.72},
        ])
        champion = select_champion_config(nested_summary)
        assert champion == {"model_name": "LinearSVC_balanced", "text_variant": "raw"}

    def test_breaks_ties_on_critical_recall_then_f1(self):
        nested_summary = pd.DataFrame([
            {"model": "A", "text_variant": "raw", "robust_score": 0.70, "critical_recall_mean": 0.60, "f1_macro_mean": 0.80},
            {"model": "B", "text_variant": "raw", "robust_score": 0.70, "critical_recall_mean": 0.65, "f1_macro_mean": 0.70},
        ])
        champion = select_champion_config(nested_summary)
        assert champion["model_name"] == "B"  # higher critical_recall_mean wins the tie


# ============================================================
# select_champion_params
# ============================================================

class TestSelectChampionParams:
    def test_picks_the_most_frequent_params_across_outer_folds(self):
        nested_best_params = {
            "raw": {
                "best_params": {
                    "LinearSVC_balanced": [
                        {"outer_fold": 1, "best_params": {"clf__C": 0.5}},
                        {"outer_fold": 2, "best_params": {"clf__C": 0.5}},
                        {"outer_fold": 3, "best_params": {"clf__C": 2.0}},
                    ]
                }
            }
        }
        params = select_champion_params(nested_best_params, "raw", "LinearSVC_balanced")
        assert params == {"clf__C": 0.5}

    def test_raises_when_no_params_found(self):
        nested_best_params = {"raw": {"best_params": {"LinearSVC_balanced": [{"outer_fold": 1, "best_params": None}]}}}
        with pytest.raises(ValueError):
            select_champion_params(nested_best_params, "raw", "LinearSVC_balanced")


# ============================================================
# train_final_model / evaluate_final_model (integration-style smoke test)
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


class TestTrainAndEvaluateFinalModel:
    def test_trained_model_predicts_on_test_set(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        model = train_final_model(registry, "LinearSVC_balanced", {"clf__C": 1.0}, X, y)
        result = evaluate_final_model(model, X, y)

        assert 0.0 <= result["f1_macro"] <= 1.0
        assert 0.0 <= result["recall_macro"] <= 1.0
        assert 0.0 <= result["critical_recall"] <= 1.0
        assert len(result["y_pred"]) == len(y)

    def test_evaluate_returns_classification_report_and_confusion_matrix(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        model = train_final_model(registry, "LinearSVC_balanced", {"clf__C": 1.0}, X, y)
        result = evaluate_final_model(model, X, y)

        assert isinstance(result["classification_report"], pd.DataFrame)
        assert "label" in result["classification_report"].columns

        cm = result["confusion_matrix"]
        assert isinstance(cm, pd.DataFrame)
        assert cm.shape[0] == cm.shape[1]  # square matrix

    def test_evaluate_returns_mcc_and_pr_auc_per_class(self):
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        model = train_final_model(registry, "LinearSVC_balanced", {"clf__C": 1.0}, X, y)
        result = evaluate_final_model(model, X, y)

        assert -1.0 <= result["mcc"] <= 1.0
        assert result["pr_auc_per_class"] is not None
        assert set(result["pr_auc_per_class"].keys()) == set(y.unique())
        assert all(0.0 <= v <= 1.0 for v in result["pr_auc_per_class"].values())

    def test_train_final_model_does_not_mutate_the_registry(self):
        # clone() must be used internally — fitting the champion must not
        # leave a fitted estimator sitting inside model_registry, which
        # would silently corrupt any later re-use of the registry.
        X, y = _tiny_dataset()
        class_weights = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weights)

        train_final_model(registry, "LinearSVC_balanced", {"clf__C": 1.0}, X, y)

        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        with pytest.raises(NotFittedError):
            check_is_fitted(registry["LinearSVC_balanced"]["pipeline"].named_steps["clf"])
