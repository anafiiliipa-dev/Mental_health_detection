"""Unit tests for src/mental_health/train/model_registry.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.config.paths import CLASS_LABELS
from mental_health.train.model_registry import (
    CLASS_WEIGHT_BOOST,
    build_model_registry,
    compute_boosted_class_weights,
)

# ============================================================
# compute_boosted_class_weights
# ============================================================

class TestComputeBoostedClassWeights:
    def test_returns_a_weight_for_every_class_present(self):
        y_train = pd.Series(["ADHD"] * 10 + ["Anxiety"] * 5 + ["Schizophrenia"] * 2)
        weights = compute_boosted_class_weights(y_train)
        assert set(weights.keys()) == {"ADHD", "Anxiety", "Schizophrenia"}

    def test_rarer_class_gets_higher_base_weight(self):
        y_train = pd.Series(["ADHD"] * 100 + ["Schizophrenia"] * 10)
        weights = compute_boosted_class_weights(y_train)
        assert weights["Schizophrenia"] > weights["ADHD"]

    def test_schizophrenia_boost_is_applied_on_correct_casing(self):
        # Regression test: with the label capitalized as "Schizophrenia"
        # (the corrected casing from cleaning.py), the boost must actually
        # fire. Compare against the unboosted balanced weight computed
        # directly with sklearn to isolate the boost factor.
        y_train = pd.Series(["ADHD"] * 50 + ["Schizophrenia"] * 10 + ["Bipolar"] * 20)
        classes = np.unique(y_train)
        unboosted = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))

        boosted = compute_boosted_class_weights(y_train)

        assert boosted["Schizophrenia"] == pytest.approx(
            unboosted["Schizophrenia"] * CLASS_WEIGHT_BOOST["Schizophrenia"]
        )
        assert boosted["Bipolar"] == pytest.approx(
            unboosted["Bipolar"] * CLASS_WEIGHT_BOOST["Bipolar"]
        )
        # A non-critical label must be untouched by the boost.
        assert boosted["ADHD"] == pytest.approx(unboosted["ADHD"])

    def test_missing_critical_label_does_not_error(self):
        # If a batch simply doesn't contain "Bipolar", the boost loop must
        # skip it silently rather than raising a KeyError.
        y_train = pd.Series(["ADHD"] * 10 + ["Schizophrenia"] * 5)
        weights = compute_boosted_class_weights(y_train)
        assert "Bipolar" not in weights


# ============================================================
# build_model_registry
# ============================================================

class TestBuildModelRegistry:
    def test_contains_the_five_expected_candidates(self):
        registry = build_model_registry({"ADHD": 1.0})
        assert set(registry.keys()) == {
            "LinearSVC_balanced",
            "LinearSVC_plain",
            "LogReg_balanced",
            "LogReg_plain",
            "MultinomialNB",
        }

    def test_every_candidate_has_a_pipeline_and_param_grid(self):
        registry = build_model_registry({"ADHD": 1.0})
        for name, spec in registry.items():
            assert "pipeline" in spec, f"{name} missing 'pipeline'"
            assert "param_grid" in spec, f"{name} missing 'param_grid'"
            assert isinstance(spec["param_grid"], dict)

    def test_balanced_variants_receive_the_class_weight_dict(self):
        class_weight_dict = {"ADHD": 0.8, "Schizophrenia": 2.1}
        registry = build_model_registry(class_weight_dict)

        svc_clf = registry["LinearSVC_balanced"]["pipeline"].named_steps["clf"]
        assert svc_clf.class_weight == class_weight_dict

        logreg_clf = registry["LogReg_balanced"]["pipeline"].named_steps["clf"]
        assert logreg_clf.class_weight == class_weight_dict

    def test_plain_variants_do_not_receive_class_weight(self):
        registry = build_model_registry({"ADHD": 0.8})

        svc_clf = registry["LinearSVC_plain"]["pipeline"].named_steps["clf"]
        assert svc_clf.class_weight is None

    def test_pipelines_are_fittable_on_tiny_synthetic_data(self):
        # End-to-end smoke test: every candidate must actually fit without
        # error on a tiny (but valid) text classification sample.
        X = pd.Series(
            [
                "I feel anxious all the time about everything",
                "racing thoughts and no sleep for days",
                "I hear voices when nobody is around",
                "I feel hopeless and empty most days",
            ]
            * 3
        )
        y = pd.Series(["Anxiety", "Bipolar", "Schizophrenia", "Depression"] * 3)

        class_weight_dict = compute_boosted_class_weights(y)
        registry = build_model_registry(class_weight_dict)

        for name, spec in registry.items():
            pipeline = spec["pipeline"]
            pipeline.fit(X, y)
            preds = pipeline.predict(X)
            assert len(preds) == len(y), f"{name} produced wrong number of predictions"


def test_boosted_labels_are_a_subset_of_class_labels():
    # Guards against a future contributor introducing a typo in
    # CLASS_WEIGHT_BOOST that doesn't match any real label.
    assert set(CLASS_WEIGHT_BOOST).issubset(set(CLASS_LABELS))
