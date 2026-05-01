"""Unit tests for model prediction utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.config.paths import CLASS_LABELS
from mental_health.models.services import fallback_demo_prediction, predict_with_model

# ============================================================
# fallback_demo_prediction
# ============================================================

class TestFallbackDemoPrediction:
    def test_returns_three_values(self):
        label, confidence, df = fallback_demo_prediction("I feel anxious all the time")
        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert isinstance(df, pd.DataFrame)

    def test_label_is_in_class_labels(self):
        label, _, _ = fallback_demo_prediction("I feel hopeless and empty")
        assert label in CLASS_LABELS

    def test_probabilities_sum_to_one(self):
        _, _, df = fallback_demo_prediction("I hear voices watching me")
        total = df["Probability"].sum()
        assert abs(total - 1.0) < 5e-3, f"Probabilities sum to {total}, expected ~1.0"

    def test_dataframe_has_correct_columns(self):
        _, _, df = fallback_demo_prediction("racing thoughts, no sleep")
        assert "Class" in df.columns
        assert "Probability" in df.columns

    def test_dataframe_sorted_descending(self):
        _, _, df = fallback_demo_prediction("I feel hopeless")
        probs = df["Probability"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_schizophrenia_keyword_match(self):
        label, confidence, _ = fallback_demo_prediction("I hear voices when nobody is around")
        assert label == "Schizophrenia"
        assert confidence > 0.8

    def test_depression_keyword_match(self):
        label, _, _ = fallback_demo_prediction("I feel hopeless and worthless every day")
        assert label == "Depression"

    def test_empty_text_returns_default(self):
        label, confidence, df = fallback_demo_prediction("")
        assert label in CLASS_LABELS
        assert 0.0 < confidence <= 1.0
        assert isinstance(df, pd.DataFrame)

    def test_confidence_is_in_valid_range(self):
        for text in ["random text", "voices watching me", "hopeless empty"]:
            _, confidence, _ = fallback_demo_prediction(text)
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range"


# ============================================================
# predict_with_model (with a mock sklearn model)
# ============================================================

class MockModelWithProba:
    """Minimal sklearn-compatible model with predict_proba."""
    classes_ = CLASS_LABELS

    def predict(self, X):
        return ["Anxiety"]

    def predict_proba(self, X):
        import numpy as np

        probs = np.zeros(len(CLASS_LABELS))
        probs[CLASS_LABELS.index("Anxiety")] = 0.9
        probs[CLASS_LABELS.index("Depression")] = 0.1
        return [probs]


class MockModelNoProba:
    """Minimal sklearn-compatible model without predict_proba."""

    def predict(self, X):
        return ["ADHD"]


class TestPredictWithModel:
    def test_with_proba_model_returns_correct_label(self):
        label, confidence, df = predict_with_model(MockModelWithProba(), "test text")
        assert label == "Anxiety"

    def test_with_proba_model_confidence_is_float(self):
        _, confidence, _ = predict_with_model(MockModelWithProba(), "test text")
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_with_proba_model_df_sorted_descending(self):
        _, _, df = predict_with_model(MockModelWithProba(), "test text")
        probs = df["Probability"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_without_proba_returns_none_confidence(self):
        label, confidence, df = predict_with_model(MockModelNoProba(), "test text")
        assert label == "ADHD"
        assert confidence is None

    def test_without_proba_df_has_single_row(self):
        _, _, df = predict_with_model(MockModelNoProba(), "test text")
        assert len(df) == 1
        assert df.iloc[0]["Probability"] == 1.0
