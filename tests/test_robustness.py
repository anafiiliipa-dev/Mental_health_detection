"""Unit tests for src/mental_health/train/robustness.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train.robustness import (
    DEFAULT_PERTURBATIONS,
    evaluate_robustness,
    inject_typos,
    perturb_series,
    randomize_casing,
    summarize_worst_case,
)

# ============================================================
# inject_typos
# ============================================================


class TestInjectTypos:
    def test_zero_rate_leaves_text_unchanged(self):
        text = "I feel anxious all the time"
        assert inject_typos(text, rate=0.0, seed=1) == text

    def test_empty_text_returns_empty(self):
        assert inject_typos("", rate=0.5, seed=1) == ""

    def test_is_deterministic_for_a_given_seed(self):
        text = "racing thoughts and no sleep for days"
        assert inject_typos(text, rate=0.3, seed=42) == inject_typos(text, rate=0.3, seed=42)

    def test_different_seeds_can_produce_different_output(self):
        text = "racing thoughts and no sleep for many many days in a row"
        outputs = {inject_typos(text, rate=0.5, seed=s) for s in range(10)}
        assert len(outputs) > 1

    def test_high_rate_changes_the_text(self):
        text = "manic episodes followed by crushing lows every week"
        assert inject_typos(text, rate=1.0, seed=1) != text


# ============================================================
# randomize_casing
# ============================================================


class TestRandomizeCasing:
    def test_upper_mode(self):
        assert randomize_casing("Hello World", mode="upper") == "HELLO WORLD"

    def test_lower_mode(self):
        assert randomize_casing("Hello World", mode="lower") == "hello world"

    def test_random_mode_preserves_letters_ignoring_case(self):
        text = "Hello World"
        result = randomize_casing(text, mode="random", seed=1)
        assert result.lower() == text.lower()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            randomize_casing("text", mode="sideways")

    def test_empty_text_returns_empty(self):
        assert randomize_casing("", mode="upper") == ""


# ============================================================
# perturb_series
# ============================================================


class TestPerturbSeries:
    def test_returns_same_length_series(self):
        texts = pd.Series(["a b c", "d e f", "g h i"])
        result = perturb_series(texts, lambda t, seed=None: t.upper())
        assert len(result) == len(texts)

    def test_is_deterministic_for_the_same_random_state(self):
        texts = pd.Series(["I feel anxious", "racing thoughts", "hearing voices"])
        a = perturb_series(texts, inject_typos, random_state=7)
        b = perturb_series(texts, inject_typos, random_state=7)
        assert list(a) == list(b)


# ============================================================
# evaluate_robustness
# ============================================================


class _StubModel:
    """Predicts based on whether the (possibly-corrupted) text still contains its trigger word."""

    def predict(self, X):
        return [
            "Anxiety" if "anxious" in text.lower() else "Depression"
            for text in X
        ]


class TestEvaluateRobustness:
    def _fixture(self):
        X_test = pd.Series(["I feel anxious", "I feel anxious", "nothing brings me joy", "nothing brings me joy"])
        y_test = pd.Series(["Anxiety", "Anxiety", "Depression", "Depression"])
        return X_test, y_test

    def test_includes_a_clean_baseline_row_with_zero_deltas(self):
        X_test, y_test = self._fixture()
        report = evaluate_robustness(_StubModel(), X_test, y_test, perturbations={})

        assert list(report["perturbation"]) == ["clean"]
        row = report.iloc[0]
        assert row["delta_f1_macro"] == pytest.approx(0.0)
        assert row["delta_recall_macro"] == pytest.approx(0.0)
        assert row["delta_critical_recall"] == pytest.approx(0.0)

    def test_perfect_clean_predictions_score_one(self):
        X_test, y_test = self._fixture()
        report = evaluate_robustness(_StubModel(), X_test, y_test, perturbations={})
        assert report.iloc[0]["f1_macro"] == pytest.approx(1.0)

    def test_default_perturbations_all_appear_as_rows(self):
        X_test, y_test = self._fixture()
        report = evaluate_robustness(_StubModel(), X_test, y_test)
        assert set(report["perturbation"]) == {"clean", *DEFAULT_PERTURBATIONS.keys()}

    def test_uppercase_perturbation_breaks_case_sensitive_stub(self):
        # The stub model's trigger-word matching is lowercased, so this
        # specific perturbation should NOT break it -- but a heavy typo
        # perturbation, which deletes/substitutes letters in "anxious",
        # should. This exercises that different perturbations genuinely
        # produce different degradation, not just a constant delta.
        X_test, y_test = self._fixture()
        report = evaluate_robustness(_StubModel(), X_test, y_test)
        uppercase_row = report[report["perturbation"] == "uppercase"].iloc[0]
        assert uppercase_row["f1_macro"] == pytest.approx(1.0)


class TestSummarizeWorstCase:
    def test_picks_the_most_negative_delta(self):
        report = pd.DataFrame([
            {"perturbation": "clean", "f1_macro": 1.0, "recall_macro": 1.0, "critical_recall": 1.0,
             "delta_f1_macro": 0.0, "delta_recall_macro": 0.0, "delta_critical_recall": 0.0},
            {"perturbation": "typos_light", "f1_macro": 0.9, "recall_macro": 0.9, "critical_recall": 0.9,
             "delta_f1_macro": -0.1, "delta_recall_macro": -0.1, "delta_critical_recall": -0.1},
            {"perturbation": "typos_heavy", "f1_macro": 0.6, "recall_macro": 0.6, "critical_recall": 0.5,
             "delta_f1_macro": -0.4, "delta_recall_macro": -0.4, "delta_critical_recall": -0.5},
        ])
        summary = summarize_worst_case(report)
        assert summary["worst_perturbation"] == "typos_heavy"
        assert summary["worst_delta_f1_macro"] == pytest.approx(-0.4)
        assert summary["worst_delta_critical_recall"] == pytest.approx(-0.5)

    def test_handles_only_a_clean_row(self):
        report = pd.DataFrame([
            {"perturbation": "clean", "f1_macro": 1.0, "recall_macro": 1.0, "critical_recall": 1.0,
             "delta_f1_macro": 0.0, "delta_recall_macro": 0.0, "delta_critical_recall": 0.0},
        ])
        summary = summarize_worst_case(report)
        assert summary["worst_perturbation"] is None
