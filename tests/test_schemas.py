"""Unit tests for src/mental_health/api/schemas.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.api.schemas import (
    MAX_TEXT_LENGTH,
    VALID_LABELS,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


class TestPredictRequest:
    def test_accepts_normal_text(self):
        req = PredictRequest(text="I've been feeling anxious lately.")
        assert req.text == "I've been feeling anxious lately."

    def test_strips_surrounding_whitespace(self):
        req = PredictRequest(text="   hello   ")
        assert req.text == "hello"

    def test_rejects_empty_text(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_rejects_whitespace_only_text(self):
        # Stripped to "" by str_strip_whitespace, then fails min_length.
        with pytest.raises(ValidationError):
            PredictRequest(text="    ")

    def test_rejects_text_over_max_length(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="a" * (MAX_TEXT_LENGTH + 1))

    def test_accepts_text_at_max_length(self):
        req = PredictRequest(text="a" * MAX_TEXT_LENGTH)
        assert len(req.text) == MAX_TEXT_LENGTH


class TestPredictResponse:
    def test_builds_with_required_fields_only(self):
        resp = PredictResponse(label="Anxiety", confidence=0.82, is_demo_fallback=False)
        assert resp.probabilities is None

    def test_confidence_must_be_within_zero_one(self):
        with pytest.raises(ValidationError):
            PredictResponse(label="Anxiety", confidence=1.5, is_demo_fallback=False)

    def test_response_never_carries_a_text_field(self):
        # Guards the privacy requirement: the schema itself has no way to echo input text.
        assert "text" not in PredictResponse.model_fields


class TestModelInfoResponse:
    def test_degraded_mode_shape(self):
        info = ModelInfoResponse(
            registered_model_name="mental_health_classifier",
            model_available=False,
            error="no version aliased 'production'",
        )
        assert info.version is None
        assert info.metrics == {}


class TestHealthResponse:
    def test_default_status_is_ok(self):
        assert HealthResponse().status == "ok"


def test_valid_labels_matches_domain_class_labels():
    assert VALID_LABELS == ["ADHD", "Anxiety", "Autism", "Bipolar", "BPD", "Depression", "Schizophrenia"]
