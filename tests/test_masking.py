"""Unit tests for src/mental_health/data/masking.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.data.masking import (
    MASK_TOKEN,
    MASKING_LEAK_LIST,
    add_masked_column,
    build_leakage_pattern,
    mask_leakage_terms,
)

# ============================================================
# mask_leakage_terms
# ============================================================

class TestMaskLeakageTerms:
    def test_masks_a_diagnostic_term(self):
        result = mask_leakage_terms("I was diagnosed with bipolar last year")
        assert "bipolar" not in result.lower()
        assert MASK_TOKEN in result

    def test_masks_a_medication_term(self):
        result = mask_leakage_terms("I take lithium every morning")
        assert "lithium" not in result.lower()
        assert MASK_TOKEN in result

    def test_is_case_insensitive(self):
        result = mask_leakage_terms("My Schizophrenia diagnosis was hard")
        assert "schizophrenia" not in result.lower()

    def test_masks_multiple_occurrences(self):
        result = mask_leakage_terms("anxiety anxiety anxiety")
        assert result.count(MASK_TOKEN) == 3

    def test_does_not_mask_partial_word_matches(self):
        # "adhd" must not match inside an unrelated longer word.
        result = mask_leakage_terms("adhderall is not a real word")
        assert MASK_TOKEN not in result

    def test_add_is_excluded_from_masking(self):
        # "add" is deliberately excluded (see MASKING_LEAK_LIST) because
        # it's a common English word, not just an ADHD abbreviation.
        result = mask_leakage_terms("please add this to the list")
        assert result == "please add this to the list"

    def test_none_input_returns_none(self):
        assert mask_leakage_terms(None) is None

    def test_nan_input_returns_none(self):
        assert mask_leakage_terms(float("nan")) is None

    def test_text_without_leak_terms_is_unchanged(self):
        text = "I feel a bit off today but can't explain why"
        assert mask_leakage_terms(text) == text


# ============================================================
# build_leakage_pattern
# ============================================================

class TestBuildLeakagePattern:
    def test_pattern_matches_known_terms(self):
        import re

        pattern = build_leakage_pattern(["bipolar", "lithium"])
        assert re.search(pattern, "I have bipolar disorder", re.IGNORECASE)
        assert not re.search(pattern, "no relevant terms here", re.IGNORECASE)


# ============================================================
# add_masked_column
# ============================================================

class TestAddMaskedColumn:
    def test_adds_the_masked_column(self):
        df = pd.DataFrame({"body": ["I have anxiety every day"], "category": ["Anxiety"]})
        out = add_masked_column(df)
        assert "body_masked" in out.columns
        assert "anxiety" not in out["body_masked"].iloc[0].lower()

    def test_does_not_mutate_original_text_column(self):
        df = pd.DataFrame({"body": ["I have anxiety every day"], "category": ["Anxiety"]})
        out = add_masked_column(df)
        assert out["body"].iloc[0] == "I have anxiety every day"

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({"body": ["I have anxiety"], "category": ["Anxiety"]})
        add_masked_column(df)
        assert "body_masked" not in df.columns


def test_add_is_not_in_the_masking_list():
    # Guards the deliberate exclusion documented in masking.py.
    assert "add" not in MASKING_LEAK_LIST
