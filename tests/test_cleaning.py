"""Unit tests for the data cleaning pipeline (src/mental_health/data/cleaning.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.data.cleaning import (
    CANONICAL_LABELS,
    clean_dataset,
    drop_exact_duplicates,
    drop_label_conflicts,
    drop_missing_values,
    filter_short_texts,
    normalize_labels,
)

# ============================================================
# normalize_labels
# ============================================================

class TestNormalizeLabels:
    def test_schizophrenia_is_capitalized(self):
        # Regression test for the exact bug found in the audit: the original
        # notebook mapped "schizophrenia" -> "schizophrenia" (lowercase).
        df = pd.DataFrame({"body": ["I hear voices"], "category": ["schizophrenia"]})
        out = normalize_labels(df)
        assert out["category"].iloc[0] == "Schizophrenia"

    def test_known_synonyms_map_to_canonical_label(self):
        df = pd.DataFrame(
            {
                "body": ["a", "b", "c"],
                "category": ["ASD", "borderline", "depressed"],
            }
        )
        out = normalize_labels(df)
        assert list(out["category"]) == ["Autism", "BPD", "Depression"]

    def test_case_and_whitespace_are_normalized_before_mapping(self):
        df = pd.DataFrame({"body": ["a"], "category": ["  ADHD  ".lower()]})
        out = normalize_labels(df)
        assert out["category"].iloc[0] == "ADHD"

    def test_unmapped_labels_are_dropped(self):
        df = pd.DataFrame(
            {"body": ["a", "b"], "category": ["adhd", "not_a_real_label"]}
        )
        out = normalize_labels(df)
        assert len(out) == 1
        assert out["category"].iloc[0] == "ADHD"

    def test_every_canonical_value_is_properly_capitalized(self):
        # Guards against a future contributor re-introducing a lowercase
        # entry like the original "schizophrenia" bug: every canonical
        # label must be either a title-cased word or a known acronym.
        acronyms = {"ADHD", "BPD"}
        for canonical in set(CANONICAL_LABELS.values()):
            assert canonical in acronyms or canonical.istitle(), (
                f"{canonical!r} is not title-cased and not a known acronym"
            )


# ============================================================
# drop_missing_values
# ============================================================

class TestDropMissingValues:
    def test_removes_rows_with_missing_body_or_category(self):
        df = pd.DataFrame(
            {
                "body": ["text one", None, "text three"],
                "category": ["ADHD", "Anxiety", None],
            }
        )
        out = drop_missing_values(df)
        assert len(out) == 1
        assert out["body"].iloc[0] == "text one"

    def test_keeps_complete_rows_untouched(self):
        df = pd.DataFrame({"body": ["a", "b"], "category": ["ADHD", "Anxiety"]})
        out = drop_missing_values(df)
        assert len(out) == 2


# ============================================================
# drop_exact_duplicates
# ============================================================

class TestDropExactDuplicates:
    def test_removes_identical_body_and_category_pairs(self):
        df = pd.DataFrame(
            {
                "body": ["same text", "same text", "different text"],
                "category": ["ADHD", "ADHD", "ADHD"],
            }
        )
        out = drop_exact_duplicates(df)
        assert len(out) == 2

    def test_same_body_different_category_is_not_removed_here(self):
        # drop_exact_duplicates only removes (body, category) pairs — a
        # same-body-different-label conflict is handled separately by
        # drop_label_conflicts.
        df = pd.DataFrame(
            {"body": ["same text", "same text"], "category": ["ADHD", "Anxiety"]}
        )
        out = drop_exact_duplicates(df)
        assert len(out) == 2


# ============================================================
# drop_label_conflicts
# ============================================================

class TestDropLabelConflicts:
    def test_removes_all_occurrences_of_a_conflicting_text(self):
        df = pd.DataFrame(
            {
                "body": ["conflicted text", "conflicted text", "clean text"],
                "category": ["ADHD", "Anxiety", "Depression"],
            }
        )
        out = drop_label_conflicts(df)
        assert len(out) == 1
        assert out["body"].iloc[0] == "clean text"

    def test_no_conflicts_leaves_dataframe_unchanged(self):
        df = pd.DataFrame(
            {"body": ["text a", "text b"], "category": ["ADHD", "Anxiety"]}
        )
        out = drop_label_conflicts(df)
        assert len(out) == 2


# ============================================================
# filter_short_texts
# ============================================================

class TestFilterShortTexts:
    def test_removes_texts_shorter_than_min_length(self):
        df = pd.DataFrame(
            {"body": ["ok", "this is long enough"], "category": ["ADHD", "Anxiety"]}
        )
        out = filter_short_texts(df, min_length=10)
        assert len(out) == 1
        assert out["body"].iloc[0] == "this is long enough"

    def test_keeps_very_long_texts(self):
        long_text = "word " * 500
        df = pd.DataFrame({"body": [long_text], "category": ["ADHD"]})
        out = filter_short_texts(df, min_length=10)
        assert len(out) == 1

    def test_strips_whitespace_before_measuring_length(self):
        df = pd.DataFrame({"body": ["   short   "], "category": ["ADHD"]})
        out = filter_short_texts(df, min_length=10)
        assert len(out) == 0


# ============================================================
# clean_dataset (full pipeline)
# ============================================================

class TestCleanDataset:
    def test_full_pipeline_on_mixed_dirty_data(self):
        df = pd.DataFrame(
            {
                "body": [
                    "I hear voices watching me all the time",  # kept
                    "I hear voices watching me all the time",  # exact dup -> removed
                    None,  # missing body -> removed
                    "too short",  # 9 chars, below MIN_BODY_LENGTH -> removed
                    "hi",  # short -> removed
                    "same body different label one",  # conflict -> removed
                    "same body different label one",  # conflict -> removed
                ],
                "category": [
                    "schizophrenia",
                    "schizophrenia",
                    "adhd",
                    "anxiety",
                    "anxiety",
                    "adhd",
                    "bipolar",
                ],
            }
        )
        out = clean_dataset(df)

        assert "Schizophrenia" in out["category"].values
        assert len(out) == out["body"].nunique(), "no duplicate bodies should remain"
        assert "same body different label one" not in out["body"].values
        assert "hi" not in out["body"].values

    def test_result_has_no_missing_values(self):
        df = pd.DataFrame(
            {"body": ["valid text here", None], "category": ["adhd", "adhd"]}
        )
        out = clean_dataset(df)
        assert out.isnull().sum().sum() == 0

    def test_result_has_masked_column(self):
        df = pd.DataFrame(
            {"body": ["I take lithium every day for my mood"], "category": ["bipolar"]}
        )
        out = clean_dataset(df)
        assert "body_masked" in out.columns
        assert "lithium" not in out["body_masked"].iloc[0].lower()

    def test_result_index_is_reset(self):
        df = pd.DataFrame(
            {
                "body": ["text one here", "text two here", "text three here"],
                "category": ["adhd", "anxiety", "bipolar"],
            }
        )
        out = clean_dataset(df)
        assert list(out.index) == list(range(len(out)))
