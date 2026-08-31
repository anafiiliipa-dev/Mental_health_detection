"""Unit tests for src/mental_health/monitoring/mock_stream.py.

Only the sampling logic is unit-tested here (pure pandas/sklearn, no
external dependency). ``drift_check.py`` itself needs Evidently + a real
model + the shared MLflow backend to run, so it is exercised manually /
by the GitHub Actions cron job rather than in the unit test suite.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mental_health.data.cleaning import TARGET_COL, TEXT_COL
from mental_health.monitoring.mock_stream import build_reference_and_holdout, sample_mock_batch


@pytest.fixture
def sample_df() -> pd.DataFrame:
    # 40 rows, 4 balanced classes — big enough for a stratified 80/20 split.
    labels = ["Anxiety", "Depression", "ADHD", "Bipolar"] * 10
    return pd.DataFrame(
        {
            TEXT_COL: [f"sample message number {i}" for i in range(len(labels))],
            TARGET_COL: labels,
        }
    )


class TestBuildReferenceAndHoldout:
    def test_returns_only_the_expected_columns(self, sample_df):
        reference_df, holdout_df = build_reference_and_holdout(sample_df)
        assert list(reference_df.columns) == [TEXT_COL, TARGET_COL]
        assert list(holdout_df.columns) == [TEXT_COL, TARGET_COL]

    def test_split_sizes_match_test_size(self, sample_df):
        reference_df, holdout_df = build_reference_and_holdout(sample_df)
        assert len(reference_df) + len(holdout_df) == len(sample_df)
        assert len(holdout_df) == round(len(sample_df) * 0.2)

    def test_reference_and_holdout_rows_are_disjoint(self, sample_df):
        reference_df, holdout_df = build_reference_and_holdout(sample_df)
        assert set(reference_df[TEXT_COL]) & set(holdout_df[TEXT_COL]) == set()

    def test_split_is_deterministic_across_calls(self, sample_df):
        reference_1, holdout_1 = build_reference_and_holdout(sample_df)
        reference_2, holdout_2 = build_reference_and_holdout(sample_df)
        assert reference_1[TEXT_COL].tolist() == reference_2[TEXT_COL].tolist()
        assert holdout_1[TEXT_COL].tolist() == holdout_2[TEXT_COL].tolist()


class TestSampleMockBatch:
    def test_returns_requested_batch_size(self, sample_df):
        _, holdout_df = build_reference_and_holdout(sample_df)
        batch = sample_mock_batch(holdout_df, n=3, random_state=0)
        assert len(batch) == 3

    def test_caps_at_pool_size_instead_of_raising(self, sample_df):
        _, holdout_df = build_reference_and_holdout(sample_df)
        batch = sample_mock_batch(holdout_df, n=1_000, random_state=0)
        assert len(batch) == len(holdout_df)

    def test_same_random_state_is_reproducible(self, sample_df):
        _, holdout_df = build_reference_and_holdout(sample_df)
        batch_1 = sample_mock_batch(holdout_df, n=3, random_state=42)
        batch_2 = sample_mock_batch(holdout_df, n=3, random_state=42)
        assert batch_1[TEXT_COL].tolist() == batch_2[TEXT_COL].tolist()
