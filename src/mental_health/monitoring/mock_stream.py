"""
Simulates the arrival of new incoming messages for drift monitoring.

The project has no live production traffic yet, so — per the documented
decision (architecture diagram nodes 02/15/16) — the "mock stream" is a
sample drawn from the training dataset's held-out test split: rows the
champion model never trained on, standing in for genuinely new messages
until real request logs exist to replace this.

Reproduces the exact same stratified split as ``train.py``'s
``build_splits`` (same ``TEST_SIZE``, same ``RANDOM_STATE``), so the
"reference" set here always matches what the currently-serving model
actually trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mental_health.data.cleaning import TARGET_COL, TEXT_COL
from mental_health.train.model_registry import RANDOM_STATE

TEST_SIZE = 0.2


def build_reference_and_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ``df`` exactly like ``train.py`` does, keeping only the columns
    the drift check needs (``TEXT_COL``, ``TARGET_COL``).

    Returns ``(reference_df, holdout_df)``:
    - ``reference_df``: the training rows — what the model has "seen".
    - ``holdout_df``: the untouched test rows — the sampling pool for the
      mock stream.
    """
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    reference_df = df.loc[train_idx, [TEXT_COL, TARGET_COL]].reset_index(drop=True)
    holdout_df = df.loc[test_idx, [TEXT_COL, TARGET_COL]].reset_index(drop=True)
    return reference_df, holdout_df


def sample_mock_batch(holdout_df: pd.DataFrame, n: int, random_state: int | None = None) -> pd.DataFrame:
    """
    Sample ``n`` rows from the holdout pool — one simulated batch of newly
    arrived messages.

    ``random_state=None`` (the default) draws a different sample every
    call, which is what a real weekly cron run should do. Pass a fixed
    seed in tests for reproducibility. If the pool has fewer than ``n``
    rows, the whole pool is returned instead of raising.
    """
    n = min(n, len(holdout_df))
    return holdout_df.sample(n=n, random_state=random_state).reset_index(drop=True)
