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


def sample_mock_batch(
    holdout_df: pd.DataFrame, n: int, random_state: int | None = None, simulate_drift: bool = False
) -> pd.DataFrame:
    """
    Sample ``n`` rows from the holdout pool — one simulated batch of newly
    arrived messages.

    ``random_state=None`` (the default) draws a different sample every
    call, which is what a real weekly cron run should do. Pass a fixed
    seed in tests for reproducibility. If the pool has fewer than ``n``
    rows, the whole pool is returned instead of raising.

    ``simulate_drift=True`` deliberately returns a SKEWED batch instead of
    an honest random sample — decided with Ana: with no live production
    traffic yet, an honest holdout sample is drawn from the exact same
    distribution as training, so it essentially never trips Evidently's
    drift preset. That makes the whole detection -> alert -> retrain loop
    impossible to observe running end-to-end. This mode exists purely to
    exercise that loop on a predictable cadence (see drift_monitoring.yml,
    which now always runs with this on) -- it is NOT a realistic traffic
    simulation and must not be read as one.

    The skew hits both columns Evidently compares in build_drift_frames:
    - "prediction" (categorical drift): the batch is drawn from a single
      label only (the rarest one in the pool), instead of the pool's
      natural class mix.
    - "text_length" (numerical drift): each sampled text is duplicated
      against itself, roughly doubling its length.
    """
    if not simulate_drift:
        n = min(n, len(holdout_df))
        return holdout_df.sample(n=n, random_state=random_state).reset_index(drop=True)

    skewed_label = holdout_df[TARGET_COL].value_counts().idxmin()
    skewed_pool = holdout_df[holdout_df[TARGET_COL] == skewed_label]
    batch = skewed_pool.sample(n=n, random_state=random_state, replace=True).reset_index(drop=True)
    batch[TEXT_COL] = batch[TEXT_COL] + " " + batch[TEXT_COL]
    return batch
