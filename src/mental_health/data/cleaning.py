"""
Data cleaning pipeline for the raw Mental Health Detection dataset.

Extracted and corrected from ``notebooks/01_data_cleaning.ipynb`` so the
cleaning logic is testable, reusable and reproducible from the command line
instead of living only inside a notebook.

Corrections applied relative to the original notebook (see
``audit-dataset-brut.md`` in the project for the full investigation):

1. ``schizophrenia`` is now correctly canonicalised to ``"Schizophrenia"``
   (title case, consistent with every other label) instead of staying
   lowercase — the original mapping silently broke case-sensitive filters
   downstream (e.g. ``CRITICAL_LABELS`` in the clinical evaluation notebook).
2. Exact duplicates (identical ``body`` + ``category``) are now removed.
   The original notebook only ever explored this in the throwaway
   ``00_exploration.ipynb`` scratchpad — it never made it into the real
   cleaning pipeline.
3. Rows where the same ``body`` text appears under more than one distinct
   ``category`` (label noise) are dropped entirely, since we have no
   reliable way to arbitrate which label is correct.
4. Near-empty texts (fewer than ``MIN_BODY_LENGTH`` characters) are
   filtered out as noise. Long texts are deliberately kept — they are
   verbose but legitimate posts, not noise.
5. The ``body_masked`` column (diagnostic/medication terms replaced with
   ``[CONDITION]``, used as the leakage-robustness text variant in
   training) is added at the end of the pipeline, via
   ``mental_health.data.masking``. Behaviourally identical to the
   notebook's masking logic — only the label-casing fix above changes
   which text variant ends up selected as champion downstream.

Near-duplicate detection (MinHash/LSH) is intentionally NOT included here.
It is real, useful work already prototyped in ``00_exploration.ipynb``, but
it adds a new dependency (``datasketch``) and non-trivial complexity — it is
left as a documented follow-up rather than ported blindly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH, RAW_DATA_PATH
from mental_health.data.masking import add_masked_column

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

TEXT_COL = "body"
TARGET_COL = "category"
MASKED_COL = "body_masked"
REQUIRED_COLUMNS = [TEXT_COL, TARGET_COL]

# Minimum text length (characters, after stripping) to keep a row.
# Anything shorter is treated as noise, not a legitimate short post.
MIN_BODY_LENGTH = 10

# Canonical label mapping. Every raw variant (case, synonym, abbreviation)
# maps to exactly one of the 7 official CLASS_LABELS, all in title case.
# NOTE: "schizophrenia" maps to "Schizophrenia" (title case) — this was the
# one entry that stayed lowercase in the original notebook.
CANONICAL_LABELS: dict[str, str] = {
    "adhd": "ADHD",
    "add": "ADHD",
    "anxiety": "Anxiety",
    "autism": "Autism",
    "asd": "Autism",
    "autistic": "Autism",
    "bipolar": "Bipolar",
    "bpd": "BPD",
    "borderline": "BPD",
    "depression": "Depression",
    "depressed": "Depression",
    "schizophrenia": "Schizophrenia",
    "schizo": "Schizophrenia",
    "schizoaffective": "Schizophrenia",
}


# ============================================================
# Loading & schema validation
# ============================================================

def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw dataset and validate that the required columns exist.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``body`` or ``category`` is missing from the columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw dataset: {missing_cols}")

    logger.info("Loaded raw dataset: %d rows from %s", len(df), path)
    return df


# ============================================================
# Individual cleaning steps
# ============================================================

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw category values to the canonical label set, dropping unmapped rows."""
    before = len(df)

    df = df.copy()
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CANONICAL_LABELS)
    )
    df = df.dropna(subset=[TARGET_COL]).copy()

    removed = before - len(df)
    logger.info("normalize_labels: removed %d unmapped/empty-label rows (%d -> %d)", removed, before, len(df))
    return df


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing text or label."""
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    removed = before - len(df)
    logger.info("drop_missing_values: removed %d rows (%d -> %d)", removed, before, len(df))
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact duplicates of (body, category)."""
    before = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS).copy()
    removed = before - len(df)
    logger.info("drop_exact_duplicates: removed %d rows (%d -> %d)", removed, before, len(df))
    return df


def drop_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop every row whose text is associated with more than one distinct label.

    We have no reliable signal for arbitrating which label is correct in
    these cases, so both (or all) conflicting occurrences are removed
    rather than guessed at.
    """
    before = len(df)

    label_counts = df.groupby(TEXT_COL)[TARGET_COL].nunique()
    conflicting_bodies = label_counts[label_counts > 1].index

    df = df[~df[TEXT_COL].isin(conflicting_bodies)].copy()

    removed = before - len(df)
    logger.info(
        "drop_label_conflicts: removed %d rows across %d conflicting texts (%d -> %d)",
        removed, len(conflicting_bodies), before, len(df),
    )
    return df


def filter_short_texts(df: pd.DataFrame, min_length: int = MIN_BODY_LENGTH) -> pd.DataFrame:
    """Drop rows whose (stripped) text is shorter than ``min_length`` characters."""
    before = len(df)
    text_len = df[TEXT_COL].astype(str).str.strip().str.len()
    df = df[text_len >= min_length].copy()
    removed = before - len(df)
    logger.info("filter_short_texts: removed %d rows shorter than %d chars (%d -> %d)", removed, min_length, before, len(df))
    return df


# ============================================================
# Orchestrator
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in order and return the cleaned dataframe."""
    logger.info("Starting cleaning pipeline: %d raw rows", len(df))

    df = normalize_labels(df)
    df = drop_missing_values(df)
    df = drop_exact_duplicates(df)
    df = drop_label_conflicts(df)
    df = filter_short_texts(df)
    df = add_masked_column(df, text_col=TEXT_COL, masked_col=MASKED_COL)

    df = df.reset_index(drop=True)
    logger.info("Cleaning pipeline complete: %d rows remaining", len(df))
    return df


def run(input_path: Path = RAW_DATA_PATH, output_path: Path = DEFAULT_CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load, clean and export the dataset. Returns the cleaned dataframe."""
    df_raw = load_raw_dataset(input_path)
    df_clean = clean_dataset(df_raw)

    # Fixed column order (body, body_masked, category), matching the
    # original notebook's export format.
    df_clean = df_clean[[TEXT_COL, MASKED_COL, TARGET_COL]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows)", output_path, len(df_clean))
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
