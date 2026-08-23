"""
Diagnostic/medication term masking — produces the ``body_masked`` column.

Ported from ``notebooks/01_data_cleaning.ipynb`` (leakage audit section).
The rationale: a model that can just read the diagnosis or medication name
out of the text ("I take Lithium for my Bipolar") isn't learning a real
linguistic signal, it's reading the answer. Masking these terms with a
neutral placeholder gives a text variant that measures whether the model
still performs well without that shortcut — used as a robustness check
alongside the raw text (see ``mental_health.train.champion`` for how the
"raw" vs "masked" comparison feeds into champion selection).

No behavioural change from the notebook — the term lists and regex logic
are ported as-is, only reorganised into pure, testable functions.
"""
from __future__ import annotations

import re

import pandas as pd

MEDICATION_TERMS: list[str] = [
    "ritalin", "concerta", "vyvanse", "adderall", "strattera",
    "lithium", "lamotrigine", "lamictal", "depakote", "valproate",
    "lexapro", "zoloft", "prozac", "sertraline", "escitalopram", "fluoxetine",
    "xanax", "alprazolam", "valium", "diazepam", "ativan", "lorazepam",
    "clozapine", "clozaril", "risperidone", "risperdal", "olanzapine",
    "zyprexa", "quetiapine", "seroquel", "haldol",
]

DIAGNOSTIC_TERMS: list[str] = [
    "adhd", "add", "asd", "autism", "autistic", "bpd", "borderline",
    "bipolar", "schizophrenia", "schizoaffective", "schizo",
    "depression", "depressed", "anxiety", "anxious",
]

FULL_LEAK_LIST: list[str] = DIAGNOSTIC_TERMS + MEDICATION_TERMS

# "add" is excluded from the masking list specifically (kept in
# FULL_LEAK_LIST above only for the leakage *audit*, not for masking):
# it's a common English word ("add up", "in addition") and masking every
# occurrence would destroy unrelated, legitimate text.
MASKING_LEAK_LIST: list[str] = [term for term in FULL_LEAK_LIST if term.lower() != "add"]

MASK_TOKEN = "[CONDITION]"


def build_leakage_pattern(term_list: list[str]) -> str:
    """Build a whole-word, case-insensitive alternation regex for the given terms."""
    escaped_terms = [re.escape(term) for term in term_list]
    return rf"(?<!\w)({'|'.join(escaped_terms)})(?!\w)"


LEAKAGE_PATTERN = build_leakage_pattern(MASKING_LEAK_LIST)


def mask_leakage_terms(text: str | float | None) -> str | None:
    """Replace every diagnostic/medication term in ``text`` with ``[CONDITION]``."""
    if pd.isna(text):
        return None
    return re.sub(LEAKAGE_PATTERN, MASK_TOKEN, str(text), flags=re.IGNORECASE)


def add_masked_column(df: pd.DataFrame, text_col: str = "body", masked_col: str = "body_masked") -> pd.DataFrame:
    """Add ``masked_col`` to ``df`` by masking diagnostic/medication terms in ``text_col``."""
    df = df.copy()
    df[masked_col] = df[text_col].apply(mask_leakage_terms)
    return df
