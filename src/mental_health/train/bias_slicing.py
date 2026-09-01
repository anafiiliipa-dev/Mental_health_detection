"""
Bias slicing: evaluation by subgroup (Phase 11, remaining slice:
"évaluation par sous-groupes (bias slicing)").

This dataset (``cleaning.py``: ``body`` + ``category``, scraped from public
mental-health forums) carries no demographic attributes (age, gender,
locale, ...) to slice on — there is nothing to fabricate here, and
inventing a proxy for a protected attribute the data doesn't actually
contain would be worse than not slicing at all. Slicing is instead done on
attributes the data genuinely has and that are known failure modes for
text classifiers:

- **per-class (label) performance**: is the model quietly failing on one
  or more of the seven diagnosis categories while headline macro metrics
  look fine? Clinically the most important slice, since the two critical
  labels (Bipolar, Schizophrenia) are also among the rarer ones.
- **text length**: short posts carry less signal than long ones — a model
  that only works on verbose posts is a real, previously undocumented
  failure mode for a triage tool meant to work on whatever a user actually
  types.

Like ``robustness.py`` and the rest of Phase 11's evaluation additions,
this is reporting/diagnostic only — it does not feed back into champion
selection or the promotion gate.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score, recall_score

from mental_health.config.paths import CRITICAL_LABELS

# ============================================================
# Slice assignment
# ============================================================

DEFAULT_LENGTH_BIN_LABELS = ["short", "medium", "long"]


def assign_length_slices(texts: pd.Series, n_bins: int = 3, labels: list[str] | None = None) -> pd.Series:
    """
    Bucket each text into an equal-frequency (quantile) length slice by
    word count — e.g. terciles ``["short", "medium", "long"]`` by default.
    Quantile (not equal-width) binning is used so each slice has a
    comparable sample size, which is what makes a per-slice recall
    comparison meaningful rather than noisy on a near-empty bin.
    """
    labels = labels if labels is not None else DEFAULT_LENGTH_BIN_LABELS[:n_bins]
    if len(labels) != n_bins:
        raise ValueError(f"Expected {n_bins} labels, got {len(labels)}: {labels}")

    word_counts = pd.Series(texts).reset_index(drop=True).map(lambda t: len(str(t).split()))
    try:
        return pd.qcut(word_counts, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        # Degenerate input (e.g. every text the same length, or too few
        # rows) — qcut can't form n_bins distinct edges. Fall back to a
        # single slice rather than raising, since this is a diagnostic
        # tool, not a training-time step that must be strict.
        return pd.Series([labels[0]] * len(word_counts))


# ============================================================
# Per-slice evaluation
# ============================================================


def _slice_metrics(y_true, y_pred) -> dict:
    # IMPORTANT: labels is pinned to the classes actually present in this
    # slice's y_true, not left to sklearn's default (the union of y_true
    # and y_pred). Left at the default, a class slice -- where y_true is a
    # SINGLE label by construction -- would macro-average over every other
    # label the model happened to (wrongly) predict in that slice too,
    # each scored 0 recall by zero_division since it has no true instances
    # here. That silently drags every slice's score down to near-noise
    # regardless of how the model actually does on its true label -- not a
    # subgroup performance measurement at all, just an artifact of how
    # wrong the model's off-label guesses were.
    labels = sorted(pd.Series(y_true).unique())
    return {
        "support": len(y_true),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def evaluate_slices(y_true, y_pred, slice_labels, slice_column_name: str = "slice") -> pd.DataFrame:
    """
    Generic per-slice evaluation: groups ``(y_true, y_pred)`` by
    ``slice_labels`` (any grouping key of the same length — a class label,
    a length bucket, a text variant, ...) and reports per-slice support,
    f1_macro and recall_macro, sorted worst-f1_macro-first so the weakest
    subgroup is immediately visible.
    """
    df = pd.DataFrame({
        slice_column_name: pd.Series(slice_labels).reset_index(drop=True),
        "y_true": pd.Series(y_true).reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
    })

    rows = [
        {slice_column_name: slice_value, **_slice_metrics(group["y_true"], group["y_pred"])}
        for slice_value, group in df.groupby(slice_column_name, observed=True)
    ]
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=True).reset_index(drop=True)


def evaluate_class_slices(y_true, y_pred) -> pd.DataFrame:
    """
    Per-label (diagnosis category) recall — the most direct bias-slicing
    view for this project: is any single class, critical or not, being
    under-served relative to the others? Flags the clinically critical
    labels (``CRITICAL_LABELS``) explicitly so a reviewer doesn't have to
    cross-reference a separate list.
    """
    report = evaluate_slices(y_true, y_pred, y_true, slice_column_name="label")
    report = report.rename(columns={"f1_macro": "f1", "recall_macro": "recall"})
    report["is_critical"] = report["label"].isin(CRITICAL_LABELS)
    return report[["label", "is_critical", "support", "recall", "f1"]]


def evaluate_length_slices(texts, y_true, y_pred, n_bins: int = 3) -> pd.DataFrame:
    """Per length-bucket (short/medium/long, by word count) macro metrics."""
    length_slices = assign_length_slices(texts, n_bins=n_bins)
    return evaluate_slices(y_true, y_pred, length_slices, slice_column_name="length_bucket")


def summarize_fairness_gap(slice_report: pd.DataFrame, metric_col: str = "recall") -> dict:
    """
    The headline fairness number for a slice report: the gap between the
    best- and worst-performing slice on ``metric_col``, plus which slice
    is worst — this is the number that should get attention even when the
    overall macro metric looks fine, since a large gap can hide behind a
    healthy average.
    """
    if slice_report.empty or metric_col not in slice_report.columns:
        return {"gap": 0.0, "worst_slice": None, "best_slice": None}

    worst_row = slice_report.loc[slice_report[metric_col].idxmin()]
    best_row = slice_report.loc[slice_report[metric_col].idxmax()]
    slice_col = next(c for c in slice_report.columns if c not in {"support", "recall", "f1", "is_critical"})

    return {
        "gap": float(best_row[metric_col] - worst_row[metric_col]),
        "worst_slice": worst_row[slice_col],
        "worst_value": float(worst_row[metric_col]),
        "best_slice": best_row[slice_col],
        "best_value": float(best_row[metric_col]),
    }


def run() -> dict[str, pd.DataFrame]:
    """
    Load the current champion (registered "production" alias, falling
    back to "staging"), rebuild its test split, and write both the
    per-class and per-length-bucket slice reports to
    ``paths.BIAS_SLICING_REPORT_PATH`` (concatenated, with a ``slice_type``
    column distinguishing the two).
    """
    import logging

    import mlflow
    from dotenv import load_dotenv

    load_dotenv()

    from mental_health.config.mlflow_config import (
        MLFLOW_REGISTERED_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        PRODUCTION_ALIAS,
        STAGING_ALIAS,
    )
    from mental_health.config.paths import BIAS_SLICING_REPORT_PATH, DEFAULT_CLEAN_DATA_PATH
    from mental_health.train.train import build_splits

    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        version = client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        alias_used = PRODUCTION_ALIAS
    except mlflow.exceptions.MlflowException:
        version = client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, STAGING_ALIAS)
        alias_used = STAGING_ALIAS

    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{alias_used}")
    logger.info("Loaded '%s' v%s (alias=%s) for bias slicing", MLFLOW_REGISTERED_MODEL_NAME, version.version, alias_used)

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    splits = build_splits(df)
    X_test, y_test = splits["raw"]["X_test"], splits["raw"]["y_test"]
    y_pred = model.predict(X_test)

    class_report = evaluate_class_slices(y_test, y_pred)
    length_report = evaluate_length_slices(X_test, y_test, y_pred)

    class_gap = summarize_fairness_gap(class_report.rename(columns={"label": "slice"}))
    length_gap = summarize_fairness_gap(length_report.rename(columns={"length_bucket": "slice", "recall_macro": "recall"}))
    logger.info("Fairness gap by class: %s", class_gap)
    logger.info("Fairness gap by text length: %s", length_gap)

    class_report.insert(0, "slice_type", "class")
    length_report.insert(0, "slice_type", "length")
    combined = pd.concat([class_report, length_report], ignore_index=True)
    BIAS_SLICING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BIAS_SLICING_REPORT_PATH, index=False)
    logger.info("Bias slicing report written to %s:\n%s", BIAS_SLICING_REPORT_PATH, combined.to_string(index=False))

    return {"class_slices": class_report, "length_slices": length_report}


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
