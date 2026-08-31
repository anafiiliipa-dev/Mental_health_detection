"""
Extra evaluation metrics and statistical rigor (Phase 11, first slice).

Layered on top of the existing champion evaluation
(``champion.evaluate_final_model``) without touching its established
headline metrics (``f1_macro``, ``recall_macro``, ``critical_recall``) or
the promotion gate in ``promote.py``, which still only looks at those
three -- this module adds evaluation rigor, it does not change any
automated decision already in production.

- **MCC** (Matthews Correlation Coefficient): robust to class imbalance,
  needs only predicted labels -- no calibration required.
- **PR-AUC per class**: more informative than ROC-AUC on the rare,
  clinically critical classes (Bipolar, Schizophrenia). Computed from raw
  decision/probability scores (works for LinearSVC via
  ``decision_function``, which has no ``predict_proba``) -- this is a
  ranking metric, so it does NOT require the model to be calibrated.
- **Paired bootstrap significance test**: resamples the held-out test set
  (same rows for both models) to estimate whether an observed metric gap
  between two models is likely real or just noise -- answers "is this
  actually a better model, or did it just get a lucky split?".

Calibration itself (Platt scaling / ``CalibratedClassifierCV``) and the
metrics that depend on it (Brier score, Expected Calibration Error) are a
separate, later slice of Phase 11 -- computing them against an
uncalibrated decision_function softmax (as the API's confidence score
already does, informally) would be misleading, so they are deliberately
NOT included here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize


def compute_mcc(y_true, y_pred) -> float:
    """Matthews Correlation Coefficient -- robust to class imbalance, unlike accuracy or F1 alone."""
    return float(matthews_corrcoef(y_true, y_pred))


def get_ranking_scores(model, X):  # noqa: N803
    """
    Best-effort per-class ranking scores for PR-AUC: ``predict_proba`` if
    the model has it (LogisticRegression, MultinomialNB), else
    ``decision_function`` (LinearSVC). Returns ``None`` if neither exists,
    so callers can skip PR-AUC gracefully instead of raising.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_pr_auc_per_class(y_true, scores: np.ndarray, labels: list[str]) -> dict[str, float]:
    """
    Average precision (area under the precision-recall curve) per class,
    from ``get_ranking_scores`` output. A ranking metric -- valid on raw,
    uncalibrated decision_function scores, unlike Brier score/ECE.

    Handles the binary case explicitly: ``label_binarize`` collapses two
    classes into a single column (the positive class = ``labels[1]``),
    unlike the one-column-per-class shape it returns for 3+ classes.
    """
    y_true_binarized = label_binarize(y_true, classes=labels)
    scores = np.asarray(scores)

    if len(labels) == 2:
        positive_scores = scores[:, 1] if scores.ndim == 2 else scores
        positive_ap = float(average_precision_score(y_true_binarized[:, 0], positive_scores))
        negative_ap = float(average_precision_score(1 - y_true_binarized[:, 0], -positive_scores))
        return {labels[0]: negative_ap, labels[1]: positive_ap}

    return {
        label: float(average_precision_score(y_true_binarized[:, i], scores[:, i]))
        for i, label in enumerate(labels)
    }


def _default_metric(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def paired_bootstrap_test(
    y_true,
    y_pred_a,
    y_pred_b,
    metric_fn=_default_metric,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Paired bootstrap significance test on the SAME held-out rows for two
    models (A, B). Resamples row indices with replacement ``n_bootstrap``
    times, recomputes ``metric_fn`` for both models on each resample, and
    reports a 95% CI + two-sided p-value on the observed difference
    (A - B) -- i.e. whether an observed metric gap is likely real or just
    noise from the particular test split.

    ``y_true``, ``y_pred_a``, ``y_pred_b`` must already be aligned on the
    same rows (e.g. champion vs. runner-up, both evaluated on the same
    X_test).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    if not (len(y_pred_a) == n and len(y_pred_b) == n):
        raise ValueError("y_true, y_pred_a and y_pred_b must have the same length (paired rows).")

    observed_diff = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)

    rng = np.random.RandomState(random_state)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])

    ci_lower, ci_upper = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))

    # Two-sided bootstrap p-value: twice the share of resamples that land
    # on (or past) the opposite side of zero from the observed difference.
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }
