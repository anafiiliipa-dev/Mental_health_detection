"""
Robustness tests under text perturbation (Phase 11, remaining slice:
"tests de robustesse (fautes de frappe/casse)").

Evaluates an already-trained model (typically the registered champion) on
deliberately corrupted copies of the held-out test set — typos and casing
changes a real user is likely to actually type — and reports how much each
headline metric degrades relative to the clean baseline. This is
reporting/diagnostic only, exactly like ``evaluation_metrics.py``'s MCC/
PR-AUC and the paired bootstrap test: it does NOT feed back into champion
selection (``benchmark.py``/``champion.py``) or the promotion gate
(``promote.py``), which are both unchanged.

Perturbations are applied to already-cleaned text (``cleaning.py`` has
already run), simulating noisy INPUT at inference time — not noisy
training data — since ``build_model_registry``'s TF-IDF/embedding
pipelines are fit once on clean text and never retrained here; only
``X_test`` is corrupted before calling ``model.predict``.
"""
from __future__ import annotations

import random
import string
from functools import partial

import pandas as pd
from sklearn.metrics import f1_score, recall_score

from mental_health.train.benchmark import critical_recall_score

# ============================================================
# Perturbations
# ============================================================


def inject_typos(text: str, rate: float = 0.1, seed: int | None = None) -> str:
    """
    Corrupt ``text`` character-by-character: each character independently
    has probability ``rate`` of being altered by one of four common typing
    mistakes (substitute with a random lowercase letter, delete, duplicate,
    or transpose with the next character). Deterministic for a given
    ``seed`` so the report is reproducible.
    """
    if not text:
        return text

    rng = random.Random(seed)
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        char = chars[i]
        if char.isalpha() and rng.random() < rate:
            mistake = rng.choice(["substitute", "delete", "duplicate", "transpose"])
            if mistake == "substitute":
                out.append(rng.choice(string.ascii_lowercase))
            elif mistake == "delete":
                pass  # drop the character entirely
            elif mistake == "duplicate":
                out.append(char)
                out.append(char)
            elif mistake == "transpose" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(char)
                i += 1  # already consumed the next character
        else:
            out.append(char)
        i += 1

    return "".join(out)


def randomize_casing(text: str, mode: str = "random", seed: int | None = None) -> str:
    """
    Change the casing of ``text``. ``mode`` is one of:

    - ``"upper"``: ALL CAPS (e.g. a user with caps-lock stuck on).
    - ``"lower"``: all lowercase (the most common real-world case — most
      people don't bother capitalising casual text).
    - ``"random"``: each letter independently upper/lowercased — a
      stress-test extreme, unlikely verbatim but exercises the same
      case-sensitivity a real model should not depend on.
    """
    if not text:
        return text

    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    if mode == "random":
        rng = random.Random(seed)
        return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)

    raise ValueError(f"Unknown casing mode: {mode!r}")


def perturb_series(texts: pd.Series, perturb_fn, random_state: int = 42) -> pd.Series:
    """
    Apply ``perturb_fn(text, seed=...)`` to every row of ``texts``, seeding
    each row deterministically off its position so re-running the same
    perturbation on the same data always yields the same corrupted text.
    """
    texts = pd.Series(texts).reset_index(drop=True)
    return pd.Series([perturb_fn(text, seed=random_state + i) for i, text in enumerate(texts)])


# Presets exercised by evaluate_robustness / run() by default. Each maps a
# human-readable name to a one-argument-short-of-ready perturbation
# function (still needs ``seed=`` at call time, supplied by perturb_series).
DEFAULT_PERTURBATIONS = {
    "typos_light": partial(inject_typos, rate=0.05),
    "typos_heavy": partial(inject_typos, rate=0.15),
    "uppercase": partial(randomize_casing, mode="upper"),
    "lowercase": partial(randomize_casing, mode="lower"),
    "random_case": partial(randomize_casing, mode="random"),
}


# ============================================================
# Evaluation
# ============================================================


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "critical_recall": critical_recall_score(y_true, y_pred),
    }


def evaluate_robustness(
    model,
    X_test,
    y_test,
    perturbations: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate ``model`` on the clean test set, then on each perturbed copy
    in ``perturbations`` (defaults to ``DEFAULT_PERTURBATIONS``). Returns
    one row per condition (``"clean"`` first) with the three headline
    metrics plus their delta versus the clean baseline — a negative delta
    means the perturbation hurt performance.
    """
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    X_test = pd.Series(X_test).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    baseline_pred = model.predict(X_test)
    baseline_metrics = _compute_metrics(y_test, baseline_pred)

    rows = [{
        "perturbation": "clean",
        **baseline_metrics,
        "delta_f1_macro": 0.0,
        "delta_recall_macro": 0.0,
        "delta_critical_recall": 0.0,
    }]

    for name, perturb_fn in perturbations.items():
        X_perturbed = perturb_series(X_test, perturb_fn, random_state=random_state)
        y_pred = model.predict(X_perturbed)
        metrics = _compute_metrics(y_test, y_pred)
        rows.append({
            "perturbation": name,
            **metrics,
            "delta_f1_macro": metrics["f1_macro"] - baseline_metrics["f1_macro"],
            "delta_recall_macro": metrics["recall_macro"] - baseline_metrics["recall_macro"],
            "delta_critical_recall": metrics["critical_recall"] - baseline_metrics["critical_recall"],
        })

    return pd.DataFrame(rows)


def summarize_worst_case(report: pd.DataFrame) -> dict:
    """
    Worst-case degradation across every non-clean perturbation in a
    ``evaluate_robustness`` report — the single number to watch: "how bad
    can it get on realistically messy input?".
    """
    perturbed = report[report["perturbation"] != "clean"]
    if perturbed.empty:
        return {"worst_delta_f1_macro": 0.0, "worst_delta_critical_recall": 0.0, "worst_perturbation": None}

    worst_row = perturbed.loc[perturbed["delta_f1_macro"].idxmin()]
    return {
        "worst_delta_f1_macro": float(perturbed["delta_f1_macro"].min()),
        "worst_delta_critical_recall": float(perturbed["delta_critical_recall"].min()),
        "worst_perturbation": str(worst_row["perturbation"]),
    }


def run() -> pd.DataFrame:
    """
    Load the current champion (registered "production" alias, falling
    back to "staging" if nothing is promoted yet), rebuild the same
    train/test split used to train it, run ``evaluate_robustness`` on the
    "raw" text variant's test set, write the report to
    ``paths.ROBUSTNESS_REPORT_PATH`` and return it.
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
    from mental_health.config.paths import DEFAULT_CLEAN_DATA_PATH, ROBUSTNESS_REPORT_PATH
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
    logger.info("Loaded '%s' v%s (alias=%s) for robustness evaluation", MLFLOW_REGISTERED_MODEL_NAME, version.version, alias_used)

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    splits = build_splits(df)
    X_test, y_test = splits["raw"]["X_test"], splits["raw"]["y_test"]

    report = evaluate_robustness(model, X_test, y_test)
    worst_case = summarize_worst_case(report)
    logger.info("Robustness worst case: %s", worst_case)

    ROBUSTNESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(ROBUSTNESS_REPORT_PATH, index=False)
    logger.info("Robustness report written to %s:\n%s", ROBUSTNESS_REPORT_PATH, report.to_string(index=False))

    return report


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
