"""Unit/integration tests for src/mental_health/train/train.py."""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.train import train as train_module
from mental_health.train.train import build_splits, compute_dataset_hash

# ============================================================
# compute_dataset_hash
# ============================================================

class TestComputeDatasetHash:
    def test_returns_16_hex_chars(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("body,category\nhello,ADHD\n")
        h = compute_dataset_hash(f)
        assert len(h) == 16
        int(h, 16)  # must be valid hex

    def test_is_deterministic(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("body,category\nhello,ADHD\n")
        assert compute_dataset_hash(f) == compute_dataset_hash(f)

    def test_differs_for_different_content(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("body,category\nhello,ADHD\n")
        f2.write_text("body,category\ngoodbye,Anxiety\n")
        assert compute_dataset_hash(f1) != compute_dataset_hash(f2)


# ============================================================
# build_splits
# ============================================================

class TestBuildSplits:
    def _tiny_df(self):
        return pd.DataFrame({
            "body": [f"raw text number {i} about feelings" for i in range(20)],
            "body_masked": [f"[CONDITION] text number {i} about feelings" for i in range(20)],
            "category": (["ADHD", "Anxiety"] * 10),
        })

    def test_returns_raw_and_masked_keys(self):
        splits = build_splits(self._tiny_df(), test_size=0.3)
        assert set(splits.keys()) == {"raw", "masked"}

    def test_raw_and_masked_share_the_same_rows(self):
        # Both variants must come from the identical train/test index split,
        # so the "raw" and "masked" runs are a fair, comparable experiment.
        splits = build_splits(self._tiny_df(), test_size=0.3)
        assert list(splits["raw"]["y_train"]) == list(splits["masked"]["y_train"])
        assert list(splits["raw"]["y_test"]) == list(splits["masked"]["y_test"])

    def test_train_and_test_sizes_are_disjoint_and_complete(self):
        df = self._tiny_df()
        splits = build_splits(df, test_size=0.3)
        n_train = len(splits["raw"]["X_train"])
        n_test = len(splits["raw"]["X_test"])
        assert n_train + n_test == len(df)

    def test_masked_variant_uses_the_masked_column(self):
        splits = build_splits(self._tiny_df(), test_size=0.3)
        assert splits["masked"]["X_train"].iloc[0].startswith("[CONDITION]")
        assert not splits["raw"]["X_train"].iloc[0].startswith("[CONDITION]")


# ============================================================
# run() — end-to-end smoke test against a tiny synthetic dataset
# ============================================================

def _write_tiny_clean_csv(path: Path, rows_per_class: int = 20) -> None:
    labels = ["ADHD", "Anxiety", "Bipolar", "Schizophrenia"]
    texts = {
        "ADHD": "I can't focus and my mind wanders constantly during the day",
        "Anxiety": "worry and panic every single day of my life about everything",
        "Bipolar": "manic episodes followed by crushing lows that last for weeks",
        "Schizophrenia": "I hear voices when nobody is around and it scares me",
    }
    rows = []
    for label in labels:
        for i in range(rows_per_class):
            rows.append({
                "body": f"{texts[label]} (variant {i})",
                "body_masked": f"[CONDITION] related text (variant {i})",
                "category": label,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _fake_precompute_dataset_embeddings(texts, model_name=None):
    """
    Fast, offline, deterministic stand-in for the real sentence-transformer
    precompute step -- this end-to-end test must not require network access
    or a real model download just to exercise train.py's orchestration.
    """
    dim = 4
    cache = {}
    for text in dict.fromkeys(texts):  # de-dup, preserves order
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        cache[text] = rng.random(dim)
    return cache


class TestRunEndToEnd:
    def test_run_produces_a_champion_and_logs_to_mlflow(self, tmp_path, monkeypatch):
        # Redirect MLflow's tracking store (SQLite db + artifact root) to a
        # throwaway tmp directory — this test must never write into the
        # project's real mlflow.db / mlruns/.
        monkeypatch.setattr(train_module, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.setattr(train_module, "MLFLOW_ARTIFACT_ROOT", f"file:{tmp_path / 'mlruns'}")
        # Same for the Phase 11 model comparison table — must never write
        # into the project's real reports/tables/classical/ either.
        comparison_path = tmp_path / "model_comparison.csv"
        monkeypatch.setattr(train_module, "MODEL_COMPARISON_PATH", comparison_path)

        data_path = tmp_path / "clean.csv"
        _write_tiny_clean_csv(data_path, rows_per_class=20)

        monkeypatch.setattr(train_module, "precompute_dataset_embeddings", _fake_precompute_dataset_embeddings)

        result = train_module.run(data_path=data_path)

        assert result["champion_config"]["model_name"] in {
            "LinearSVC_balanced", "LinearSVC_plain", "LogReg_balanced", "LogReg_plain", "MultinomialNB",
            "XGBoost_balanced", "LightGBM_balanced", "Embedding_LogReg", "Embedding_SVM",
        }
        assert result["champion_config"]["text_variant"] in {"raw", "masked"}
        assert 0.0 <= result["eval_result"]["f1_macro"] <= 1.0
        assert (tmp_path / "mlflow.db").exists(), "MLflow SQLite tracking store was not created"
        assert (tmp_path / "mlruns").exists(), "MLflow artifact store was not created"

        assert result["registered_version"] is not None
        client = mlflow.MlflowClient()
        staged = client.get_model_version_by_alias(result["registered_model_name"], "staging")
        assert staged.version == result["registered_version"]

        # Phase 11: the comparison table covers every benchmarked candidate
        # (7 classical models x 2 text variants, plus Embedding_LogReg/
        # Embedding_SVM on the "raw" variant only = 16 rows), written to
        # disk and returned in the result dict.
        comparison_df = result["model_comparison"]
        assert comparison_path.exists(), "Model comparison CSV was not written"
        assert len(comparison_df) == 16
        for column in ["model", "text_variant", "f1_macro", "recall_macro", "critical_recall", "mcc"]:
            assert column in comparison_df.columns
        assert comparison_df["f1_macro"].is_monotonic_decreasing
