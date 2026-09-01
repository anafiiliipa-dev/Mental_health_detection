"""
DistilBERT fine-tuning (Phase 11, remaining slice: "DistilBERT
(fine-tuning léger)") -- the "full transformer fine-tune" end of the
spectrum this project's roadmap always meant to compare against, with the
frozen sentence-embedding candidates (``embedding_wrapper.py``) as the
deliberately cheaper "solution intermediaire" in between TF-IDF and this.

Unlike every classical candidate in ``model_registry.py``, this is NOT
wired into the nested-CV benchmark (``benchmark.py``) or the champion
selection (``champion.py``) -- fine-tuning a transformer inside a nested
CV loop (outer x inner x hyperparameter grid) would mean dozens of
multi-epoch training runs, which is not a "léger" fine-tune, it's a
different order of magnitude of compute. Instead this trains ONE model on
the same train/test split ``train.py`` uses (via ``build_splits``), with a
small fixed hyperparameter set, and reports the same headline metrics
(f1_macro, recall_macro, critical_recall) plus MCC -- directly comparable
to a row of ``model_comparison.csv`` -- so it can be read side by side with
every classical/embedding candidate without pretending it went through
the same nested-CV rigor.

``torch``/``transformers``/``datasets`` are imported lazily (inside
functions, never at module import time), exactly like
``embedding_wrapper.py``'s ``sentence-transformers`` import -- so this
module can be imported (e.g. by a test, or by something that just wants
``DISTILBERT_MODEL_NAME``) without the heavy ``transformers`` extra
installed. Actually training requires ``pip install -e ".[transformers]"``
and, realistically, a GPU -- CPU fine-tuning of even DistilBERT on this
dataset will be slow; this is deliberately left to run from Ana's own
machine (device_bash here has no GPU and can't run the Windows venv
anyway), same convention as every other real training run in this
project.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score

from mental_health.train.benchmark import critical_recall_score

logger = logging.getLogger(__name__)

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 256
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5


class _TextDataset:
    """Minimal torch Dataset wrapping tokenized texts + integer labels."""

    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _build_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    sorted_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(sorted_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label


def _compute_metrics_fn(id_to_label: dict[int, str]):
    """Build the ``compute_metrics`` callback the HF Trainer calls after each eval pass."""

    def _compute(eval_pred):
        logits, label_ids = eval_pred
        y_pred_ids = np.argmax(logits, axis=1)
        y_pred = [id_to_label[i] for i in y_pred_ids]
        y_true = [id_to_label[i] for i in label_ids]
        return {
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "critical_recall": critical_recall_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

    return _compute


def finetune_distilbert(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str = DISTILBERT_MODEL_NAME,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_dir: str | None = None,
) -> dict:
    """
    Fine-tune ``model_name`` (DistilBERT by default) as a sequence
    classifier on ``X_train``/``y_train``, evaluate on ``X_test``/
    ``y_test``, and return the trained pipeline (tokenizer + model) plus
    the same headline metrics used everywhere else in this project.

    Requires the ``transformers`` extra
    (``pip install -e ".[transformers]"``) -- ``torch``, ``transformers``
    are imported here, not at module level.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    X_train = pd.Series(X_train).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    X_test = pd.Series(X_test).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    label_to_id, id_to_label = _build_label_maps(list(y_train) + list(y_test))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH
    )
    test_encodings = tokenizer(
        list(X_test), truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH
    )

    train_dataset = _TextDataset(train_encodings, [label_to_id[label] for label in y_train])
    test_dataset = _TextDataset(test_encodings, [label_to_id[label] for label in y_test])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_to_id), id2label=id_to_label, label2id=label_to_id
    )

    training_args = TrainingArguments(
        output_dir=output_dir or "distilbert_finetune_tmp",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],  # this project logs to MLflow explicitly, not via the HF integration
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=_compute_metrics_fn(id_to_label),
    )

    logger.info("Fine-tuning %s for %d epoch(s) on %d training rows", model_name, num_epochs, len(X_train))
    trainer.train()
    eval_result = trainer.evaluate()

    metrics = {
        "f1_macro": eval_result["eval_f1_macro"],
        "recall_macro": eval_result["eval_recall_macro"],
        "critical_recall": eval_result["eval_critical_recall"],
        "mcc": eval_result["eval_mcc"],
    }
    logger.info("DistilBERT fine-tune metrics: %s", metrics)

    return {"trainer": trainer, "tokenizer": tokenizer, "model": model, "metrics": metrics, "label_to_id": label_to_id}


def run(num_epochs: int = DEFAULT_NUM_EPOCHS) -> dict:
    """
    End-to-end runner: load the same clean dataset and train/test split
    ``train.py`` uses (raw text variant only -- masking is a leakage
    control for TF-IDF's literal token matching; a fine-tuned transformer's
    exposure to diagnostic terms is a separate, not-yet-scoped question),
    fine-tune DistilBERT, write its metrics next to the classical
    ``model_comparison.csv`` (as ``paths.DISTILBERT_METRICS_PATH``) so both
    are readable side by side, and save the fine-tuned model under
    ``paths.DISTILBERT_MODEL_DIR``.
    """
    from mental_health.config.paths import (
        DEFAULT_CLEAN_DATA_PATH,
        DISTILBERT_METRICS_PATH,
        DISTILBERT_MODEL_DIR,
    )
    from mental_health.train.train import build_splits

    df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
    splits = build_splits(df)
    X_train, y_train = splits["raw"]["X_train"], splits["raw"]["y_train"]
    X_test, y_test = splits["raw"]["X_test"], splits["raw"]["y_test"]

    result = finetune_distilbert(X_train, y_train, X_test, y_test, num_epochs=num_epochs, output_dir=str(DISTILBERT_MODEL_DIR))

    DISTILBERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result["trainer"].save_model(str(DISTILBERT_MODEL_DIR))
    result["tokenizer"].save_pretrained(str(DISTILBERT_MODEL_DIR))

    metrics_row = {"model": "DistilBERT_finetuned", "text_variant": "raw", **result["metrics"]}
    DISTILBERT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_row]).to_csv(DISTILBERT_METRICS_PATH, index=False)
    logger.info("DistilBERT metrics written to %s: %s", DISTILBERT_METRICS_PATH, metrics_row)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
