"""
Fine-tuning de DistilBERT (Phase 11, slice restante : "DistilBERT
(fine-tuning léger)") -- l'extrémité "full transformer fine-tune" du
spectre auquel la roadmap de ce projet a toujours prévu de se comparer,
les candidats à sentence-embedding gelés (``embedding_wrapper.py``) étant
la "solution intermediaire" délibérément moins coûteuse entre TF-IDF et
ceci.

Contrairement à tous les candidats classiques de ``model_registry.py``,
ceci N'EST PAS branché sur le benchmark de nested CV (``benchmark.py``) ni
sur la sélection du champion (``champion.py``) -- fine-tuner un
transformer à l'intérieur d'une boucle de nested CV (externe x interne x
grille d'hyperparamètres) impliquerait des dizaines d'exécutions
d'entraînement multi-époques, ce qui n'est pas un fine-tune "léger", c'est
un ordre de grandeur de calcul différent. À la place, ceci entraîne UN
seul modèle sur le même split train/test que celui utilisé par
``train.py`` (via ``build_splits``), avec un petit ensemble
d'hyperparamètres fixes, et rapporte les mêmes métriques principales
(f1_macro, recall_macro, critical_recall) plus le MCC -- directement
comparable à une ligne de ``model_comparison.csv`` -- pour pouvoir être lu
côte à côte avec chaque candidat classique/embedding sans prétendre être
passé par la même rigueur de nested CV.

``torch``/``transformers``/``datasets`` sont importés paresseusement (dans
les fonctions, jamais au niveau du module), exactement comme l'import de
``sentence-transformers`` dans ``embedding_wrapper.py`` -- afin que ce
module puisse être importé (par exemple par un test, ou par quelque chose
qui veut juste ``DISTILBERT_MODEL_NAME``) sans que l'extra ``transformers``
(lourd) soit installé. Entraîner réellement le modèle nécessite
``pip install -e ".[transformers]"`` et, de manière réaliste, un GPU -- le
fine-tuning CPU même de DistilBERT sur ce dataset sera lent ; ceci est
volontairement laissé à exécuter depuis la propre machine d'Ana (le
device_bash ici n'a pas de GPU et ne peut de toute façon pas exécuter le
venv Windows), même convention que pour tout autre entraînement réel de
ce projet.
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
    """Dataset torch minimal enveloppant des textes tokenizés + des labels entiers."""

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
    """Construit le callback ``compute_metrics`` que le Trainer HF appelle après chaque passe d'évaluation."""

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
    Fine-tune ``model_name`` (DistilBERT par défaut) comme classifieur de
    séquence sur ``X_train``/``y_train``, évalue sur ``X_test``/
    ``y_test``, et retourne le pipeline entraîné (tokenizer + modèle) plus
    les mêmes métriques principales utilisées partout ailleurs dans ce
    projet.

    Nécessite l'extra ``transformers``
    (``pip install -e ".[transformers]"``) -- ``torch``, ``transformers``
    sont importés ici, pas au niveau du module.
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
        report_to=[],  # ce projet logge vers MLflow explicitement, pas via l'intégration HF
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
    Runner de bout en bout : charge le même dataset nettoyé et le même
    split train/test que ``train.py`` utilise (variante de texte raw
    uniquement -- le masking est un contrôle de fuite pour la
    correspondance littérale de tokens de TF-IDF ; l'exposition d'un
    transformer fine-tuné aux termes diagnostiques est une question
    séparée, pas encore cadrée), fine-tune DistilBERT, écrit ses métriques
    à côté du ``model_comparison.csv`` classique (dans
    ``paths.DISTILBERT_METRICS_PATH``) pour que les deux soient lisibles
    côte à côte, et sauvegarde le modèle fine-tuné dans
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
