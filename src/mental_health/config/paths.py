"""
Constantes de chemins centralisées pour le projet Mental Health Intelligence.

Tous les autres modules doivent importer les chemins depuis ce fichier —
ne jamais coder en dur des chemins de répertoires dans le code applicatif
ou les notebooks.
"""
from __future__ import annotations

from pathlib import Path

# ============================================================
# Ancrages racine
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_DIR: Path = PROJECT_ROOT / "src"
SAMPLE_OUTPUTS_DIR: Path = PROJECT_ROOT / "docs" / "sample_outputs"

# ============================================================
# Données
# ============================================================

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_CLEAN_DIR: Path = DATA_DIR / "clean"

RAW_DATA_PATH: Path = DATA_RAW_DIR / "Mental Health Disorder Detection Dataset.csv"
DEFAULT_CLEAN_DATA_PATH: Path = DATA_CLEAN_DIR / "mental_health_detection_clean.csv"

# ============================================================
# Modèles
# ============================================================

MODELS_DIR: Path = PROJECT_ROOT / "models"

MODEL_CANDIDATES: dict[str, list[Path]] = {
    "LinearSVC Balanced": [
        MODELS_DIR / "best_classical_model.joblib",
        SRC_DIR / "model" / "best_classical_model.joblib",
    ],
    "Hybrid SVC": [
        MODELS_DIR / "hybrid_svc_model.joblib",
    ],
    "MentalBERT": [
        MODELS_DIR / "mentalbert_pipeline.joblib",
        MODELS_DIR / "mental_bert_pipeline.joblib",
        MODELS_DIR / "mentalbert_model.joblib",
    ],
    "BERT Base": [
        MODELS_DIR / "bert_base_pipeline.joblib",
        MODELS_DIR / "bert_pipeline.joblib",
        MODELS_DIR / "bert_model.joblib",
    ],
}

# ============================================================
# Rapports et artefacts d'évaluation
# ============================================================

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CLASSICAL_REPORTS_DIR: Path = REPORTS_DIR / "tables" / "classical"
TRANSFORMER_REPORTS_DIR: Path = REPORTS_DIR / "transformers"
CLINICAL_TABLES_DIR: Path = REPORTS_DIR / "tables" / "clinical"

FINAL_TEST_METRICS_PATH: Path = CLASSICAL_REPORTS_DIR / "final_test_metrics.csv"
NESTED_CV_SUMMARY_PATH: Path = CLASSICAL_REPORTS_DIR / "nested_cv_summary.csv"
NORMAL_CV_SUMMARY_PATH: Path = CLASSICAL_REPORTS_DIR / "normal_cv_summary.csv"

# Phase 11 : chaque candidat benchmarké (pas seulement le champion), évalué
# sur le MÊME jeu de test held-out avec la suite complète de métriques de
# rigueur (f1_macro, recall_macro, critical_recall, mcc, brier_score, ece) --
# régénéré à chaque exécution de train.py, afin de rester à jour à mesure
# que de nouveaux candidats (XGBoost/LightGBM maintenant, embeddings/DistilBERT
# plus tard) rejoignent le registry.
MODEL_COMPARISON_PATH: Path = CLASSICAL_REPORTS_DIR / "model_comparison.csv"
GLOBAL_CLINICAL_REVIEW_PATH: Path = CLINICAL_TABLES_DIR / "global_comparison_for_clinical_review.csv"

# Phase 11, tranches restantes : robustesse (perturbations de fautes de
# frappe/casse), découpage par biais (évaluation par sous-groupe) et le
# fine-tune DistilBERT — les rapports se trouvent aux côtés de
# model_comparison.csv (candidats classiques) et sous TRANSFORMER_REPORTS_DIR
# (DistilBERT), régénérés par leurs propres scripts d'exécution, pas par le
# run() principal de train.py.
ROBUSTNESS_REPORT_PATH: Path = CLASSICAL_REPORTS_DIR / "robustness_report.csv"
BIAS_SLICING_REPORT_PATH: Path = CLASSICAL_REPORTS_DIR / "bias_slicing_report.csv"
DISTILBERT_METRICS_PATH: Path = TRANSFORMER_REPORTS_DIR / "distilbert_metrics.csv"
DISTILBERT_MODEL_DIR: Path = MODELS_DIR / "distilbert_finetuned"

# ============================================================
# Base de connaissances RAG
# ============================================================

RAG_SOURCE_DIR: Path = PROJECT_ROOT / "rag_source"
RAG_INDEX_DIR: Path = PROJECT_ROOT / "faiss_index"

# ============================================================
# Constantes du domaine
# ============================================================

CLASS_LABELS: list[str] = [
    "ADHD",
    "Anxiety",
    "Autism",
    "Bipolar",
    "BPD",
    "Depression",
    "Schizophrenia",
]

CRITICAL_LABELS: list[str] = ["Bipolar", "Schizophrenia"]
