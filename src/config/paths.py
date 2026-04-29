"""
Centralised path constants for the Mental Health Intelligence project.

All other modules should import paths from here — never hardcode
directory paths inside application or notebook code.
"""
from __future__ import annotations

from pathlib import Path

# ============================================================
# Root anchors
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_DIR: Path = PROJECT_ROOT / "src"

# ============================================================
# Data
# ============================================================

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_CLEAN_DIR: Path = DATA_DIR / "clean"
DEFAULT_CLEAN_DATA_PATH: Path = DATA_CLEAN_DIR / "mental_health_detection_clean.csv"

# ============================================================
# Models
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
# Reports & evaluation artifacts
# ============================================================

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CLASSICAL_REPORTS_DIR: Path = REPORTS_DIR / "tables" / "classical"
TRANSFORMER_REPORTS_DIR: Path = REPORTS_DIR / "transformers"
CLINICAL_TABLES_DIR: Path = REPORTS_DIR / "tables" / "clinical"

FINAL_TEST_METRICS_PATH: Path = CLASSICAL_REPORTS_DIR / "final_test_metrics.csv"
NESTED_CV_SUMMARY_PATH: Path = CLASSICAL_REPORTS_DIR / "nested_cv_summary.csv"
NORMAL_CV_SUMMARY_PATH: Path = CLASSICAL_REPORTS_DIR / "normal_cv_summary.csv"
GLOBAL_CLINICAL_REVIEW_PATH: Path = CLINICAL_TABLES_DIR / "global_comparison_for_clinical_review.csv"

# ============================================================
# RAG knowledge base
# ============================================================

RAG_SOURCE_DIR: Path = PROJECT_ROOT / "rag_source"
RAG_INDEX_DIR: Path = PROJECT_ROOT / "faiss_index"

# ============================================================
# Domain constants
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
