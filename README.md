# 🧠 Mental Health Intelligence — NLP Clinical Dashboard

> A clinical-grade NLP system combining robust Machine Learning (Nested CV) and Large Language Models (RAG / OpenRouter) for mental health text triage.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project provides an **early mental health screening support tool** from text data (social media posts, patient narratives). It transforms unstructured text into actionable clinical insights while guaranteeing maximum statistical rigour.

> ⚠️ **Non-Diagnostic Disclaimer** — This tool is a decision-support aid, not an autonomous diagnostic device. It must not replace clinical judgment.

### Clinical Problem

How to ensure reliable, explainable triage of mental health conditions (ADHD, Bipolar Disorder, Schizophrenia, etc.) despite class imbalance and the semantic subtlety of patient testimonials?

---

## Architecture

### 1. Machine Learning Pipeline (Robust Baseline)

- **Champion Model**: LinearSVC with balanced class weighting
- **Statistical Validation**: Nested Cross-Validation (outer K-Fold + inner GridSearch) to eliminate selection bias
- **Key Metric**: Critical Recall (94%+) to minimise false negatives on high-risk conditions

### 2. Generative AI Module (Copilot)

- **RAG (Retrieval-Augmented Generation)**: Question-answering system grounded in project documentation
- **OpenRouter Client**: Resilient LLM integration (GPT-4o mini) with timeout and exponential backoff

---

## Dashboard

A Streamlit application with glassmorphism design, structured around six pages:

| Page | Purpose |
|---|---|
| **Overview** | Global metric monitoring and dataset status |
| **Predictions** | Real-time text classification with probability bars |
| **Monitoring** | Validation artifact analysis from the training pipeline |
| **Chat** | RAG-powered assistant to explore project methodology |
| **History** | Session-level prediction log |
| **About** | Project context and next steps |

---

## Project Structure

```
Mental_health/
├── notebooks/
│   ├── 01_data_cleaning_and_export.ipynb
│   ├── 02_classical_ml_benchmark.ipynb
│   ├── 02b_smote_sensitivity_analysis.ipynb
│   ├── 03_transformers_benchmark.ipynb      # Colab / GPU
│   ├── 04_clinical_evaluation_error_analysis.ipynb
│   └── 05_deployment_mvp_llm_perspectives.ipynb
├── src/
│   ├── app/
│   │   ├── app.py                           # Streamlit entry point
│   │   └── openrouter_client.py             # LLM client with retry
│   ├── config/
│   │   └── paths.py                         # Centralised path constants
│   ├── model/
│   │   └── services.py                      # Model loading service
│   └── rag/
│       └── simple_rag.py                    # RAG pipeline
├── rag_source/                              # RAG knowledge base
├── tests/
│   ├── test_predictions.py
│   ├── test_rag.py
│   └── test_client.py
├── data/                                    # gitignored
├── models/                                  # gitignored
├── reports/                                 # gitignored
├── requirements.txt
├── requirements_streamlit.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/anafiiliipa-dev/Mental_health_detection.git
   cd Mental_health_detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   ```bash
   cp .env.example .env
   # Edit .env and fill in your OPENROUTER_API_KEY
   ```

5. **Run the dashboard**
   ```bash
   python -m streamlit run src/app/app.py
   ```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Notebook Pipeline

| Notebook | Purpose |
|---|---|
| `01` | Data cleaning, text normalisation, near-duplicate detection |
| `02` | Classical ML benchmark (TF-IDF + LinearSVC, Nested CV) |
| `02b` | SMOTE sensitivity analysis |
| `03` | Transformer benchmark (BERT, MentalBERT) — requires Colab / GPU |
| `04` | Clinical evaluation and error analysis |
| `05` | Deployment MVP and LLM perspectives |

> Notebooks `03` and `05` are designed for Google Colab. Run them via the Colab badge at the top of each notebook.

---

## Ethics & Limitations

- **Non-Diagnostic**: Decision-support only — must not replace clinical judgment.
- **Privacy**: Classical models run locally; no patient data sent to external APIs.
- **Transparency**: All metrics grounded in Nested CV to guarantee generalisability.
- **Recall Focus**: System optimises for recall on high-risk conditions to minimise false negatives.

---

## Authors

**Ana Gouveia & Nicolas Moignard — DSFS-OD-14 cohort**
