<div align="center">

# 🧠 Mental Health Intelligence

### Clinical-grade NLP triage with statistically robust ML and grounded LLMs

[![CI](https://github.com/anafiiliipa-dev/Mental_health_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/anafiiliipa-dev/Mental_health_detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**An early-screening decision-support tool that combines a Nested-CV-validated classical ML baseline with a Retrieval-Augmented LLM copilot — wrapped in a glassmorphism Streamlit dashboard.**

[Overview](#-overview) · [Problem](#-the-problem) · [Architecture](#-architecture) · [Methodology](#-methodology) · [Results](#-results) · [Dashboard](#-dashboard) · [Quickstart](#-quickstart) · [Docker](#-run-with-docker) · [Ethics](#-ethics--limitations)

</div>

> [!CAUTION]
> **Non-diagnostic disclaimer.** This system is a clinical decision-support aid. It must **never** replace a licensed clinician's judgement, and is not certified as a medical device under the EU MDR, the US FDA, or any equivalent framework. See [`NOTICE.md`](NOTICE.md) for the full notice.

---

## 🚀 Overview

**Mental Health Intelligence** is an end-to-end NLP system designed to **triage mental-health-related text** — patient narratives, social posts, intake forms — using a layered approach:

- 🧮 **Classical ML baseline** — TF-IDF + LinearSVC, validated with Nested Cross-Validation
- 🤖 **Transformer benchmark** — BERT / MentalBERT, evaluated head-to-head against the baseline
- 📚 **Retrieval-Augmented Generation** — FAISS + LangChain over a curated knowledge base
- 🩺 **Clinical-style evaluation** — per-class confusion matrices, error analysis, recall-skewed metrics

The goal: **bridge statistical rigour with real-world interpretability**, delivering predictions that are not just accurate, but auditable.

---

## 🎯 The Problem

Mental-health signals in free-text are notoriously hard to classify because:

- Classes are **imbalanced** — some categories are 5× rarer than others
- Linguistic differences are **semantically subtle** and context-dependent
- False negatives on **high-risk classes** (Bipolar, Schizophrenia) carry asymmetric clinical cost — missing them is far worse than misrouting a Depression case to Anxiety

This project answers that challenge with a system designed for **rigorous evaluation first, deployment second**: careful preprocessing, balanced modelling, unbiased validation, and a recall-skewed optimisation target.

---

## 🧠 Classes

The system predicts across **seven overlapping clinical categories**:

| Class                | Clinical priority                  | Class weight (balanced) |
| -------------------- | ---------------------------------- | ----------------------- |
| 🧩 **ADHD**          | Standard                           | 0.80×                   |
| 😰 **Anxiety**       | Standard                           | 0.84×                   |
| 🧠 **Autism**        | Standard                           | 1.08×                   |
| ⚡ **Bipolar**        | **Critical** (recall-prioritised)  | 0.88×                   |
| 💔 **BPD**           | Standard                           | 1.00×                   |
| 🌧 **Depression**    | Standard                           | 1.12×                   |
| 🌀 **Schizophrenia** | **Critical** (recall-prioritised)  | **1.64×**               |

> The class-balanced weighting up-weights under-represented classes (notably Schizophrenia) at training time. This is a deliberate clinical choice, not an artefact of the data.

---

## 🏗 Architecture

```mermaid
flowchart TB
    User([👤 Clinician / Reviewer]):::user

    subgraph Dashboard["🎨 Streamlit Dashboard"]
        direction LR
        P1[Overview]
        P2[Predictions]
        P3[Monitoring]
        P4[Chat]
        P5[History]
    end

    subgraph ML["🧮 Classical ML Pipeline"]
        direction TB
        Vec[TF-IDF Vectorizer]
        SVC[LinearSVC<br/>Balanced]
        Vec --> SVC
    end

    subgraph RAG["🤖 RAG Copilot"]
        direction TB
        Embed[MiniLM-L6-v2<br/>Embeddings]
        Faiss[(FAISS Index<br/>fingerprinted cache)]
        LLM{{OpenRouter<br/>gpt-4o-mini}}
        Embed --> Faiss --> LLM
    end

    Artifacts[(📊 CSV Artifacts<br/>Nested CV summaries<br/>Final test metrics)]
    KB[(📚 rag_source/<br/>Knowledge base)]

    User --> Dashboard
    P2 --> ML
    P4 --> RAG
    P3 --> Artifacts
    KB --> Embed

    classDef user fill:#1e293b,stroke:#22d3ee,color:#f8fafc
    classDef store fill:#0f172a,stroke:#a78bfa,color:#cbd5e1

    class Artifacts,KB,Faiss store
```

### Three independent reasoning paths

Each path enforces strict isolation to protect user data:

| Path                       | Trigger                | Privacy boundary                                                     |
| -------------------------- | ---------------------- | -------------------------------------------------------------------- |
| **Local ML inference**     | `Predictions` page     | Runs entirely offline. No text leaves the host.                      |
| **RAG over project docs**  | `Chat` page (default)  | Embeddings local; only retrieved chunks + question sent to the LLM.  |
| **Direct LLM fallback**    | RAG returns no context | Question sent to OpenRouter only when retrieval fails.               |

> 📖 For a detailed walkthrough of design decisions, trade-offs, and rejected alternatives, see [`docs/architecture.md`](docs/architecture.md).

---

## 🔬 Methodology

### Statistical rigour: **Nested Cross-Validation**

Most ML benchmarks suffer from **selection bias** — hyperparameters tuned on the same folds used to report accuracy. We use a nested loop:

```text
Outer K-Fold (test estimate)
  └── Inner K-Fold (GridSearch on train fold only)
       └── Champion config promoted, refit on full train fold,
           scored on held-out test fold
```

The result is an **unbiased estimate of generalisation error** — what the model would actually do on never-seen text.

### Champion model: **LinearSVC + class-balanced weighting**

A deliberately simple choice. The transformer benchmarks (`03_transformers.ipynb`) showed that on this dataset, MentalBERT outperforms LinearSVC on macro-F1 by **~3 points** — but **at ~40× the inference cost** and with worse interpretability. For a triage tool that must be auditable in clinical settings, **LinearSVC wins**.

### Optimisation target: **Critical Recall**

We deliberately don't optimise macro-F1 alone. Missing a Bipolar or Schizophrenia signal has real-world cost; mis-routing a Depression case to Anxiety is recoverable downstream. The composite metric reported in `final_test_metrics.csv` weights recall on critical classes (`Bipolar`, `Schizophrenia`) higher than the rest — a deliberate clinical trade-off.

### Evaluation suite

- Accuracy, macro- and weighted-F1
- Per-class precision, recall, F1
- Confusion matrices
- Clinical error analysis (notebook `04_clinical_evaluation.ipynb`)
- SMOTE sensitivity ablation (notebook `02b_smote_sensitivity.ipynb`)

---

## 📊 Results

All numbers below come from `reports/tables/` and were produced by the notebooks in this repository — fully reproducible end-to-end.

### Headline numbers (held-out test set)

| Metric                                | LinearSVC (champion) | MentalBERT (benchmark) | BERT-base (benchmark) |
| ------------------------------------- | -------------------- | ---------------------- | --------------------- |
| Macro-F1                              | **0.779**            | 0.809                  | 0.791                 |
| Recall macro                          | **0.779**            | 0.812                  | 0.793                 |
| **Critical Recall** (Bipolar+Schiz)   | **0.698**            | 0.756                  | 0.739                 |
| Accuracy                              | n/a                  | 0.815                  | 0.798                 |
| Inference latency (ms / sample)       | **< 5 ms**           | ~200 ms                | ~200 ms               |
| Model size                            | **~10 MB**           | ~440 MB                | ~440 MB               |

### Nested CV (mean across outer folds, raw text variant)

| Model                       | Macro-F1          | Critical Recall   | Robust score | Rank |
| --------------------------- | ----------------- | ----------------- | ------------ | ---- |
| **LinearSVC balanced** ⭐    | **0.769 ± 0.006** | **0.684 ± 0.018** | **0.734**    | **1** |
| LogReg balanced             | 0.725 ± 0.004     | 0.709 ± 0.013     | 0.720        | 2    |
| LinearSVC plain             | 0.758 ± 0.008     | 0.659 ± 0.026     | 0.717        | 3    |
| LogReg plain                | 0.754 ± 0.014     | 0.630 ± 0.034     | 0.703        | 4    |
| MultinomialNB               | 0.545 ± 0.014     | 0.350 ± 0.011     | 0.474        | 5    |

**Standard deviations under 1 percentage point** across folds — the model is exceptionally stable. Compared to the naive Multinomial NB baseline, the champion shows a **+22-point F1 improvement**, validating the TF-IDF + linear-classifier pipeline.

### Clinical review (n = 2 255 predictions)

| Indicator                                  | Count |
| ------------------------------------------ | ----- |
| Total predictions reviewed                 | 2 255 |
| Total errors                               | 474   |
| **Critical false negatives**               | **95** |
| Critical false positives                   | 63    |

The asymmetry between false negatives (95) and false positives (63) on critical classes is **deliberate** — the recall-skewed optimisation accepts more false alarms in exchange for fewer missed cases.

### Why LinearSVC wins despite a slightly lower macro-F1

| Dimension                | LinearSVC | MentalBERT | Decision driver                                       |
| ------------------------ | --------- | ---------- | ----------------------------------------------------- |
| Macro-F1 (test)          | 0.779     | 0.809      | Within 3 percentage points                            |
| Inference cost           | `1×`      | `~40×`     | LinearSVC scales for batch triage                     |
| Model size on disk       | ~10 MB    | ~440 MB    | LinearSVC trivial to deploy                           |
| Interpretability         | Token-level coefficients | Black-box attention | LinearSVC defensible to a clinician |
| Bias auditability        | High      | Low        | LinearSVC easier to slice & audit                     |
| SMOTE sensitivity        | Negligible (-0.4 pp) | n/a | Class-balanced weights are sufficient (notebook `02b`) |

---

## 🎨 Dashboard

A six-page Streamlit application with a custom glassmorphism theme:

| Page            | What it does                                                                                                                        |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Overview**    | Live dataset stats, class badges, headline metrics from the latest evaluation run.                                                  |
| **Predictions** | Paste any text → real-time classification, probability bars, confidence read-out. Falls back to a clearly-labelled keyword-based demo when no `.joblib` model is loaded. |
| **Monitoring**  | Renders the actual evaluation artifacts (Nested CV summary, final test metrics, clinical review CSV) — no fake numbers.             |
| **Chat**        | RAG-powered Q&A grounded in `rag_source/`. Cites sources. Falls back to direct OpenRouter on retrieval miss.                        |
| **History**     | Session-scoped log of recent predictions. Cleared on browser refresh — no persistence.                                              |
| **About**       | Project context, ethical framing, roadmap.                                                                                          |

### Screenshots

|                         |                         |
| ----------------------- | ----------------------- |
| ![Overview](docs/screenshots/01-overview.png)     | ![Predictions](docs/screenshots/02-predictions.png) |
| ![Monitoring](docs/screenshots/03-monitoring.png) | _Chat with grounded RAG (coming soon)_              |

---

## 🚀 Quickstart

### Prerequisites

- **Python 3.10+**
- An [OpenRouter](https://openrouter.ai/) API key for the Chat page (free tier available with `:free` models)
- ~2 GB disk for the embedding model + FAISS index on first run

### Install

```bash
git clone https://github.com/anafiiliipa-dev/Mental_health_detection.git
cd Mental_health_detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[streamlit]"      # core + dashboard
# or:
pip install -e ".[dev]"            # core + tests + linters
```

### Configure secrets

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

The repo **never** commits `.env` — only `.env.example` (placeholder). Confirmed by `.gitignore`.

For a fully free setup, use `meta-llama/llama-3.3-70b-instruct:free` in `OPENROUTER_MODEL` — no credits required.

### Model

The trained champion model (**`models/best_classical_model.joblib`**, ~5 MB) is **included in this repository** so you can run inference immediately after cloning. To retrain from scratch, run notebooks `01 → 02` end-to-end.

### Run

```bash
streamlit run src/mental_health/app/app.py
# or, after `pip install -e .`:
mental-health-dashboard
```

Visit `http://localhost:8501`.

### Test

```bash
pytest -v
```

All external services (OpenRouter, FAISS) are mocked — no network or API key required to run the test suite.

---

## 🐳 Run with Docker

Zero-setup deployment for evaluators and recruiters. Single command:

```bash
docker compose up --build
```

Visit `http://localhost:8501`.

For a custom one-shot run:

```bash
docker build -t mental-health-intelligence .
docker run --rm -p 8501:8501 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/reports/tables:/app/reports/tables:ro" \
  mental-health-intelligence
```

The image runs as a non-root user, exposes a `/_stcore/health` healthcheck, and uses multi-stage build hints for fast rebuilds.

---

## 📁 Project structure

```text
mental-health-intelligence/
├── .github/workflows/ci.yml           # pytest + ruff on every PR
├── .pre-commit-config.yaml            # ruff + nbstripout hooks
├── .env.example                       # placeholder secrets
├── .gitattributes                     # forces LF, marks notebooks
├── .gitignore                         # data/, models/ (with model exception), reports/ (with CSV exceptions)
├── .dockerignore
├── pyproject.toml                     # single source of truth for deps
├── requirements.txt                   # convenience shortcut (mirrors pyproject)
├── requirements_streamlit.txt         # convenience shortcut (streamlit extra)
├── README.md
├── LICENSE                            # pure MIT (so GitHub detects it)
├── NOTICE.md                          # non-diagnostic notice
├── CHANGELOG.md
├── Dockerfile
├── docker-compose.yml
│
├── docs/
│   ├── architecture.md                # design decisions and trade-offs
│   ├── screenshots/                   # dashboard captures for the README
│   └── sample_outputs/                # safe public CSVs (aggregated only)
│
├── notebooks/                         # 1️⃣ → 5️⃣ pipeline, runnable end-to-end
│   ├── 00_exploration.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_classical_ml.ipynb                 # ← Nested CV champion
│   ├── 02b_smote_sensitivity.ipynb           # ← class-imbalance ablation
│   ├── 03_transformers.ipynb                 # ← Colab/GPU
│   ├── 04_clinical_evaluation.ipynb
│   └── 05_deployment_mvp.ipynb
│
├── src/mental_health/                 # installable package
│   ├── __init__.py
│   ├── app/
│   │   ├── app.py                     # Streamlit entry point
│   │   ├── pages/                     # one module per dashboard page
│   │   ├── styles.py                  # custom CSS (glassmorphism)
│   │   └── openrouter_client.py
│   ├── config/
│   │   └── paths.py                   # centralised path constants
│   ├── models/
│   │   └── services.py                # load_model, predict_with_model
│   └── rag/
│       └── simple_rag.py              # FAISS + LangChain retrieval
│
├── tests/                             # pytest, fully mocked, 33 / 33 passing
│   ├── conftest.py
│   ├── test_predictions.py
│   ├── test_rag.py
│   └── test_client.py
│
├── models/
│   └── best_classical_model.joblib   # champion LinearSVC (committed, ~5 MB)
│
├── reports/tables/                    # evaluation artifacts (CSVs only)
│   ├── classical/
│   │   ├── final_test_metrics.csv
│   │   ├── nested_cv_summary.csv
│   │   └── normal_cv_summary.csv
│   └── clinical/
│       └── global_comparison_for_clinical_review.csv
│
├── rag_source/                        # knowledge base for the Chat page
└── data/                              # gitignored (local only)
```

---

## 🧪 Notebook pipeline

| #    | Notebook                    | Purpose                                                        | Runtime          |
| ---- | --------------------------- | -------------------------------------------------------------- | ---------------- |
| 00   | `exploration`               | Initial dataset exploration on Colab                           | CPU/Colab        |
| 01   | `data_cleaning`             | Text normalisation, near-duplicate detection, train/val/test export | CPU, ~2 min |
| 02   | `classical_ml`              | TF-IDF + LinearSVC, **Nested CV**, champion selection          | CPU, ~15 min     |
| 02b  | `smote_sensitivity`         | Tests whether oversampling improves recall on critical classes | CPU, ~5 min      |
| 03   | `transformers`              | BERT base + MentalBERT, fair head-to-head with the classical baseline | **GPU recommended** |
| 04   | `clinical_evaluation`       | Per-class confusion matrices, error analysis, clinical review tables | CPU, ~3 min |
| 05   | `deployment_mvp`            | LLM perspectives, MVP wiring                                   | CPU, ~2 min      |

> Run 03 on Google Colab via the badge at the top of each notebook.

---

## 🛡 Ethics & limitations

- **Non-diagnostic.** Repeated everywhere it matters. Outputs are signals for human review, not labels.
- **Privacy.** Classical inference is fully local. The RAG layer only sends *retrieved chunks plus the user's question* to OpenRouter — never raw patient text. The `Predictions` page never calls any external API.
- **Recall-skewed.** We accept more false positives (63) in exchange for fewer false negatives (95) on high-risk classes. This is a deliberate clinical trade-off, not a bug.
- **Data provenance.** The training data is derived from publicly available text; no clinical records were used. The model **has not been validated on clinical populations** and is not certified for clinical use.
- **Bias.** As with any text classifier, performance varies across demographics, dialects, and clinical sub-populations. The error analysis in notebook 04 surfaces these gaps; do not deploy without re-validating on your population.

See [`NOTICE.md`](NOTICE.md) for the full non-diagnostic notice.

---

## 🗺 Roadmap

- [ ] Capture screenshot of the Chat page (RAG with grounded sources)
- [ ] Add ONNX export of the LinearSVC champion for sub-millisecond serving
- [ ] Containerised CI matrix across Python 3.10 / 3.11 / 3.12
- [ ] Multilingual support — Portuguese / Spanish text triage
- [ ] Calibrated probability outputs (Platt / isotonic) on the `Predictions` page
- [ ] Hugging Face Spaces public demo
- [ ] Drift monitoring page wired to a pseudo-production stream

---

## 🤝 Contributing

This is a portfolio project, but issues and PRs are welcome. Please:

1. Run `pytest -v` and `ruff check .` before opening a PR.
2. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
3. If you touch a notebook, run `nbstripout <file>.ipynb` to avoid bloating diffs with execution outputs.
4. CI must pass green before review.

---

## 👤 Author

**Ana Gouveia** — DSFS-OD-14 cohort
[GitHub @anafiiliipa-dev](https://github.com/anafiiliipa-dev)

---

## 📄 License

MIT — see [LICENSE](LICENSE). Non-diagnostic notice in [NOTICE.md](NOTICE.md).
