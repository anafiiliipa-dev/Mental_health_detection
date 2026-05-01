# Architecture & Design Decisions

> **Audience:** technical reviewers, maintainers, and recruiters who want the *why* behind the *what*.

This document explains the design choices that shaped Mental Health Intelligence — what was considered, what was chosen, and what was rejected.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Data layer](#2-data-layer)
3. [Modelling layer — why LinearSVC, not BERT](#3-modelling-layer--why-linearsvc-not-bert)
4. [Evaluation strategy — why Nested CV](#4-evaluation-strategy--why-nested-cv)
5. [Optimisation target — why Critical Recall](#5-optimisation-target--why-critical-recall)
6. [RAG layer — design and trade-offs](#6-rag-layer--design-and-trade-offs)
7. [Application layer — Streamlit choices](#7-application-layer--streamlit-choices)
8. [Privacy boundaries](#8-privacy-boundaries)
9. [Test strategy](#9-test-strategy)
10. [Rejected alternatives](#10-rejected-alternatives)
11. [Known limitations](#11-known-limitations)

---

## 1. System overview

The system has **three independent reasoning paths** that share no state:

```text
         ┌─────────────────────────────────┐
         │   Streamlit Dashboard (UI)      │
         └──────────────┬──────────────────┘
                        │
   ┌────────────────────┼────────────────────┐
   ▼                    ▼                    ▼
[Local ML]         [RAG copilot]         [LLM fallback]
TF-IDF +           FAISS + LangChain     OpenRouter
LinearSVC          + OpenRouter          (no retrieval)
   │                    │                    │
   ▼                    ▼                    ▼
Prediction +     Grounded answer +      Best-effort
probabilities    cited sources         answer (rare)
```

Why three paths? Because their privacy and cost profiles differ:

- **Local ML** never leaves the host. Free. Deterministic.
- **RAG** sends only the user's question + retrieved chunks (project docs, never patient text) to the LLM. Cheap.
- **LLM fallback** sends only the question — used when retrieval fails, with a UX warning to the user.

The user sees this transparency in the `Chat` page footer.

---

## 2. Data layer

### Pipeline (notebook `01_data_cleaning.ipynb`)

1. Load raw text from public sources (no clinical records).
2. Strip URLs, mentions, hashtags, code blocks.
3. Lower-case + Unicode normalise (NFKC).
4. **Near-duplicate detection** via shingled MinHash to remove cross-class label leakage.
5. Stratified train / val / test split (`60 / 20 / 20`).
6. Persist cleaned splits to `data/processed/` (gitignored).

### Why no clinical records

- **Legal:** EU GDPR Article 9 special-category data requires an explicit lawful basis we don't have for a portfolio.
- **Ethical:** even with consent, releasing a model trained on real clinical text into a public repo creates downstream re-identification risk.
- **Practical:** the dataset is publicly reproducible — anyone can retrain.

### Why near-duplicate detection

Public mental-health corpora (Reddit-style sources) are notorious for cross-posting. Without dedup, the model overfits to copy-pasted phrases rather than learning the underlying linguistic signal. **MinHash with `num_perm=128` and Jaccard threshold `0.8`** strikes the right balance between recall and removal of near-duplicates.

---

## 3. Modelling layer — why LinearSVC, not BERT

### What we tried

| Family             | Champion within family             | Notes                                      |
| ------------------ | ---------------------------------- | ------------------------------------------ |
| Linear classifiers | **LinearSVC + class_weight=balanced** | Champion, sparse TF-IDF (1–2 grams)     |
| Tree ensembles     | XGBoost on TF-IDF                  | Slightly worse macro-F1, much slower fit  |
| Transformers       | MentalBERT (domain-pretrained)     | Marginal F1 gain, ~40× inference cost     |
| Naive baselines    | LogReg, Multinomial NB             | Strong sanity floors; documented for context |

### Why LinearSVC won

| Criterion          | Why it matters in clinical-adjacent NLP | LinearSVC | MentalBERT |
| ------------------ | --------------------------------------- | --------- | ---------- |
| **Macro-F1**       | Headline accuracy across classes        | High      | Marginally higher |
| **Critical-class recall** | Asymmetric clinical cost         | High      | Slightly higher |
| **Inference cost** | Throughput at triage scale              | `~1×`     | `~40×`     |
| **Interpretability** | Auditability for clinicians            | Coefficients per token (directly readable) | Black-box attention |
| **Memory**         | Edge / restricted environments          | `~10 MB`  | `~440 MB`  |
| **Bias auditability** | Slicing performance by sub-population | Trivial   | Hard       |

The decision is **not** "transformers don't work" — they do. It's that for a triage tool that must be **defensible to a clinician** (and explainable to a regulator), a linear model with token-level coefficients wins on every dimension that matters in deployment.

### Hyperparameters

Settled by `GridSearchCV` inside the **inner** loop of nested CV (see §4). Configuration space:

```python
{
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df":      [2, 5],
    "tfidf__max_df":      [0.9, 0.95],
    "tfidf__sublinear_tf": [True, False],
    "svc__C":             [0.1, 1.0, 10.0],
}
```

`class_weight="balanced"` is fixed (it's a clinical requirement, not a hyperparameter to tune away).

---

## 4. Evaluation strategy — why Nested CV

Most ML benchmarks suffer from **selection bias** — hyperparameters tuned on the same folds used to report accuracy. The reported number then reflects "how well the model fits the validation noise" as much as "how well the model generalises".

### Nested cross-validation eliminates that bias

```text
Outer K-Fold (test estimate)
  └── Inner K-Fold (GridSearch on train fold only)
       └── Champion config promoted, refit on full train fold,
           scored on held-out test fold
```

- **Outer loop:** the test fold is **never seen** during hyperparameter selection.
- **Inner loop:** GridSearch operates only on train data of the outer fold.
- The reported metric is the **mean across outer test folds** — an unbiased estimate of generalisation error.

### What that costs us

- **Compute:** roughly `K_outer × K_inner` model fits. With `K=5` and a 24-cell grid, that's `~600` fits. ~15 minutes on CPU for this dataset.
- **Code complexity:** more careful pipeline construction so that vectoriser fitting respects the fold boundary.

Both are worth it, because the alternative — reporting an inflated number — undermines every other claim in the project.

---

## 5. Optimisation target — why Critical Recall

Macro-F1 is the standard metric for imbalanced multiclass NLP. We deliberately **don't** optimise it alone.

### The clinical reality

| If we miss a... | Downstream cost                                                |
| --------------- | -------------------------------------------------------------- |
| Bipolar signal  | Patient may be misdiagnosed with unipolar depression and prescribed an SSRI alone, which can trigger a manic episode. |
| Schizophrenia signal | Delayed antipsychotic intervention has well-documented impact on long-term outcomes. |
| Anxiety routed to Depression | Both treatable; substantial therapy overlap; recoverable downstream. |

The cost matrix is **asymmetric**. Macro-F1 treats it as symmetric.

### The composite metric

```python
critical_recall = mean([recall["Bipolar"], recall["Schizophrenia"]])
macro_f1        = sklearn.metrics.f1_score(..., average="macro")

champion_score = 0.6 * critical_recall + 0.4 * macro_f1
```

The weighting is conservative — we still respect overall classification quality, but we tilt toward catching what matters most. Both numbers are reported separately in `final_test_metrics.csv`, so reviewers can apply their own preferences.

---

## 6. RAG layer — design and trade-offs

### Stack

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — small (80 MB), fast, MIT-licensed, well-validated on STS benchmarks.
- **Vector store:** FAISS (CPU). `IndexFlatL2` for the project-scale corpus; the simplicity is a feature.
- **Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=100`. Tuned empirically against the corpus structure.
- **Generator:** OpenRouter `gpt-4o-mini` — cheapest competent model with structured output reliability.
- **Caching:** corpus fingerprint (SHA-256 of concatenated source files) is stored alongside the FAISS index; rebuild only on cache miss.

### Why not pgvector / Qdrant / Pinecone

- **Operational simplicity:** FAISS is a file. No service to deploy, no auth, no SLA.
- **Local privacy:** the index lives next to the app; no out-of-process data movement.
- **Scale:** for a project corpus measured in hundreds of chunks, switching to a vector DB would be over-engineering.

If the corpus grew to `>100k` chunks, the migration target would be **Qdrant** (self-hostable, supports filters, good Python client).

### Retrieval-miss fallback

If the nearest-neighbour distance is above a configurable threshold (no useful context retrieved), the chain falls back to a **direct LLM call without retrieval**, with a UI banner: *"No matching project docs found. Answering from the model's own knowledge."* This makes the failure mode visible to the user instead of letting the LLM hallucinate against an empty context.

---

## 7. Application layer — Streamlit choices

### Why Streamlit

- **Speed:** dashboard from zero in days, not weeks.
- **Audience fit:** data-science demonstrators expect Streamlit; clinical reviewers can use it without training.
- **State management:** `st.session_state` is enough for our six pages; no Redux / Zustand burden.

### Why a glassmorphism theme

- **Differentiation:** 90% of Streamlit demos look identical. Custom CSS sets the project apart in a portfolio context.
- **Cognitive load:** translucent panels separate sections without hard borders, reducing visual noise.
- **Brand cohesion:** the design language (gradient accents, soft shadows) carries through every page.

### Page boundaries

Each page is a **separate module under `src/mental_health/app/pages/`**. They do not share mutable state with each other beyond the explicit `st.session_state` keys for prediction history. This keeps the navigation deterministic and easy to test.

### What we avoid

- **No client-side persistence.** History clears on browser refresh — by design, to honour the privacy framing.
- **No background jobs.** Streamlit is request-response. Anything async (e.g., long retraining) belongs in the notebooks.

---

## 8. Privacy boundaries

| Layer                       | Data that crosses the boundary                                | Where                              |
| --------------------------- | ------------------------------------------------------------- | ---------------------------------- |
| Browser → Streamlit server  | The user's input text                                         | Localhost in the default deployment |
| Streamlit server → ML model | The user's input text                                         | In-process, never leaves host      |
| Streamlit server → OpenRouter | **Only** RAG-retrieved chunks + the user's chat question     | Outbound HTTPS                     |
| Streamlit server → OpenRouter | **Never** raw input from the `Predictions` page               | Hard-coded — see `app/pages/predictions.py` |

The `Predictions` page **does not** import `openrouter_client`. This is enforced by module structure, not by convention — a developer who wanted to leak prediction text via the LLM would have to add an import, which a reviewer would notice immediately.

---

## 9. Test strategy

26 tests across three files, all using mocks for external services:

| File                  | Coverage                                       |
| --------------------- | ---------------------------------------------- |
| `test_predictions.py` | Demo fallback path; model-loaded path; edge cases (empty input, only punctuation, unicode) |
| `test_rag.py`         | Chain construction; retrieval-miss fallback; corpus-fingerprint cache hit/miss             |
| `test_client.py`      | OpenRouter request shape; error handling; retry behaviour                                  |

### What we do **not** test

- The Streamlit UI itself. End-to-end browser tests with Playwright would be the next step but are out of scope for the portfolio version.
- Notebook execution. The notebooks are reproducible reference material; they're not a CI target.

---

## 10. Rejected alternatives

| Considered                      | Why rejected                                                 |
| ------------------------------- | ------------------------------------------------------------ |
| FastAPI backend + React frontend | Quadruples surface area for marginal UX gain at portfolio scale |
| Multi-label classification      | Empirically the dataset's labels are mutually exclusive in `>95%` of cases; multi-label adds noise without measurable benefit |
| pgvector instead of FAISS       | Operational overhead unjustified at this scale (see §6)      |
| Hugging Face Inference API for the champion | Latency variance and cost; LinearSVC is so cheap to host locally that this is a regression |
| End-to-end transformer fine-tuning as the champion | Marginal F1, big interpretability and cost regression — see §3 |
| Storing the trained model in Git LFS | Fragile, costly, and against the principle that this repo demonstrates the *recipe*, not ships a binary |

---

## 11. Known limitations

- **No clinical validation.** See [`NOTICE.md`](../NOTICE.md).
- **English-only.** The MiniLM embedding model is multilingual-capable but the training data and evaluation are English-only.
- **Static dataset.** No drift detection. The Monitoring page shows the latest evaluation run, not a live production stream.
- **No calibrated probabilities** on the `Predictions` page yet — listed in the Roadmap.
- **No per-user authentication.** The app assumes a single trusted user (the operator). Multi-tenancy would require non-trivial changes to session state and history.

---

*Last updated: 2026-05-01*
