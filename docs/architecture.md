# Architecture & Design Decisions

This document captures the *why* behind the choices in the codebase. It's
intended for reviewers, future contributors, and the authors' own sanity
six months from now.

---

## 1. Why a classical baseline as the production model?

The transformer benchmarks (`03_transformers_benchmark.ipynb`) showed:

| Model | Macro F1 | Inference cost | Interpretable? |
|---|:-:|:-:|:-:|
| LinearSVC + TF-IDF | 0.779 | <1 ms (CPU) | YES (word-level coefficients) |
| BERT base | 0.791 | ~50 ms (GPU) | NO |
| MentalBERT | 0.809 | ~50 ms (GPU) | NO |

The transformer wins by ~3 F1 points. For a clinical decision-support tool,
that's not enough to justify:

1. ~40x the inference cost,
2. losing direct interpretability (a clinician can read the coefficient
   table and reason about *why* a sample was flagged), and
3. needing a GPU in deployment.

LinearSVC stays the champion. The transformer notebooks are kept as a
reproducibility artefact and a starting point if the inference budget ever
expands.

---

## 2. Why Nested Cross-Validation?

Most ML projects tune hyperparameters with `GridSearchCV` and report the
best score from the search. That score is **biased upward** because it has
already seen every fold during model selection.

Nested CV separates the two roles cleanly:

    Outer K-Fold        <- reports unbiased generalisation
    +- Inner K-Fold     <- used only for hyperparameter selection
       +- refit on full outer-train fold, score on outer-test fold

The `nested_cv_summary.csv` artefact reports the outer-fold means with
standard deviations. This is the number we trust when comparing the
classical baseline to the transformers.

---

## 3. Why optimise for *Critical Recall*?

In a triage setting, errors are not symmetric:

- **False negative on Bipolar/Schizophrenia** -> high-risk case missed.
- **False positive on Anxiety** -> reviewer spends an extra minute checking.

We weight the metric accordingly. The model is allowed to be slightly less
precise on the rarer high-risk classes if that buys us higher recall.

The SMOTE ablation (`02b_smote_sensitivity_analysis.ipynb`) showed that
class-weighted training (`class_weight='balanced'`) reaches the same
critical recall as SMOTE oversampling without the extra complexity. We
keep the simpler approach.

---

## 4. Why three reasoning paths in the dashboard?

| Path | Privacy boundary |
|---|---|
| Local ML inference (`Predictions`) | Runs entirely offline. No text leaves the host. |
| RAG over project docs (`Chat`) | Embeddings local; only retrieved chunks + question sent to OpenRouter. |
| Direct LLM fallback | Question sent to OpenRouter only when retrieval fails. |

This split lets us route patient-style text exclusively through the local
classifier, while reserving the LLM for project-explanation Q&A.

The `Predictions` page **never** calls the LLM. That is a hard guarantee
enforced by module separation.

---

## 5. Why FAISS with a fingerprinted cache?

The original RAG prototype rebuilt the FAISS index from scratch on every
Streamlit reload (~30 s). That is painful in development and looks broken
in a demo.

The improved RAG layer:

1. Hashes the contents of `rag_source/` (path + mtime + size, SHA1).
2. Saves the index under `.cache/faiss/<fingerprint>/`.
3. Reuses it on subsequent loads if the fingerprint matches.

Adding or editing a doc in `rag_source/` invalidates the cache
automatically — no manual cleanup.

---

## 6. Trade-offs we accepted

- **No real-time monitoring.** The Monitoring page reads static CSVs from
  the last evaluation run. Adding live drift detection is in the roadmap.
- **No authentication.** This is a portfolio dashboard, not a production
  system. Don't deploy to public internet without an auth proxy.
- **English-only.** The dataset is English. Performance on translated
  text is unknown and unvalidated.
