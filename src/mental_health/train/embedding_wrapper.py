"""
sklearn-compatible sentence-embedding feature extractor (Phase 11:
"tester une solution intermediaire: embeddings sentence-transformers
(MiniLM deja utilise pour le RAG) + LogisticRegression/SVM").

Design decision (confirmed with the project owner): unlike TF-IDF, MiniLM
embeddings are expensive to compute and come from a frozen, pretrained
model -- there is no fitting involved, so recomputing them inside every
nested-CV fold (as TfidfVectorizer legitimately needs to, since IDF
weights depend on the training fold) would be pure waste, not rigor. The
whole dataset's embeddings are therefore precomputed ONCE
(``precompute_dataset_embeddings``) before the benchmark runs, and
``EmbeddingVectorizer`` looks them up from that shared cache instead of
re-encoding on every fit/transform call.

``sentence-transformers`` is imported lazily (inside functions/methods,
never at module import time) so that importing this module -- and
therefore ``model_registry.py``, which imports it unconditionally --
never requires the (heavy, torch-backed) dependency to be installed.
Only code paths that actually build or use an embedding-based candidate
need the ``embedding_models`` extra installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def precompute_dataset_embeddings(
    texts: pd.Series, model_name: str = DEFAULT_EMBEDDING_MODEL
) -> dict[str, np.ndarray]:
    """
    Encode every distinct text in ``texts`` once and return a
    ``{text: embedding_vector}`` cache. Meant to be computed once over the
    FULL dataset (train + test rows) before any CV split, so every lookup
    during benchmarking/training/evaluation is a cache hit.
    """
    from sentence_transformers import SentenceTransformer

    unique_texts = list(dict.fromkeys(texts))  # de-dup, preserves order
    model = SentenceTransformer(model_name)
    vectors = model.encode(unique_texts, show_progress_bar=False)
    return dict(zip(unique_texts, vectors, strict=True))


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Turns text into sentence-transformer embeddings. ``fit`` is a no-op
    (the underlying model is frozen/pretrained -- there is nothing to
    learn from the training fold, unlike TfidfVectorizer's IDF weights).

    ``embedding_cache``, when provided (a shared dict, typically from
    ``precompute_dataset_embeddings``), is checked first; any text not
    already in it is encoded on demand and added to the cache -- this
    keeps the object correct even for text it has never seen (e.g. a real
    user's /predict request at serving time), not just fast during
    training.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, embedding_cache: dict[str, np.ndarray] | None = None):
        self.model_name = model_name
        self.embedding_cache = embedding_cache

    def _get_model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        texts = list(X)
        cache = self.embedding_cache if self.embedding_cache is not None else {}

        missing = [t for t in texts if t not in cache]
        if missing:
            vectors = self._get_model().encode(missing, show_progress_bar=False)
            for text, vector in zip(missing, vectors, strict=True):
                cache[text] = vector

        return np.asarray([cache[t] for t in texts])
