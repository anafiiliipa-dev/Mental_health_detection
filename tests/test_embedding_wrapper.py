"""
Unit tests for src/mental_health/train/embedding_wrapper.py.

The real sentence-transformers model is never downloaded here: every test
monkeypatches EmbeddingVectorizer._get_model (or, for
precompute_dataset_embeddings, the SentenceTransformer import site) with a
tiny deterministic fake encoder, so the suite runs fast, offline, and
without the heavy optional dependency installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from mental_health.train.embedding_wrapper import EmbeddingVectorizer, precompute_dataset_embeddings

EMBEDDING_DIM = 4


class _FakeEncoder:
    """Deterministic, hash-based fake standing in for SentenceTransformer."""

    def encode(self, texts, show_progress_bar=False):
        return np.array([self._vector(text) for text in texts])

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(EMBEDDING_DIM)


class TestEmbeddingVectorizer:
    def test_transform_returns_one_vector_per_row(self, monkeypatch):
        texts = pd.Series(["i feel anxious", "racing thoughts", "hearing voices"])
        vectorizer = EmbeddingVectorizer()
        monkeypatch.setattr(vectorizer, "_get_model", lambda: _FakeEncoder())

        vectors = vectorizer.fit(texts).transform(texts)

        assert vectors.shape == (3, EMBEDDING_DIM)

    def test_transform_is_deterministic_for_the_same_text(self, monkeypatch):
        texts = pd.Series(["i feel anxious", "i feel anxious"])
        vectorizer = EmbeddingVectorizer()
        monkeypatch.setattr(vectorizer, "_get_model", lambda: _FakeEncoder())

        vectors = vectorizer.transform(texts)

        assert np.array_equal(vectors[0], vectors[1])

    def test_uses_precomputed_cache_without_calling_the_model(self, monkeypatch):
        texts = pd.Series(["i feel anxious", "racing thoughts"])
        cache = {text: np.full(EMBEDDING_DIM, i) for i, text in enumerate(texts)}
        vectorizer = EmbeddingVectorizer(embedding_cache=cache)

        def _boom():
            raise AssertionError("should not need the real/fake model — everything is cached")

        monkeypatch.setattr(vectorizer, "_get_model", _boom)

        vectors = vectorizer.transform(texts)

        assert np.array_equal(vectors[0], cache[texts.iloc[0]])
        assert np.array_equal(vectors[1], cache[texts.iloc[1]])

    def test_falls_back_to_the_model_for_texts_missing_from_the_cache(self, monkeypatch):
        texts = pd.Series(["cached text", "new text"])
        cache = {"cached text": np.zeros(EMBEDDING_DIM)}
        vectorizer = EmbeddingVectorizer(embedding_cache=cache)
        monkeypatch.setattr(vectorizer, "_get_model", lambda: _FakeEncoder())

        vectors = vectorizer.transform(texts)

        assert np.array_equal(vectors[0], np.zeros(EMBEDDING_DIM))
        assert "new text" in cache  # the cache dict is populated in place

    def test_is_clonable_and_set_params_works(self):
        # Required for benchmark.py's CV loops, which clone() + set_params()
        # every candidate on every fold.
        vectorizer = EmbeddingVectorizer(model_name="all-MiniLM-L6-v2")
        cloned = clone(vectorizer)
        cloned.set_params(model_name="some-other-model")
        assert cloned.model_name == "some-other-model"

    def test_fit_returns_self_without_touching_the_model(self, monkeypatch):
        vectorizer = EmbeddingVectorizer()
        monkeypatch.setattr(
            vectorizer, "_get_model", lambda: (_ for _ in ()).throw(AssertionError("fit must not load the model"))
        )
        assert vectorizer.fit(pd.Series(["irrelevant"])) is vectorizer


class TestPrecomputeDatasetEmbeddings:
    @staticmethod
    def _install_fake_sentence_transformers_module(monkeypatch):
        # precompute_dataset_embeddings does `from sentence_transformers
        # import SentenceTransformer` INSIDE the function (lazy import, by
        # design, so the heavy dependency is opt-in) -- the only reliable
        # way to intercept that from a test is to install a fake module
        # under that name in sys.modules before the function runs.
        import sys
        import types

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = lambda model_name: _FakeEncoder()
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    def test_returns_one_vector_per_unique_text(self, monkeypatch):
        self._install_fake_sentence_transformers_module(monkeypatch)
        texts = pd.Series(["a", "b", "a", "c"])

        cache = precompute_dataset_embeddings(texts)

        assert set(cache.keys()) == {"a", "b", "c"}
        for vector in cache.values():
            assert vector.shape == (EMBEDDING_DIM,)

    def test_deduplicates_before_encoding(self, monkeypatch):
        self._install_fake_sentence_transformers_module(monkeypatch)
        texts = pd.Series(["a", "a", "a"])

        cache = precompute_dataset_embeddings(texts)

        assert list(cache.keys()) == ["a"]


class TestEmbeddingVectorizerWithPrecomputedCache:
    def test_end_to_end_precompute_then_transform(self, monkeypatch):
        # Mirrors how train.py uses this in practice: precompute once for
        # the whole dataset, then have every fold's EmbeddingVectorizer
        # (constructed fresh via clone()) look the same texts up in that
        # shared cache instead of re-encoding them.
        TestPrecomputeDatasetEmbeddings._install_fake_sentence_transformers_module(monkeypatch)
        texts = pd.Series(["i feel anxious", "racing thoughts", "hearing voices"])

        cache = precompute_dataset_embeddings(texts)
        vectorizer = EmbeddingVectorizer(embedding_cache=cache)
        monkeypatch.setattr(
            vectorizer, "_get_model", lambda: (_ for _ in ()).throw(AssertionError("should use the cache"))
        )

        vectors = vectorizer.transform(texts)

        assert vectors.shape == (3, EMBEDDING_DIM)
