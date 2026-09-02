"""
Extracteur de features par sentence-embedding compatible sklearn (Phase 11 :
"tester une solution intermediaire: embeddings sentence-transformers
(MiniLM deja utilise pour le RAG) + LogisticRegression/SVM").

Décision de conception (confirmée avec le propriétaire du projet) : contrairement
à TF-IDF, les embeddings MiniLM sont coûteux à calculer et proviennent d'un
modèle pré-entraîné et gelé -- il n'y a pas de fitting impliqué, donc les
recalculer à chaque fold de la nested-CV (comme TfidfVectorizer doit
légitimement le faire, puisque les poids IDF dépendent du fold
d'entraînement) serait du gaspillage pur, pas de la rigueur. Les embeddings de
tout le dataset sont donc précalculés UNE SEULE FOIS
(``precompute_dataset_embeddings``) avant l'exécution du benchmark, et
``EmbeddingVectorizer`` va les chercher dans ce cache partagé au lieu de
les recalculer à chaque appel fit/transform.

``sentence-transformers`` est importé paresseusement (dans les fonctions/méthodes,
jamais au moment de l'import du module) afin que l'import de ce module -- et
donc de ``model_registry.py``, qui l'importe sans condition -- ne requière
jamais que la dépendance (lourde, basée sur torch) soit installée.
Seuls les chemins de code qui construisent ou utilisent réellement un candidat
basé sur des embeddings ont besoin de l'extra ``embedding_models`` installé.
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
    Encode chaque texte distinct de ``texts`` une seule fois et retourne un
    cache ``{text: embedding_vector}``. Conçu pour être calculé une seule fois sur
    TOUT le dataset (lignes train + test) avant tout split de CV, afin que chaque
    recherche pendant le benchmark/l'entraînement/l'évaluation soit un cache hit.
    """
    from sentence_transformers import SentenceTransformer

    unique_texts = list(dict.fromkeys(texts))  # de-dup, préserve l'ordre
    model = SentenceTransformer(model_name)
    vectors = model.encode(unique_texts, show_progress_bar=False)
    return dict(zip(unique_texts, vectors, strict=True))


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Transforme du texte en embeddings sentence-transformer. ``fit`` est un no-op
    (le modèle sous-jacent est gelé/pré-entraîné -- il n'y a rien à
    apprendre du fold d'entraînement, contrairement aux poids IDF de TfidfVectorizer).

    ``embedding_cache``, lorsqu'il est fourni (un dict partagé, typiquement issu de
    ``precompute_dataset_embeddings``), est vérifié en premier ; tout texte pas
    déjà présent est encodé à la demande et ajouté au cache -- cela
    garde l'objet correct même pour du texte jamais vu (par exemple une vraie
    requête /predict d'un utilisateur en serving), pas seulement rapide pendant
    l'entraînement.
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
