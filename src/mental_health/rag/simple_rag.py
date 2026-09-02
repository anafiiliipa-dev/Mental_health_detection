"""
Pipeline RAG (Retrieval-Augmented Generation).

Architecture :
    1. Charge les documents .txt / .md depuis rag_source/
    2. Les découpe en chunks et les embed avec un modèle sentence-transformers local
    3. Persiste l'index FAISS sur disque (faiss_index/) — reconstruit
       uniquement quand les fichiers source changent ou que l'index est absent
    4. À la requête : récupère les top-k chunks, construit un prompt ancré
       (grounded), appelle le LLM via OpenRouter, retourne la réponse générée
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mental_health.config.paths import RAG_INDEX_DIR, RAG_SOURCE_DIR

# ============================================================
# Constantes
# ============================================================

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120
_TOP_K = 4
_INDEX_MANIFEST = RAG_INDEX_DIR / "manifest.json"


# ============================================================
# Chargement des documents
# ============================================================

def load_documents() -> list[Any]:
    """Charge tous les fichiers .txt et .md depuis le répertoire source RAG."""
    docs: list[Any] = []
    if not RAG_SOURCE_DIR.exists():
        return docs
    for pattern in ("**/*.txt", "**/*.md"):
        for file in RAG_SOURCE_DIR.glob(pattern):
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def _source_fingerprint() -> str:
    """Retourne un hash du contenu de tous les fichiers source pour détecter les changements."""
    h = hashlib.md5()
    if not RAG_SOURCE_DIR.exists():
        return ""
    for f in sorted(RAG_SOURCE_DIR.glob("**/*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def _load_manifest() -> dict:
    if _INDEX_MANIFEST.exists():
        return json.loads(_INDEX_MANIFEST.read_text())
    return {}


def _save_manifest(fingerprint: str) -> None:
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX_MANIFEST.write_text(json.dumps({"fingerprint": fingerprint}))


# ============================================================
# Base vectorielle
# ============================================================

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)


def _build_vectorstore(docs: list[Any]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def _load_or_build_vectorstore() -> FAISS | None:
    """
    Retourne une base vectorielle FAISS, en la chargeant depuis le disque quand possible.
    Reconstruit (et persiste) quand les fichiers source ont changé.
    """
    docs = load_documents()
    if not docs:
        return None

    fingerprint = _source_fingerprint()
    manifest = _load_manifest()

    embeddings = _get_embeddings()

    # Tente de charger depuis le disque si le fingerprint correspond
    if (
        manifest.get("fingerprint") == fingerprint
        and RAG_INDEX_DIR.exists()
        and (RAG_INDEX_DIR / "index.faiss").exists()
    ):
        try:
            return FAISS.load_local(
                str(RAG_INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass  # Se rabat sur la reconstruction

    # Construit à partir de zéro et persiste
    vectorstore = _build_vectorstore(docs)
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(RAG_INDEX_DIR))
    _save_manifest(fingerprint)
    return vectorstore


# ============================================================
# Chaîne RAG
# ============================================================

@dataclass
class SimpleLocalRAG:
    """
    Chaîne RAG retrieve-then-generate.

    Récupère les chunks pertinents depuis l'index FAISS local,
    construit un prompt ancré (grounded), et appelle le LLM via OpenRouter.
    """
    vectorstore: Any
    _system_prompt: str = field(default=(
        "You are a careful assistant for a mental health NLP project. "
        "Use ONLY the provided context to answer the question. "
        "If the context does not contain the answer, say so clearly. "
        "Do not present any output as a medical diagnosis. "
        "Be concise, accurate, and professional."
    ), init=False, repr=False)

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        from mental_health.app.openrouter_client import (
            ask_llm,  # import local pour éviter les dépendances circulaires
        )

        query = inputs.get("query", "").strip()
        if not query:
            return {"result": "Please provide a question.", "source_documents": []}

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": _TOP_K})
        source_documents = retriever.invoke(query)

        if not source_documents:
            return {
                "result": "I could not find relevant information in the local knowledge base.",
                "source_documents": [],
            }

        context_blocks = [
            f"[Source {i}] {doc.page_content.strip().replace(chr(10), ' ')}"
            for i, doc in enumerate(source_documents, start=1)
        ]
        context_text = "\n\n".join(context_blocks)

        prompt = (
            f"Context from project documentation:\n\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above:"
        )

        answer = ask_llm(prompt=prompt, system_prompt=self._system_prompt)

        return {"result": answer, "source_documents": source_documents}


def build_qa_chain() -> SimpleLocalRAG | None:
    """Construit et retourne un SimpleLocalRAG, ou None si aucun document n'existe."""
    vectorstore = _load_or_build_vectorstore()
    if vectorstore is None:
        return None
    return SimpleLocalRAG(vectorstore=vectorstore)
