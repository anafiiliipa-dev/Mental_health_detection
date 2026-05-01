"""Unit tests for the RAG pipeline."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.rag.simple_rag import (
    SimpleLocalRAG,
    _source_fingerprint,
    load_documents,
)

# ============================================================
# load_documents
# ============================================================

class TestLoadDocuments:
    def test_returns_empty_list_when_dir_missing(self):
        with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", Path("/nonexistent/path")):
            docs = load_documents()
        assert docs == []

    def test_loads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "test.txt").write_text("Hello from test file", encoding="utf-8")
            with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", tmp):
                docs = load_documents()
        assert len(docs) >= 1
        contents = " ".join(d.page_content for d in docs)
        assert "Hello from test file" in contents

    def test_loads_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "notes.md").write_text("# Title\nSome markdown content", encoding="utf-8")
            with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", tmp):
                docs = load_documents()
        assert len(docs) >= 1


# ============================================================
# _source_fingerprint
# ============================================================

class TestSourceFingerprint:
    def test_returns_empty_string_when_dir_missing(self):
        with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", Path("/nonexistent")):
            fp = _source_fingerprint()
        assert fp == ""

    def test_fingerprint_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            f = tmp / "doc.txt"
            f.write_text("original content", encoding="utf-8")
            with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", tmp):
                fp1 = _source_fingerprint()
            f.write_text("modified content", encoding="utf-8")
            with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", tmp):
                fp2 = _source_fingerprint()
        assert fp1 != fp2

    def test_same_content_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "doc.txt").write_text("stable content", encoding="utf-8")
            with patch("mental_health.rag.simple_rag.RAG_SOURCE_DIR", tmp):
                fp1 = _source_fingerprint()
                fp2 = _source_fingerprint()
        assert fp1 == fp2


# ============================================================
# SimpleLocalRAG.invoke
# ============================================================

class TestSimpleLocalRAG:
    def _make_rag(self, docs=None):
        """Build a SimpleLocalRAG with a mocked vectorstore."""
        if docs is None:
            docs = [
                MagicMock(
                    page_content="The project uses Nested CV for robust evaluation.",
                    metadata={"source": "project.txt"},
                )
            ]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        return SimpleLocalRAG(vectorstore=mock_vs)

    def test_empty_query_returns_prompt(self):
        rag = self._make_rag()
        result = rag.invoke({"query": ""})

        assert "Please provide a question" in result["result"]
        assert result["source_documents"] == []

    def test_query_with_only_spaces_returns_prompt(self):
        rag = self._make_rag()
        result = rag.invoke({"query": "   "})

        assert "Please provide a question" in result["result"]
        assert result["source_documents"] == []

    def test_returns_result_and_source_docs(self):
        rag = self._make_rag()
        with patch("mental_health.app.openrouter_client.ask_llm", return_value="Nested CV eliminates selection bias."):
            result = rag.invoke({"query": "What validation method is used?"})

        assert "result" in result
        assert "source_documents" in result

    def test_source_documents_preserve_metadata(self):
        doc = MagicMock(
            page_content="The project uses Nested CV for robust evaluation.",
            metadata={"source": "project.txt"},
        )
        rag = self._make_rag(docs=[doc])

        with patch("mental_health.app.openrouter_client.ask_llm", return_value="Answer."):
            result = rag.invoke({"query": "What validation method is used?"})

        assert result["source_documents"][0].metadata["source"] == "project.txt"

    def test_no_docs_found_returns_fallback_message(self):
        rag = self._make_rag(docs=[])
        result = rag.invoke({"query": "What is the meaning of life?"})

        assert "could not find" in result["result"].lower()

    def test_llm_is_called_with_context(self):
        rag = self._make_rag()
        with patch("mental_health.app.openrouter_client.ask_llm") as mock_llm:
            mock_llm.return_value = "Generated answer."
            result = rag.invoke({"query": "Explain the model"})

        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        prompt_arg = call_kwargs[1].get("prompt") or call_kwargs[0][0]
        assert "Nested CV" in prompt_arg
