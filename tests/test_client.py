"""Unit tests for the OpenRouter LLM client."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.openrouter_client import ask_llm, get_default_model


# ============================================================
# get_default_model
# ============================================================

class TestGetDefaultModel:
    def test_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
        assert get_default_model() == "anthropic/claude-3-haiku"

    def test_returns_default_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        model = get_default_model()
        assert isinstance(model, str)
        assert len(model) > 0


# ============================================================
# ask_llm
# ============================================================

def _mock_response(content: str):
    """Build a minimal mock matching the openai SDK response shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class TestAskLlm:
    def test_returns_string_on_success(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        with patch("src.app.openrouter_client.get_openrouter_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_response("Hello!")
            mock_factory.return_value = client

            result = ask_llm("Say hello")
        assert result == "Hello!"

    def test_missing_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            ask_llm("test")

    def test_system_prompt_is_included_in_messages(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        with patch("src.app.openrouter_client.get_openrouter_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_response("ok")
            mock_factory.return_value = client

            ask_llm("user prompt", system_prompt="You are a bot.")

            call_kwargs = client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            roles = [m["role"] for m in messages]
            assert "system" in roles
            assert "user" in roles

    def test_retries_on_rate_limit(self, monkeypatch):
        from openai import RateLimitError

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        with patch("src.app.openrouter_client.get_openrouter_client") as mock_factory:
            with patch("src.app.openrouter_client.time.sleep"):  # skip waits
                client = MagicMock()
                rate_limit_exc = RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body={},
                )
                client.chat.completions.create.side_effect = [
                    rate_limit_exc,
                    _mock_response("Retry succeeded"),
                ]
                mock_factory.return_value = client

                result = ask_llm("test retry")
        assert result == "Retry succeeded"
        assert client.chat.completions.create.call_count == 2

    def test_raises_after_max_retries(self, monkeypatch):
        from openai import RateLimitError

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        with patch("src.app.openrouter_client.get_openrouter_client") as mock_factory:
            with patch("src.app.openrouter_client.time.sleep"):
                client = MagicMock()
                exc = RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body={},
                )
                client.chat.completions.create.side_effect = exc
                mock_factory.return_value = client

                with pytest.raises(RuntimeError, match="failed after"):
                    ask_llm("test")
