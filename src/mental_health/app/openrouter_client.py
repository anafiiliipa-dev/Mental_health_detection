"""
Client LLM OpenRouter avec timeout et retry en backoff exponentiel.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

load_dotenv()

# ============================================================
# Constantes
# ============================================================

_DEFAULT_MODEL = "openai/gpt-4o-mini"
_MAX_TOKENS = 1_500
_TEMPERATURE = 0.2
_TIMEOUT = 30.0          # secondes par requête
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0      # secondes ; double à chaque nouvelle tentative


# ============================================================
# Fabrique de client
# ============================================================

def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Add it to your .env file or environment variables."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=_TIMEOUT,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-OpenRouter-Title": "mental-health-intelligence",
        },
    )


def get_default_model() -> str:
    return os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)


# ============================================================
# API publique
# ============================================================

def ask_llm(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = _MAX_TOKENS,
) -> str:
    """
    Envoie un prompt au LLM configuré via OpenRouter.

    Retente jusqu'à _MAX_RETRIES fois avec un backoff exponentiel en cas
    d'erreurs de connexion et de rate limits. Lève une exception en cas
    d'échec persistant.
    """
    client = get_openrouter_client()
    model_name = get_default_model()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=_TEMPERATURE,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        except RateLimitError as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            time.sleep(wait)

        except APIConnectionError as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            time.sleep(wait)

        except APIStatusError as exc:
            # Les erreurs 4xx (sauf 429) ne sont pas réessayables
            raise exc

    raise RuntimeError(
        f"OpenRouter request failed after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )
