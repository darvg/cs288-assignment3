"""Thin LLM wrapper.

All model calls must route through this file. The default implementation
supports an OpenRouter-compatible HTTP API when the environment is configured.
If the environment is missing, callers should handle the raised error and
fallback gracefully.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


ALLOWED_MODELS = {
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
}


class LLMError(RuntimeError):
    """Raised when a model call cannot be completed."""


def generate(prompt: str, model_name: str, timeout_sec: int = 20) -> str:
    if model_name not in ALLOWED_MODELS:
        raise LLMError(f"Model not allowed: {model_name}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set")

    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.invalid",
            "X-Title": "cs288-a3-rag",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMError(str(exc)) from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Malformed LLM response") from exc
