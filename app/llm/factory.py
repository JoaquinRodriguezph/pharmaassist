"""Pick an LLM backend from the LLM_BACKEND environment variable.

Each backend is imported inside its own branch, so choosing `ollama` never
"""

from __future__ import annotations

import os

from app.llm.base import LLMClient

BACKENDS = ("ollama", "gemini")


def create_llm_client() -> LLMClient:
    """Build the client named by LLM_BACKEND (case-insensitive).

    Raises ValueError if it is missing or not one of BACKENDS.
    """
    raw = os.environ.get("LLM_BACKEND", "")
    backend = raw.strip().lower()
    allowed = ", ".join(BACKENDS)

    if not backend:
        raise ValueError(f"LLM_BACKEND is not set. Allowed values: {allowed}.")

    if backend == "ollama":
        from app.llm.ollama_client import OllamaClient

        return OllamaClient()

    if backend == "gemini":
        from app.llm.gemini_client import GeminiClient

        return GeminiClient()

    raise ValueError(f"Unknown LLM_BACKEND {raw!r}. Allowed values: {allowed}.")