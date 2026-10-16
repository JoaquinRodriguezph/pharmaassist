"""Ollama backend: translates LLMClient calls to and from the `ollama` package."""

from __future__ import annotations

import os
import time

import ollama

from app.llm.base import LLMClient, LLMConnectionError, LLMResponse, Message


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set (see .env.example).")
    return value


class OllamaClient(LLMClient):
    """Calls a local Ollama server. Model and host come from the environment
    (OLLAMA_MODEL, OLLAMA_HOST) unless passed explicitly."""

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or _require_env("OLLAMA_MODEL")
        self.host = host or _require_env("OLLAMA_HOST")
        self._client = ollama.Client(host=self.host)

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Ollama takes the system prompt as the first message.
        ollama_messages = list(messages)
        if system:
            ollama_messages.insert(0, {"role": "system", "content": system})

        options: dict[str, float | int] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        start = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.model, messages=ollama_messages, options=options
            )
        except ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not reach the Ollama server at {self.host}. "
                "Is it running? Start it (or check OLLAMA_HOST) and try again."
            ) from exc
        latency = time.perf_counter() - start

        return LLMResponse(
            text=response.message.content or "",
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            latency_seconds=latency,
            model=response.model or self.model,
        )