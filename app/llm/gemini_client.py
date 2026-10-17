"""Gemini backend: translates LLMClient calls to and from the `google-genai` SDK.

Checked against the SDK docs (googleapis.github.io/python-genai): the system
prompt, temperature and output-token limit go in
`config=types.GenerateContentConfig(system_instruction=..., temperature=...,
max_output_tokens=...)`; `contents` is a list of `types.Content`; and the
assistant's role is "model", not "assistant". Retries use call_with_retry with
classify_gemini_error; after retries, SDK errors callers should not need to
know about are translated to LLMError subclasses.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.llm.base import (
    LLMClient,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponse,
    Message,
)
from app.llm.env import require_env
from app.llm.gemini_errors import classify_gemini_error, is_daily_quota, parse_retry_delay
from app.llm.retry import call_with_retry

# Neutral role -> Gemini role.
_ROLES = {"user": "user", "assistant": "model"}


class GeminiClient(LLMClient):
    """Calls the Gemini API. With no arguments, reads LLM_API_KEY and LLM_MODEL
    from the environment.

    `client` (anything with `.models.generate_content(...)`), `sleep` and
    `clock` can be injected so tests run without the network or real waiting.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        max_attempts: int = 4,
    ) -> None:
        self.model = model or require_env("LLM_MODEL")
        if client is None:
            client = genai.Client(api_key=api_key or require_env("LLM_API_KEY"))
        self._client = client
        self._sleep = sleep
        self._clock = clock
        self.max_attempts = max_attempts

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate the assistant's next reply.

        Token mapping: prompt_tokens <- prompt_token_count, completion_tokens
        <- candidates_token_count (visible answer only), reasoning_tokens <-
        thoughts_token_count (None when the model reports no thinking). See
        LLMResponse for why thinking tokens are kept separate.

        latency_seconds is the duration of the successful attempt only; time
        spent in retry back-off is not included.

        Caution: Google's docs for its newest model say thinking tokens count
        toward max_output_tokens (a third-party article says the same for
        earlier thinking models), so a small `max_tokens` can leave little or
        no room for the visible answer. Not verified for the model in use.
        """
        contents = [
            types.Content(
                role=_to_gemini_role(m["role"]),
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        config = types.GenerateContentConfig(**config_kwargs)

        attempt_seconds = 0.0

        def attempt():
            nonlocal attempt_seconds
            start = self._clock()
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            attempt_seconds = self._clock() - start
            return response

        try:
            response = call_with_retry(
                attempt,
                classify_gemini_error,
                max_attempts=self.max_attempts,
                sleep=self._sleep,
            )
        except APIError as exc:
            if exc.code == 404:
                raise LLMModelNotFoundError(
                    f"Gemini has no model '{self.model}' for this API key. "
                    "Check LLM_MODEL."
                ) from exc
            if exc.code == 429:
                raise LLMRateLimitError(self._rate_limit_message(exc)) from exc
            raise

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=response.text or "",
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            latency_seconds=attempt_seconds,
            model=self.model,
            reasoning_tokens=getattr(usage, "thoughts_token_count", None),
        )

    def _rate_limit_message(self, exc: APIError) -> str:
        details = getattr(exc, "details", None)
        delay = parse_retry_delay(details)
        if is_daily_quota(details):
            hint = f" The server suggested waiting about {delay / 3600:.1f} hours." if delay else ""
            return (
                f"Gemini's daily quota for model '{self.model}' is used up, so "
                f"retrying now will fail.{hint} Wait for the quota to reset, use "
                "another model or API key, or switch to the Ollama backend."
            )
        hint = f" The server asked for a wait of about {delay:.0f}s." if delay else ""
        return (
            f"Gemini rate limit hit for model '{self.model}' and retries did not "
            f"clear it.{hint} Wait a bit and try again, or slow down requests."
        )


def _to_gemini_role(role: str) -> str:
    try:
        return _ROLES[role]
    except KeyError:
        raise ValueError(
            f"Unsupported message role {role!r}: use 'user' or 'assistant'. "
            "Pass system prompts through the `system` argument."
        ) from None