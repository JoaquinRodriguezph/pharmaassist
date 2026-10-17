"""Backend-neutral types for talking to a language model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    """One turn of conversation history. System prompts are NOT messages."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Result of one generate() call.

    Backends fill in what they know and leave the rest as None.

    Design decision on hidden "thinking" tokens: `reasoning_tokens` is a
    separate optional field (None when a backend reports none, as Ollama does)
    and `completion_tokens` counts only the visible answer. Folding them
    together would hide what drives cost and latency in an evaluation report
    (a 40-token answer that cost ~450 thinking tokens would look like 490),
    and dropping them would hide tokens that Gemini counts in its total (see
    the Gemini pricing docs for how they are billed); a separate field lets
    the report show visible length, total output (completion + reasoning) and
    latency side by side.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float | None = None
    model: str | None = None
    reasoning_tokens: int | None = None


class LLMError(Exception):
    """Base class for errors raised by any LLM backend."""


class LLMConnectionError(LLMError):
    """The backend could not be reached (e.g. the server is not running)."""


class LLMModelNotFoundError(LLMError):
    """The backend is reachable but does not have the requested model."""


class LLMRateLimitError(LLMError):
    """The backend refused the request because of quota or rate limits."""


class LLMClient(ABC):
    """Interface every LLM backend implements."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate the assistant's next reply.

        Args:
            messages: Conversation history, oldest first.
            system: System prompt. Passed separately because backends place
                it differently (a message for Ollama, a config field for
                Gemini); each backend handles that itself.
            temperature: Sampling temperature.
            max_tokens: Cap on generated tokens; None means backend default.
        """