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
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float | None = None
    model: str | None = None


class LLMConnectionError(Exception):
    """The backend could not be reached (e.g. the server is not running)."""


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