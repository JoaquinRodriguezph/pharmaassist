"""Request and response bodies for the PharmaAssist API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_CHARS = 4000


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ToolCallOut(BaseModel):
    name: str
    arguments: dict
    ok: bool
    rows: int | None
    error: str | None


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[str]
    tool_calls: list[ToolCallOut]
    refusal: bool
    medical_refusal: bool
    scope_refusal: bool
    guard: str | None
    steps: int
    latency_seconds: float | None
    model: str | None

    @classmethod
    def from_result(cls, result) -> "ChatResponse":
        return cls(
            answer=result.text,
            citations=list(result.valid_citations),
            sources=list(result.kb_sources),
            tool_calls=[
                ToolCallOut(name=c.name, arguments=c.arguments, ok=c.ok, rows=c.rows, error=c.error)
                for c in result.tool_calls
            ],
            refusal=result.refusal,
            medical_refusal=result.medical_refusal,
            scope_refusal=result.scope_refusal,
            guard=result.blocked_by,
            steps=result.steps,
            latency_seconds=result.latency_seconds,
            model=result.model,
        )


class HealthResponse(BaseModel):
    status: str
    backend: str