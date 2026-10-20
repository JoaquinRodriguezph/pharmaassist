"""Tool calling for both backends, behind one small session interface.

A session holds one conversation with tool declarations. `step()` asks the model for its next
turn: either text (the final answer) or one or more tool calls. After running the calls,
`add_tool_results` sends their outputs back, and `step()` is called again. `add_user_message`
adds a follow-up user message, and `step(allow_tools=False)` asks for text only (used for
the citation repair turn).

Shapes confirmed by scripts/probe_tool_calling.py (ollama 0.6.3, google-genai 2.31.0):
- Ollama returns message.tool_calls[i].function.name and .arguments (a dict). A tool result
  is the message {"role": "tool", "content": <text>, "tool_name": <name>}.
- Gemini returns response.function_calls (FunctionCall with id, name, args) and a model
  Content whose parts carry a thought_signature, so that Content is sent back unchanged.
  Results go back as FunctionResponse parts with the call's id and {"output": <text>}.
  Sending a result back to Gemini has not been run live yet.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import ollama
from google.genai import types
from google.genai.errors import APIError

from app.llm.base import LLMConnectionError, LLMModelNotFoundError, LLMRateLimitError, Message
from app.llm.gemini_client import GeminiClient, _to_gemini_role
from app.llm.gemini_errors import classify_gemini_error
from app.llm.ollama_client import OllamaClient
from app.llm.retry import call_with_retry


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None


@dataclass(frozen=True)
class ToolTurn:
    text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_seconds: float | None = None
    model: str | None = None


class ToolSession(ABC):
    @abstractmethod
    def step(self, allow_tools: bool = True) -> ToolTurn:
        """Ask the model for its next turn. With allow_tools=False it may only answer in text."""

    @abstractmethod
    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        """Send back the output text of each tool call from the last turn."""

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Add a follow-up user message before the next step."""


def _arguments(raw: Any) -> dict:
    """Tool-call arguments as a dict. Anything unreadable comes back as {'_unparsed': raw},
    which tool validation then refuses with a clear message."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            raw = json.loads(raw)
        except ValueError:
            return {"_unparsed": raw}
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {"_unparsed": raw}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaToolSession(ToolSession):
    def __init__(self, client: OllamaClient, *, system: str | None, question: str, history: list[Message],
                 tool_specs: list[dict], temperature: float, max_tokens: int | None) -> None:
        self._client = client
        self._messages: list[Any] = [{"role": "system", "content": system}] if system else []
        self._messages += [{"role": m["role"], "content": m["content"]} for m in history]
        self._messages.append({"role": "user", "content": question})
        self._tools = [{"type": "function", "function": spec} for spec in tool_specs]
        self._options: dict[str, float | int] = {"temperature": temperature}
        if max_tokens is not None:
            self._options["num_predict"] = max_tokens

    def step(self, allow_tools: bool = True) -> ToolTurn:
        start = time.perf_counter()
        try:
            response = self._client._client.chat(
                model=self._client.model, messages=list(self._messages),
                tools=self._tools if allow_tools else None, options=self._options,
            )
        except ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not reach the Ollama server at {self._client.host}. Is it running?"
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMModelNotFoundError(
                    f"Ollama has no model '{self._client.model}'. Run `ollama pull {self._client.model}`."
                ) from exc
            raise
        latency = time.perf_counter() - start
        message = response.message
        self._messages.append(message)
        calls = tuple(
            ToolCall(name=call.function.name, arguments=_arguments(call.function.arguments))
            for call in (message.tool_calls or [])
        )
        return ToolTurn(
            text=message.content or "",
            tool_calls=calls,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            latency_seconds=latency,
            model=response.model or self._client.model,
        )

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        for call, output in results:
            self._messages.append({"role": "tool", "content": output, "tool_name": call.name})

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiToolSession(ToolSession):
    def __init__(self, client: GeminiClient, *, system: str | None, question: str, history: list[Message],
                 tool_specs: list[dict], temperature: float, max_tokens: int | None) -> None:
        self._client = client
        self._contents: list[types.Content] = [
            types.Content(role=_to_gemini_role(m["role"]), parts=[types.Part.from_text(text=m["content"])])
            for m in history
        ]
        self._contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        declarations = [
            types.FunctionDeclaration(
                name=spec["name"], description=spec["description"], parameters_json_schema=spec["parameters"]
            )
            for spec in tool_specs
        ]
        config: dict[str, Any] = {"temperature": temperature, "tools": [types.Tool(function_declarations=declarations)]}
        if system:
            config["system_instruction"] = system
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        self._config = types.GenerateContentConfig(**config)
        self._text_only_config = self._config.model_copy(update={
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
            )
        })

    def step(self, allow_tools: bool = True) -> ToolTurn:
        contents = list(self._contents)
        config = self._config if allow_tools else self._text_only_config
        attempt_seconds = 0.0

        def attempt():
            nonlocal attempt_seconds
            start = self._client._clock()
            response = self._client._client.models.generate_content(
                model=self._client.model, contents=contents, config=config
            )
            attempt_seconds = self._client._clock() - start
            return response

        try:
            response = call_with_retry(
                attempt, classify_gemini_error, max_attempts=self._client.max_attempts, sleep=self._client._sleep
            )
        except APIError as exc:
            if exc.code == 404:
                raise LLMModelNotFoundError(f"Gemini has no model '{self._client.model}'. Check GEMINI_MODEL.") from exc
            if exc.code == 429:
                raise LLMRateLimitError(self._client._rate_limit_message(exc)) from exc
            raise

        candidates = getattr(response, "candidates", None) or []
        content = candidates[0].content if candidates else None
        if content is not None:
            self._contents.append(content)
        parts = (getattr(content, "parts", None) or []) if content is not None else []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        calls = tuple(
            ToolCall(name=call.name, arguments=_arguments(call.args), id=call.id)
            for call in (getattr(response, "function_calls", None) or [])
        )
        usage = getattr(response, "usage_metadata", None)
        return ToolTurn(
            text=text,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            reasoning_tokens=getattr(usage, "thoughts_token_count", None),
            latency_seconds=attempt_seconds,
            model=self._client.model,
        )

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        parts = [
            types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response={"output": output}))
            for call, output in results
        ]
        self._contents.append(types.Content(role="user", parts=parts))

    def add_user_message(self, text: str) -> None:
        self._contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))


def start_tool_session(client, *, system: str | None, question: str, history: list[Message] | None = None,
                       tool_specs: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> ToolSession:
    kwargs = dict(system=system, question=question, history=list(history or []), tool_specs=tool_specs,
                  temperature=temperature, max_tokens=max_tokens)
    if isinstance(client, OllamaClient):
        return OllamaToolSession(client, **kwargs)
    if isinstance(client, GeminiClient):
        return GeminiToolSession(client, **kwargs)
    raise TypeError(f"tool calling is not supported for {type(client).__name__}")