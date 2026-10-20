"""Tests for app.llm.tools with fake Ollama and Gemini clients. No network."""

import json
from types import SimpleNamespace

import ollama
import pytest
from google.genai import types

from app.llm.base import LLMConnectionError, LLMModelNotFoundError
from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient
from app.llm.tools import (
    GeminiToolSession,
    OllamaToolSession,
    ToolCall,
    _arguments,
    start_tool_session,
)

SPEC = {
    "name": "get_restock",
    "description": "Restock list.",
    "parameters": {"type": "object", "properties": {"branch_id": {"type": "integer"}}, "required": ["branch_id"],
                   "additionalProperties": False},
}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class FakeOllama:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)


def ollama_response(content="", calls=(), prompt=50, completion=10):
    tool_calls = [SimpleNamespace(function=SimpleNamespace(name=n, arguments=a)) for n, a in calls] or None
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls),
                           prompt_eval_count=prompt, eval_count=completion, model="llama3.2:latest")


def ollama_session(fake, history=()):
    client = OllamaClient(model="llama3.2:latest", host="http://localhost:11434")
    client._client = fake
    return start_tool_session(client, system="SYS", question="Q?", history=list(history), tool_specs=[SPEC],
                              temperature=0.0, max_tokens=256)


def test_ollama_first_call_has_system_history_question_tools_and_options():
    fake = FakeOllama([ollama_response("done")])
    session = ollama_session(fake, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])

    session.step()

    (call,) = fake.calls
    assert isinstance(session, OllamaToolSession)
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "Q?"},
    ]
    assert call["tools"] == [{"type": "function", "function": SPEC}]
    assert call["options"] == {"temperature": 0.0, "num_predict": 256}


def test_ollama_tool_call_then_result_then_answer():
    fake = FakeOllama([ollama_response(calls=[("get_restock", {"branch_id": 3})]), ollama_response("Final answer.")])
    session = ollama_session(fake)

    first = session.step()
    assert first.tool_calls == (ToolCall("get_restock", {"branch_id": 3}),)
    assert (first.prompt_tokens, first.completion_tokens, first.model) == (50, 10, "llama3.2:latest")

    session.add_tool_results([(first.tool_calls[0], '{"rows": []}')])
    second = session.step()

    sent = fake.calls[1]["messages"]
    assert sent[-1] == {"role": "tool", "content": '{"rows": []}', "tool_name": "get_restock"}
    assert sent[-2] is fake.calls[1]["messages"][-2]  # the model's own message is sent back
    assert second.text == "Final answer." and second.tool_calls == ()


def test_ollama_connection_error_is_translated():
    session = ollama_session(FakeOllama(error=ConnectionError("refused")))

    with pytest.raises(LLMConnectionError, match="Ollama"):
        session.step()


def test_ollama_missing_model_is_translated():
    session = ollama_session(FakeOllama(error=ollama.ResponseError("model not found", 404)))

    with pytest.raises(LLMModelNotFoundError):
        session.step()


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        ("not json", {"_unparsed": "not json"}),
        (5, {"_unparsed": 5}),
    ],
)
def test_arguments_are_read_as_a_dict(raw, expected):
    assert _arguments(raw) == expected


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def gemini_function_response():
    call = types.FunctionCall(id="call_1", name="get_restock", args={"branch_id": 3})
    content = types.Content(role="model", parts=[types.Part(function_call=call)])
    return SimpleNamespace(
        function_calls=[call],
        candidates=[SimpleNamespace(content=content)],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=50),
    ), content


def gemini_text_response():
    content = types.Content(role="model", parts=[types.Part(text="thinking...", thought=True), types.Part(text="Final.")])
    return SimpleNamespace(
        function_calls=None,
        candidates=[SimpleNamespace(content=content)],
        usage_metadata=SimpleNamespace(prompt_token_count=150, candidates_token_count=5, thoughts_token_count=None),
    )


def gemini_session(models, history=()):
    client = GeminiClient(api_key="test", model="gemini-test", client=SimpleNamespace(models=models), sleep=lambda s: None)
    return start_tool_session(client, system="SYS", question="Q?", history=list(history), tool_specs=[SPEC],
                              temperature=0.0, max_tokens=256)


def test_gemini_config_declares_the_tools_and_system_prompt():
    first, _ = gemini_function_response()
    models = FakeModels([first])
    session = gemini_session(models)

    session.step()

    config = models.calls[0]["config"]
    assert isinstance(session, GeminiToolSession)
    assert config.system_instruction == "SYS" and config.temperature == 0.0 and config.max_output_tokens == 256
    (declaration,) = config.tools[0].function_declarations
    assert declaration.name == "get_restock" and declaration.parameters_json_schema == SPEC["parameters"]


def test_gemini_history_uses_the_model_role():
    models = FakeModels([gemini_text_response()])
    session = gemini_session(models, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])

    session.step()

    roles = [c.role for c in models.calls[0]["contents"]]
    assert roles == ["user", "model", "user"]


def test_gemini_tool_call_then_result_then_answer():
    first, model_content = gemini_function_response()
    models = FakeModels([first, gemini_text_response()])
    session = gemini_session(models)

    turn = session.step()
    assert turn.tool_calls == (ToolCall("get_restock", {"branch_id": 3}, id="call_1"),)
    assert (turn.prompt_tokens, turn.completion_tokens, turn.reasoning_tokens) == (100, 20, 50)

    session.add_tool_results([(turn.tool_calls[0], '{"rows": []}')])
    final = session.step()

    contents = models.calls[1]["contents"]
    assert contents[-2] is model_content  # sent back unchanged, with its thought signature
    response_part = contents[-1].parts[0].function_response
    assert contents[-1].role == "user"
    assert (response_part.id, response_part.name, response_part.response) == ("call_1", "get_restock",
                                                                             {"output": '{"rows": []}'})
    assert final.text == "Final." and final.tool_calls == ()


def test_an_unsupported_client_is_refused():
    with pytest.raises(TypeError, match="not supported"):
        start_tool_session(object(), system=None, question="Q?", tool_specs=[SPEC])