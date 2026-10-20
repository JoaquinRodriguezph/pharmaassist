"""num_ctx reaches Ollama's options and is accepted (and ignored) by Gemini."""

from types import SimpleNamespace

from google.genai import types

from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient
from app.llm.tools import start_tool_session

SPEC = {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}


def ollama_calls(num_ctx):
    calls = []
    reply = SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), prompt_eval_count=1,
                            eval_count=1, model="m")
    client = OllamaClient(model="m", host="http://localhost:11434")
    client._client = SimpleNamespace(chat=lambda **kw: calls.append(kw) or reply)
    start_tool_session(client, system=None, question="Q?", tool_specs=[SPEC], num_ctx=num_ctx).step()
    return calls


def test_ollama_num_ctx_is_set_when_given():
    assert ollama_calls(8192)[0]["options"]["num_ctx"] == 8192


def test_ollama_num_ctx_is_absent_by_default():
    assert "num_ctx" not in ollama_calls(None)[0]["options"]


def test_gemini_accepts_num_ctx():
    content = types.Content(role="model", parts=[types.Part(text="ok")])
    reply = SimpleNamespace(function_calls=None, candidates=[SimpleNamespace(content=content)], usage_metadata=None)
    models = SimpleNamespace(generate_content=lambda **kw: reply)
    client = GeminiClient(api_key="k", model="g", client=SimpleNamespace(models=models), sleep=lambda s: None)

    assert start_tool_session(client, system=None, question="Q?", tool_specs=[SPEC], num_ctx=8192).step().text == "ok"