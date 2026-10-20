"""step(allow_tools=False) for both tool sessions."""

from types import SimpleNamespace

from google.genai import types

from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient
from app.llm.tools import start_tool_session

SPEC = {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}


def test_ollama_text_only_step_sends_no_tools():
    calls = []
    reply = SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), prompt_eval_count=1,
                            eval_count=1, model="m")
    client = OllamaClient(model="m", host="http://localhost:11434")
    client._client = SimpleNamespace(chat=lambda **kw: calls.append(kw) or reply)
    session = start_tool_session(client, system=None, question="Q?", tool_specs=[SPEC])

    session.step()
    session.step(allow_tools=False)

    assert calls[0]["tools"] == [{"type": "function", "function": SPEC}] and calls[1]["tools"] is None


def test_gemini_text_only_step_sets_function_calling_to_none():
    calls = []
    content = types.Content(role="model", parts=[types.Part(text="ok")])
    reply = SimpleNamespace(function_calls=None, candidates=[SimpleNamespace(content=content)], usage_metadata=None)
    models = SimpleNamespace(generate_content=lambda **kw: calls.append(kw) or reply)
    client = GeminiClient(api_key="k", model="g", client=SimpleNamespace(models=models), sleep=lambda s: None)
    session = start_tool_session(client, system=None, question="Q?", tool_specs=[SPEC])

    session.step()
    session.step(allow_tools=False)

    assert calls[0]["config"].tool_config is None
    mode = calls[1]["config"].tool_config.function_calling_config.mode
    assert mode == types.FunctionCallingConfigMode.NONE
    assert calls[1]["config"].tools == calls[0]["config"].tools