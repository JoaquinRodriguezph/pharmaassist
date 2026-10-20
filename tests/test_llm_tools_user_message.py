"""add_user_message for both tool sessions."""

from types import SimpleNamespace

from google.genai import types

from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient
from app.llm.tools import start_tool_session

SPEC = {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}


def test_ollama_user_message_is_appended_before_the_next_call():
    calls = []
    reply = SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), prompt_eval_count=1,
                            eval_count=1, model="m")
    fake = SimpleNamespace(chat=lambda **kw: calls.append(kw) or reply)
    client = OllamaClient(model="m", host="http://localhost:11434")
    client._client = fake
    session = start_tool_session(client, system=None, question="Q?", tool_specs=[SPEC])

    session.step()
    session.add_user_message("Add citations.")
    session.step()

    assert calls[1]["messages"][-1] == {"role": "user", "content": "Add citations."}


def test_gemini_user_message_is_a_user_content():
    calls = []
    content = types.Content(role="model", parts=[types.Part(text="ok")])
    reply = SimpleNamespace(function_calls=None, candidates=[SimpleNamespace(content=content)], usage_metadata=None)
    models = SimpleNamespace(generate_content=lambda **kw: calls.append(kw) or reply)
    client = GeminiClient(api_key="k", model="g", client=SimpleNamespace(models=models), sleep=lambda s: None)
    session = start_tool_session(client, system=None, question="Q?", tool_specs=[SPEC])

    session.step()
    session.add_user_message("Add citations.")
    session.step()

    last = calls[1]["contents"][-1]
    assert last.role == "user" and last.parts[0].text == "Add citations."