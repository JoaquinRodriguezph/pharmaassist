"""Tests for app.llm.factory. No API calls: constructing a client must not touch the network."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.llm.base import LLMClient
from app.llm.factory import create_llm_client
from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_fake_env(monkeypatch, backend):
    monkeypatch.setenv("LLM_BACKEND", backend)
    monkeypatch.setenv("OLLAMA_MODEL", "fake-ollama-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-gemini-model")


@pytest.mark.parametrize("value", ["ollama", "OLLAMA", "Ollama", " ollama "])
def test_ollama_backend(value, monkeypatch):
    set_fake_env(monkeypatch, value)

    client = create_llm_client()

    assert isinstance(client, OllamaClient)
    assert isinstance(client, LLMClient)
    assert client.model == "fake-ollama-model"


@pytest.mark.parametrize("value", ["gemini", "GEMINI", "Gemini"])
def test_gemini_backend(value, monkeypatch):
    set_fake_env(monkeypatch, value)

    client = create_llm_client()

    assert isinstance(client, GeminiClient)
    assert isinstance(client, LLMClient)
    assert client.model == "fake-gemini-model"


@pytest.mark.parametrize(
    "value, expected_fragment",
    [
        pytest.param(None, "not set", id="missing"),
        pytest.param("", "not set", id="empty"),
        pytest.param("   ", "not set", id="blank"),
        pytest.param("openai", "'openai'", id="unknown-value-is-echoed"),
    ],
)
def test_missing_or_unknown_backend_lists_allowed_values(value, expected_fragment, monkeypatch):
    if value is None:
        monkeypatch.delenv("LLM_BACKEND", raising=False)
    else:
        monkeypatch.setenv("LLM_BACKEND", value)

    with pytest.raises(ValueError) as exc_info:
        create_llm_client()

    message = str(exc_info.value)
    assert "LLM_BACKEND" in message
    assert expected_fragment in message
    assert "ollama" in message and "gemini" in message  # the allowed values


def run_in_fresh_interpreter(code):
    env = {
        **os.environ,
        "LLM_BACKEND": "ollama",
        "OLLAMA_MODEL": "fake-ollama-model",
        "OLLAMA_HOST": "http://localhost:11434",
    }
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )


def test_importing_factory_does_not_import_google():
    result = run_in_fresh_interpreter(
        "import sys, app.llm.factory\n"
        "bad = [m for m in sys.modules if m == 'google' or m.startswith('google.')]\n"
        "assert not bad, bad\n"
    )
    assert result.returncode == 0, result.stderr


def test_choosing_ollama_never_imports_google():
    result = run_in_fresh_interpreter(
        "import sys\n"
        "from app.llm.factory import create_llm_client\n"
        "client = create_llm_client()\n"
        "assert type(client).__name__ == 'OllamaClient', type(client)\n"
        "bad = [m for m in sys.modules if m == 'google' or m.startswith('google.')]\n"
        "assert not bad, bad\n"
        "assert 'app.llm.gemini_client' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr