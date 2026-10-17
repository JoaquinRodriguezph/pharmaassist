"""Tests for app.llm.gemini_client. A fake client stands in for the SDK: no
network, no real sleeping, and no quota used."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai.errors import APIError

from app.llm.base import (
    LLMClient,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
)
from app.llm.gemini_client import GeminiClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# Modeled on a real daily-quota 429 observed from the API.
DAILY_429 = load_fixture("gemini_429_daily.json")
# ASSUMED shape: no real per-minute 429 has been observed yet.
PER_MINUTE_429_ASSUMED = load_fixture("gemini_429_per_minute_ASSUMED.json")


def api_error(code: int) -> APIError:
    return APIError(code, {"error": {"code": code, "message": f"HTTP {code}"}})


# ---------------------------------------------------------------------------
# Fakes: same `.models.generate_content(...)` shape as the SDK client
# ---------------------------------------------------------------------------

class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # responses to return or errors to raise, in order
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes):
        self.models = FakeModels(outcomes)


def usage(prompt=20, candidates=40, thoughts=None):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=prompt + candidates + (thoughts or 0),
    )


def fake_response(text="Hello.", usage_metadata=None):
    return SimpleNamespace(text=text, usage_metadata=usage_metadata or usage())


def make_client(*outcomes, **kwargs):
    """Return (GeminiClient, fake SDK client, list of recorded sleeps)."""
    fake = FakeClient(*outcomes)
    sleeps: list[float] = []
    client = GeminiClient(model="test-model", client=fake, sleep=sleeps.append, **kwargs)
    return client, fake, sleeps


def system_text(config):
    """The SDK may keep the system prompt as a str or wrap it in a Content/Part."""
    value = config.system_instruction
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "parts"):
        return value.parts[0].text
    return value.text


USER_MSG = {"role": "user", "content": "What is a reorder level?"}


# ---------------------------------------------------------------------------
# Inputs arrive in the right place
# ---------------------------------------------------------------------------

def test_is_an_llm_client():
    client, _, _ = make_client(fake_response())
    assert isinstance(client, LLMClient)


def test_inputs_arrive_in_the_right_place():
    client, fake, _ = make_client(fake_response())
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "What is a reorder level?"},
    ]

    client.generate(history, system="Be brief.", temperature=0.7, max_tokens=123)

    (call,) = fake.models.calls
    assert call.model == "test-model"
    assert system_text(call.config) == "Be brief."
    assert call.config.temperature == 0.7
    assert call.config.max_output_tokens == 123
    # The assistant's turns use Gemini's role name "model", and the system
    # prompt is not part of the message list.
    assert [c.role for c in call.contents] == ["user", "model", "user"]
    assert [c.parts[0].text for c in call.contents] == [m["content"] for m in history]


@pytest.mark.parametrize("system", [None, ""])
def test_unset_options_are_left_out(system):
    client, fake, _ = make_client(fake_response())

    client.generate([USER_MSG], system=system)

    config = fake.models.calls[0].config
    assert system_text(config) is None
    assert config.max_output_tokens is None
    assert config.temperature == 0.2  # the interface default


@pytest.mark.parametrize("role", ["system", "tool"])
def test_unsupported_role_is_rejected_before_any_call(role):
    client, fake, _ = make_client(fake_response())

    with pytest.raises(ValueError, match="system"):
        client.generate([{"role": role, "content": "x"}])

    assert fake.models.calls == []


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "usage_metadata, prompt, completion, reasoning",
    [
        pytest.param(usage(20, 40, 450), 20, 40, 450, id="with-thinking-tokens"),
        pytest.param(usage(20, 40, None), 20, 40, None, id="thoughts-None-stays-None"),
        pytest.param(usage(20, 40, 0), 20, 40, 0, id="zero-thinking-tokens-kept-as-0"),
        pytest.param(
            SimpleNamespace(
                prompt_token_count=None,
                candidates_token_count=None,
                thoughts_token_count=None,
                total_token_count=None,
            ),
            None,
            None,
            None,
            id="all-counts-None",
        ),
    ],
)
def test_token_counts_map_correctly(usage_metadata, prompt, completion, reasoning):
    client, _, _ = make_client(fake_response("Two sentences.", usage_metadata))

    result = client.generate([USER_MSG])

    assert result.text == "Two sentences."
    assert result.model == "test-model"
    assert result.prompt_tokens == prompt
    assert result.completion_tokens == completion
    assert result.reasoning_tokens == reasoning


def test_missing_usage_metadata_gives_none_counts():
    response = SimpleNamespace(text="Hi", usage_metadata=None)
    client, _, _ = make_client(response)

    result = client.generate([USER_MSG])

    assert (result.prompt_tokens, result.completion_tokens, result.reasoning_tokens) == (
        None,
        None,
        None,
    )


def test_text_none_becomes_empty_string():
    # e.g. the output limit was used up before any visible text appeared
    response = SimpleNamespace(text=None, usage_metadata=usage())
    client, _, _ = make_client(response)

    assert client.generate([USER_MSG]).text == ""


def test_latency_covers_only_the_successful_attempt():
    ticks = iter([0.0, 5.0, 5.5])  # failed attempt starts at 0.0; success runs 5.0 -> 5.5
    client, _, _ = make_client(
        api_error(503), fake_response(), clock=lambda: next(ticks)
    )

    assert client.generate([USER_MSG]).latency_seconds == 0.5


# ---------------------------------------------------------------------------
# Errors after retries
# ---------------------------------------------------------------------------

def test_404_becomes_model_not_found():
    original = api_error(404)
    client, fake, sleeps = make_client(original)

    with pytest.raises(LLMModelNotFoundError) as exc_info:
        client.generate([USER_MSG])

    assert "test-model" in str(exc_info.value)
    assert exc_info.value.__cause__ is original
    assert len(fake.models.calls) == 1
    assert sleeps == []


def test_daily_429_becomes_rate_limit_error_without_retrying():
    original = APIError(429, DAILY_429)
    client, fake, sleeps = make_client(original)

    with pytest.raises(LLMRateLimitError) as exc_info:
        client.generate([USER_MSG])

    message = str(exc_info.value)
    assert "daily quota" in message
    assert "15.6 hours" in message  # the fixture's retryDelay is 56026s
    assert exc_info.value.__cause__ is original
    assert len(fake.models.calls) == 1
    assert sleeps == []


def test_429_still_failing_after_retries_becomes_rate_limit_error():
    error = APIError(429, PER_MINUTE_429_ASSUMED)
    client, fake, sleeps = make_client(error, error, error, error)

    with pytest.raises(LLMRateLimitError) as exc_info:
        client.generate([USER_MSG])

    assert "rate limit" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMError)
    assert len(fake.models.calls) == 4
    assert sleeps == [5.0, 5.0, 5.0]  # the server's retryDelay, three times


@pytest.mark.parametrize(
    "error, expected_calls",
    [
        pytest.param(api_error(401), 1, id="401-not-retried-and-not-translated"),
        pytest.param(api_error(500), 1, id="500-not-retried-and-not-translated"),
        pytest.param(api_error(503), 4, id="503-retried-then-not-translated"),
    ],
)
def test_other_errors_pass_through_unchanged(error, expected_calls):
    client, fake, _ = make_client(*[error] * expected_calls)

    with pytest.raises(APIError) as exc_info:
        client.generate([USER_MSG])

    assert exc_info.value is error
    assert len(fake.models.calls) == expected_calls


def test_503_twice_then_success_uses_injected_sleep():
    client, fake, sleeps = make_client(
        api_error(503), api_error(503), fake_response("Recovered.")
    )

    result = client.generate([USER_MSG])

    assert result.text == "Recovered."
    assert len(fake.models.calls) == 3
    assert len(sleeps) == 2
    for attempt, slept in enumerate(sleeps, start=1):
        assert 2**attempt + 0.1 <= slept <= 2**attempt + 1.0


# ---------------------------------------------------------------------------
# Configuration from the environment
# ---------------------------------------------------------------------------

def test_model_comes_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "env-model")

    client = GeminiClient(client=FakeClient(fake_response()))

    assert client.model == "env-model"


def test_missing_model_gives_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(ValueError, match="LLM_MODEL"):
        GeminiClient(client=FakeClient())


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        GeminiClient(model="test-model")  # no client injected, so a key is needed


def test_injected_client_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    GeminiClient(model="test-model", client=FakeClient())  # must not raise


# ---------------------------------------------------------------------------
# Design check: the Ollama side must never pull in google-genai or Gemini code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["app.llm", "app.llm.ollama_client"])
def test_module_does_not_import_google_or_gemini_client(module):
    code = (
        f"import sys, {module}\n"
        "bad = [m for m in sys.modules if m == 'google' or m.startswith('google.')]\n"
        "assert not bad, bad\n"
        "assert 'app.llm.gemini_client' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr