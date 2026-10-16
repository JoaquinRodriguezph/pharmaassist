"""Tests for app.llm.retry and app.llm.gemini_errors. No API calls, no real sleeping."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from google.genai.errors import APIError

from app.llm.gemini_errors import (
    classify_gemini_error,
    get_details_list,
    parse_retry_delay,
)
from app.llm.retry import RetryDecision, call_with_retry

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# MODELED ON REAL
DAILY_429 = load_fixture("gemini_429_daily.json")

# ASSUMED: we have not observed a real per-minute 429 yet. This assumes the
# same QuotaFailure/RetryInfo structure as the daily one. Replace the fixture
# with a real body when one is seen.
PER_MINUTE_429_ASSUMED = load_fixture("gemini_429_per_minute_ASSUMED.json")

RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"


def api_error(code: int) -> APIError:
    return APIError(code, {"error": {"code": code, "message": f"HTTP {code}"}})


# ---------------------------------------------------------------------------
# Classification: one error in, one RetryDecision out
# ---------------------------------------------------------------------------

CLASSIFY_CASES = [
    pytest.param(APIError(429, DAILY_429), False, None, id="429-daily-quota-fails-fast"),
    pytest.param(
        APIError(429, PER_MINUTE_429_ASSUMED), True, 5.0, id="429-per-minute-ASSUMED-uses-server-delay"
    ),
    pytest.param(APIError(429, {}), True, None, id="429-empty-body-falls-back-to-backoff"),
    pytest.param(
        APIError(429, {"error": {"details": [{"@type": RETRY_INFO, "retryDelay": "120s"}]}}),
        False,
        None,
        id="429-delay-over-threshold-fails-fast",
    ),
    pytest.param(api_error(503), True, None, id="503-retryable"),
    *[
        pytest.param(api_error(code), False, None, id=f"{code}-client-error-not-retryable")
        for code in (400, 401, 403, 404)
    ],
    pytest.param(api_error(500), False, None, id="500-unhandled-not-retryable"),
    pytest.param(TypeError("bug"), False, None, id="non-api-error-not-retryable"),
]


@pytest.mark.parametrize("error, retryable, wait_seconds", CLASSIFY_CASES)
def test_classify_gemini_error(error, retryable, wait_seconds):
    decision = classify_gemini_error(error)
    assert decision.retryable is retryable
    assert decision.wait_seconds == wait_seconds


def test_daily_quota_reason_mentions_daily():
    decision = classify_gemini_error(APIError(429, DAILY_429))
    assert "daily" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "details, expected",
    [
        pytest.param(DAILY_429, 56026.0, id="real-daily-body"),
        pytest.param(PER_MINUTE_429_ASSUMED, 5.0, id="assumed-per-minute-body"),
        pytest.param([{"@type": RETRY_INFO, "retryDelay": 7}], 7.0, id="numeric-delay"),
        pytest.param([{"retryDelay": "xs"}], None, id="unparseable-delay"),
        pytest.param({}, None, id="empty-dict"),
        pytest.param(None, None, id="none"),
    ],
)
def test_parse_retry_delay(details, expected):
    assert parse_retry_delay(details) == expected


INNER = [{"@type": RETRY_INFO, "retryDelay": "5s"}]


@pytest.mark.parametrize(
    "details, expected",
    [
        pytest.param(None, [], id="none"),
        pytest.param({}, [], id="empty-dict"),
        pytest.param(INNER, INNER, id="already-a-list"),
        pytest.param({"error": {"details": INNER}}, INNER, id="nested-under-error"),
        pytest.param({"details": INNER}, INNER, id="top-level-details"),
        pytest.param({"error": "oops"}, [], id="error-not-a-dict"),
    ],
)
def test_get_details_list(details, expected):
    assert get_details_list(details) == expected


# ---------------------------------------------------------------------------
# Retry loop with the Gemini classifier (attempts and sleeps)
# ---------------------------------------------------------------------------

def run_failing_endpoint(error, max_attempts=4):
    """Call an endpoint that always raises `error`; return (attempts, sleeps, raised)."""
    attempts = 0
    sleeps: list[float] = []

    def endpoint():
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error)) as exc_info:
        call_with_retry(
            endpoint, classify_gemini_error, max_attempts=max_attempts, sleep=sleeps.append
        )
    return attempts, sleeps, exc_info.value


LOOP_CASES = [
    pytest.param(api_error(401), 1, 0, id="401-one-attempt-no-sleep"),
    pytest.param(APIError(429, DAILY_429), 1, 0, id="daily-429-one-attempt-no-sleep"),
    pytest.param(api_error(503), 4, 3, id="persistent-503-four-attempts-three-sleeps"),
    pytest.param(
        APIError(429, PER_MINUTE_429_ASSUMED),
        4,
        3,
        id="persistent-429-per-minute-ASSUMED-four-attempts",
    ),
]


@pytest.mark.parametrize("error, expected_attempts, expected_sleeps", LOOP_CASES)
def test_loop_attempts_and_sleeps(error, expected_attempts, expected_sleeps):
    attempts, sleeps, raised = run_failing_endpoint(error)
    assert attempts == expected_attempts
    assert len(sleeps) == expected_sleeps
    assert raised is error  # the original exception comes back, not a wrapper


def test_persistent_503_backs_off_exponentially_with_jitter():
    _, sleeps, _ = run_failing_endpoint(api_error(503))
    for attempt, slept in enumerate(sleeps, start=1):
        assert 2**attempt + 0.1 <= slept <= 2**attempt + 1.0


def test_server_delay_is_used_instead_of_backoff():
    _, sleeps, _ = run_failing_endpoint(APIError(429, PER_MINUTE_429_ASSUMED))
    assert sleeps == [5.0, 5.0, 5.0]


def test_recovering_endpoint_succeeds_on_attempt_3():
    attempts = 0
    sleeps: list[float] = []
    error = APIError(429, PER_MINUTE_429_ASSUMED)

    def endpoint():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise error
        return "SUCCESS"

    result = call_with_retry(endpoint, classify_gemini_error, max_attempts=4, sleep=sleeps.append)
    assert result == "SUCCESS"
    assert attempts == 3
    assert sleeps == [5.0, 5.0]


# ---------------------------------------------------------------------------
# The retry loop on its own (no Google types involved)
# ---------------------------------------------------------------------------

def test_loop_works_with_any_classifier():
    sleeps: list[float] = []
    attempts = 0

    def endpoint():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("flaky")
        return "ok"

    def classify(error):
        return RetryDecision(isinstance(error, ConnectionError), 1.5, "flaky network")

    assert call_with_retry(endpoint, classify, sleep=sleeps.append) == "ok"
    assert sleeps == [1.5]


def test_success_on_first_try_never_sleeps():
    sleeps: list[float] = []
    assert call_with_retry(lambda: "ok", classify_gemini_error, sleep=sleeps.append) == "ok"
    assert sleeps == []


def test_max_attempts_below_one_is_rejected():
    with pytest.raises(ValueError):
        call_with_retry(lambda: "ok", classify_gemini_error, max_attempts=0)


# ---------------------------------------------------------------------------
# Design check: only the Gemini backend may pull in google-genai
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["app.llm.retry", "app.llm.ollama_client"])
def test_module_does_not_import_google(module):
    code = (
        f"import sys, {module}\n"
        "bad = [m for m in sys.modules if m == 'google' or m.startswith('google.')]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr