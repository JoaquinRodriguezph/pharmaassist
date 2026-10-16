import os
import sys
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError


# =====================================================================
# 1. Error Parsing & Retry Helpers
# =====================================================================

def get_details_list(details):
    """
    Safely retrieves the inner details list from error.details.
    In the Google GenAI SDK, error.details is the full parsed response JSON:
    {'error': {'code': 429, 'status': '...', 'details': [...]}}
    This helper navigates to the inner details list without crashing if keys are missing.
    """
    if not details:
        return []
    if isinstance(details, list):
        return details
    if isinstance(details, dict):
        if "error" in details and isinstance(details["error"], dict):
            inner = details["error"].get("details")
            if isinstance(inner, list):
                return inner
        inner = details.get("details")
        if isinstance(inner, list):
            return inner
    return []


def parse_retry_delay(details):
    """
    Extracts retryDelay in seconds from Google API error details.
    Navigates under error.details['error']['details'] for RetryInfo:
    {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '56026s'}
    """
    details_list = get_details_list(details)
    for item in details_list:
        if isinstance(item, dict):
            item_type = item.get("@type", "")
            if "RetryInfo" in item_type or "retryDelay" in item:
                val = item.get("retryDelay")
                if isinstance(val, str) and val.endswith("s"):
                    try:
                        return float(val[:-1])
                    except ValueError:
                        pass
                elif isinstance(val, (int, float)):
                    return float(val)
    return None


def is_daily_quota(details):
    """
    Checks if QuotaFailure indicates daily quota exhaustion.
    Inspects quotaId inside violations of google.rpc.QuotaFailure under
    error.details['error']['details'].
    """
    details_list = get_details_list(details)
    for item in details_list:
        if isinstance(item, dict):
            item_type = item.get("@type", "")
            if "QuotaFailure" in item_type:
                violations = item.get("violations", [])
                if isinstance(violations, list):
                    for v in violations:
                        if isinstance(v, dict):
                            quota_id = v.get("quotaId", "")
                            desc = v.get("description", "")
                            if "PerDay" in quota_id or "PerDay" in desc or "per_day" in quota_id.lower():
                                return True
    return False


def evaluate_error(error, max_allowed_delay=60.0):
    """
    Decides whether an error is retryable and how long to wait.
    Returns: (is_retryable: bool, wait_seconds: float | None, reason: str)
    """
    if isinstance(error, APIError):
        code = getattr(error, "code", None)
        details = getattr(error, "details", None)

        # 400, 401, 403, 404: Fatal client/auth errors (never retry)
        if code in (400, 401, 403, 404):
            return False, None, f"Client error HTTP {code}: not retryable."

        # 503: Service Unavailable (transient backend outage)
        if code == 503:
            return True, None, "HTTP 503: Service Unavailable (transient server error)."

        # 429: Rate Limit / Resource Exhausted
        if code == 429:
            if is_daily_quota(details):
                return False, None, "Daily quota limit reached (QuotaFailure contains 'PerDay'). Failing fast."

            delay = parse_retry_delay(details)
            if delay is not None:
                if delay > max_allowed_delay:
                    return False, None, f"429 retryDelay of {delay}s exceeds threshold ({max_allowed_delay}s). Failing fast."
                return True, delay, f"429 transient rate limit with server delay of {delay}s."

            return True, None, "429 transient rate limit (using exponential backoff)."

        return False, None, f"Unhandled HTTP {code} APIError."

    # Programming bugs (TypeError, NameError, etc.) are never retried
    return False, None, f"Non-retryable exception type: {type(error).__name__}."


def call_with_retry(fn, *args, max_attempts=4, max_allowed_delay=60.0, sleep=time.sleep, **kwargs):
    """
    Invokes fn(*args, **kwargs) up to max_attempts with exponential backoff.
    Accepts any callable (real API client or mock function for testing).
    Uses dependency injection for `sleep` to allow instant tests without real waiting.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            attempt += 1
            return fn(*args, **kwargs)
        except Exception as e:
            is_retryable, server_delay, reason = evaluate_error(e, max_allowed_delay=max_allowed_delay)

            if not is_retryable:
                print(f"[Retry Engine] Non-retryable error ({reason}). Raising immediately.")
                raise e

            if attempt >= max_attempts:
                print(f"[Retry Engine] Max retry attempts ({max_attempts}) reached. Raising error.")
                raise e

            if server_delay is not None:
                sleep_seconds = server_delay
            else:
                sleep_seconds = (2 ** attempt) + random.uniform(0.1, 1.0)

            print(f"[Retry Engine] Attempt {attempt}/{max_attempts} failed: {reason} Waiting {sleep_seconds:.2f}s...")
            sleep(sleep_seconds)

    raise RuntimeError("Max retry attempts reached without success.")


# =====================================================================
# 2. Self-Test Suite (Runs only with --test, Zero API Calls, Zero Sleep)
# =====================================================================

def run_self_tests():
    """
    Verifies error evaluation, delay extraction, and retry loop logic.
    Uses real APIError constructors, exact Google RPC error shapes, and dependency-injected sleep.
    """
    t_start = time.perf_counter()

    # Case A: Modeled on a real daily 429 error returned by Google RPC
    # Preserves exact quotaMetric, quotaId, quotaDimensions, and retryDelay from the observed response.
    real_daily_response_json = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head here: https://ai.google.dev/gemini-api/docs/troubleshooting/quota-errors",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.Help",
                    "links": [
                        {
                            "description": "Google AI Studio API rate limits",
                            "url": "https://ai.google.dev/gemini-api/docs/troubleshooting/quota-errors"
                        }
                    ]
                },
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaDimensions": {
                                "location": "global",
                                "model": "gemini-3.8-flash"
                            },
                            "quotaValue": "20"
                        }
                    ]
                },
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "56026s"
                }
            ]
        }
    }

    # Case B: Assumed shape for transient per-minute rate limit (we have not observed a real
    # per-minute error payload from the API yet, so this assumes matching QuotaFailure/RetryInfo structure)
    assumed_per_minute_response_json = {
        "error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                            "quotaDimensions": {
                                "location": "global",
                                "model": "gemini-3.8-flash"
                            },
                            "quotaValue": "5"
                        }
                    ]
                },
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "5s"
                }
            ]
        }
    }

    # 1. Test Case A: Real daily quota fails fast
    err_daily = APIError(429, real_daily_response_json)
    retryable, delay, reason = evaluate_error(err_daily, max_allowed_delay=60.0)
    assert not retryable, f"Expected daily 429 to fail fast, got retryable={retryable}, reason={reason}"
    assert delay is None

    # 2. Test Case B: Assumed transient per-minute quota retries with explicit delay
    err_transient = APIError(429, assumed_per_minute_response_json)
    retryable, delay, reason = evaluate_error(err_transient, max_allowed_delay=60.0)
    assert retryable, f"Expected transient 429 to be retryable, got: {reason}"
    assert delay == 5.0, f"Expected delay 5.0s, got: {delay}"

    # 3. Test Case C: 401 Unauthorized fails immediately (1 attempt, 0 sleeps)
    sleeps_401 = []
    attempts_401 = 0
    err_401 = APIError(401, {"error": {"code": 401, "message": "API key not valid."}})

    def mock_401_endpoint():
        nonlocal attempts_401
        attempts_401 += 1
        raise err_401

    try:
        call_with_retry(mock_401_endpoint, max_attempts=4, sleep=lambda s: sleeps_401.append(s))
        assert False, "Expected 401 to raise"
    except APIError:
        pass
    assert attempts_401 == 1, f"Expected exactly 1 attempt on 401, got {attempts_401}"
    assert len(sleeps_401) == 0, f"Expected 0 sleeps on 401, got {len(sleeps_401)}"

    # 4. Test Case D: Persistent 503 makes exactly 4 attempts, sleeps 3 times, then raises
    sleeps_503 = []
    attempts_503 = 0
    err_503 = APIError(503, {"error": {"code": 503, "message": "Service Unavailable"}})

    def mock_503_endpoint():
        nonlocal attempts_503
        attempts_503 += 1
        raise err_503

    try:
        call_with_retry(mock_503_endpoint, max_attempts=4, sleep=lambda s: sleeps_503.append(s))
        assert False, "Expected persistent 503 to raise after 4 attempts"
    except APIError:
        pass
    assert attempts_503 == 4, f"Expected exactly 4 attempts on persistent 503, got {attempts_503}"
    assert len(sleeps_503) == 3, f"Expected exactly 3 sleeps on persistent 503, got {len(sleeps_503)}"

    # 5. Test Case E: Transient failure recovering on attempt 3
    sleeps_transient = []
    attempts_transient = 0

    def mock_recovering_endpoint():
        nonlocal attempts_transient
        attempts_transient += 1
        if attempts_transient < 3:
            raise err_transient
        return "SUCCESS"

    result = call_with_retry(mock_recovering_endpoint, max_attempts=4, sleep=lambda s: sleeps_transient.append(s))
    assert result == "SUCCESS", "Expected recovering endpoint to succeed"
    assert attempts_transient == 3, f"Expected 3 attempts, got {attempts_transient}"
    assert sleeps_transient == [5.0, 5.0], f"Expected sleeps [5.0, 5.0], got {sleeps_transient}"

    # 6. Test Case F: Empty/malformed response body handled gracefully without crashing
    err_empty = APIError(429, {})
    retryable_empty, delay_empty, _ = evaluate_error(err_empty, max_allowed_delay=60.0)
    assert retryable_empty, "Empty 429 details should fallback to transient retry"
    assert delay_empty is None

    elapsed = time.perf_counter() - t_start
    print(f"[Self-Test] All error evaluation and retry loop assertions passed in {elapsed:.4f}s.")


# =====================================================================
# 3. Main Script Execution
# =====================================================================

def main():
    # If --test is provided, run self-tests only and exit without calling the API
    if "--test" in sys.argv:
        run_self_tests()
        sys.exit(0)

    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    if not api_key:
        print("Error: LLM_API_KEY is not set in the environment or .env file.", file=sys.stderr)
        sys.exit(1)
    if not model_name:
        print("Error: LLM_MODEL is not set in the environment or .env file.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    prompt = "In two sentences, what does a pharmacy reorder level mean?"
    print(f"Sending prompt to model '{model_name}':\n\"{prompt}\"\n")

    try:
        response = call_with_retry(
            client.models.generate_content,
            model=model_name,
            contents=prompt,
            max_attempts=4,
            max_allowed_delay=60.0,
        )

        print("--- Response ---")
        print(response.text)
        print()

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            print("--- Token Usage ---")
            print(f"Prompt tokens: {getattr(meta, 'prompt_token_count', 'N/A')}")
            print(f"Response tokens: {getattr(meta, 'candidates_token_count', 'N/A')}")

            # Verify thoughts_token_count explicitly (preserving 0 if present)
            thoughts_count = getattr(meta, "thoughts_token_count", None)
            if thoughts_count is not None:
                print(f"Thoughts tokens: {thoughts_count}")
            else:
                print("Thoughts tokens: None")

            print(f"Total tokens: {getattr(meta, 'total_token_count', 'N/A')}")
        else:
            print("Token usage metadata not available in response.")

    except Exception as e:
        print(f"Fatal execution error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()