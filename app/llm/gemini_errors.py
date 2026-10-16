"""Google Gemini error parsing and retry rules.

Google's error format is parsed here, not in retry.py, so that the retry loop and the Ollama backend never import google-genai; only the Gemini backend imports this module.
"""

from __future__ import annotations

from google.genai.errors import APIError

from app.llm.retry import RetryDecision


def get_details_list(details) -> list:
    """Return the inner details list from `APIError.details`, or [].

    In the Google GenAI SDK, `error.details` is the full parsed response JSON:
    {'error': {'code': 429, 'status': '...', 'details': [...]}}
    This navigates to the inner list without crashing if keys are missing.
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


def parse_retry_delay(details) -> float | None:
    """Return RetryInfo's retryDelay in seconds, or None.

    Looks for {'@type': 'type.googleapis.com/google.rpc.RetryInfo',
    'retryDelay': '56026s'} in the inner details list.
    """
    for item in get_details_list(details):
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


def is_daily_quota(details) -> bool:
    """True if a google.rpc.QuotaFailure violation names a per-day quota."""
    for item in get_details_list(details):
        if isinstance(item, dict):
            item_type = item.get("@type", "")
            if "QuotaFailure" in item_type:
                violations = item.get("violations", [])
                if isinstance(violations, list):
                    for v in violations:
                        if isinstance(v, dict):
                            quota_id = v.get("quotaId", "")
                            desc = v.get("description", "")
                            if (
                                "PerDay" in quota_id
                                or "PerDay" in desc
                                or "per_day" in quota_id.lower()
                            ):
                                return True
    return False


def classify_gemini_error(
    error: Exception, max_allowed_delay: float = 60.0
) -> RetryDecision:
    """Decide whether a Gemini error is retryable and how long to wait."""
    if isinstance(error, APIError):
        code = getattr(error, "code", None)
        details = getattr(error, "details", None)

        # 400, 401, 403, 404: fatal client/auth errors (never retry)
        if code in (400, 401, 403, 404):
            return RetryDecision(False, None, f"Client error HTTP {code}: not retryable.")

        # 503: service unavailable (transient backend outage)
        if code == 503:
            return RetryDecision(
                True, None, "HTTP 503: Service Unavailable (transient server error)."
            )

        # 429: rate limit / resource exhausted
        if code == 429:
            if is_daily_quota(details):
                return RetryDecision(
                    False,
                    None,
                    "Daily quota limit reached (QuotaFailure contains 'PerDay'). Failing fast.",
                )

            delay = parse_retry_delay(details)
            if delay is not None:
                if delay > max_allowed_delay:
                    return RetryDecision(
                        False,
                        None,
                        f"429 retryDelay of {delay}s exceeds threshold "
                        f"({max_allowed_delay}s). Failing fast.",
                    )
                return RetryDecision(
                    True, delay, f"429 transient rate limit with server delay of {delay}s."
                )

            return RetryDecision(
                True, None, "429 transient rate limit (using exponential backoff)."
            )

        return RetryDecision(False, None, f"Unhandled HTTP {code} APIError.")

    # Programming bugs (TypeError, NameError, etc.) are never retried
    return RetryDecision(
        False, None, f"Non-retryable exception type: {type(error).__name__}."
    )