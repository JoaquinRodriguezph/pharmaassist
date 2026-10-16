"""Provider-neutral retry with exponential backoff.

The loop never sees a provider's error format: callers pass a `classify`
function that says whether an error is retryable and how long to wait.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryDecision:
    """A classifier's verdict on one error."""

    retryable: bool
    # Seconds to wait before retrying. None means "use exponential backoff".
    wait_seconds: float | None = None
    reason: str = ""


def call_with_retry(
    fn: Callable[[], T],
    classify: Callable[[Exception], RetryDecision],
    *,
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()` up to `max_attempts` times, retrying errors `classify` allows.

    `sleep` is injectable so tests can run instantly. The original exception
    is re-raised when an error is not retryable or attempts run out.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            decision = classify(exc)

            if not decision.retryable:
                logger.warning(
                    "[Retry Engine] Non-retryable error (%s). Raising immediately.",
                    decision.reason,
                )
                raise

            if attempt >= max_attempts:
                logger.warning(
                    "[Retry Engine] Max retry attempts (%d) reached. Raising error.",
                    max_attempts,
                )
                raise

            if decision.wait_seconds is not None:
                wait = decision.wait_seconds
            else:
                wait = (2**attempt) + random.uniform(0.1, 1.0)

            logger.warning(
                "[Retry Engine] Attempt %d/%d failed: %s Waiting %.2fs...",
                attempt,
                max_attempts,
                decision.reason,
                wait,
            )
            sleep(wait)

    raise RuntimeError("unreachable")  # the loop always returns or raises