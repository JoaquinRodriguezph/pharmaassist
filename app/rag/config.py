"""Retrieval settings read from the environment.

`RAG_WITH_CONTEXT` decides whether chunk text is embedded with its
"Document: ... / Section: ..." prefix (see `Chunk.embed_text`).

The default is **off**. That follows the tie-break I wrote down before the
Step 4c run: the prefix did not clearly help, so the simpler setting wins.
It is a setting and not a hardcoded choice because the measured numbers
leaned the other way (see docs/retrieval_eval.md), and I will test it again
once answers can be graded.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

ENV_NAME = "RAG_WITH_CONTEXT"
DEFAULT_WITH_CONTEXT = False

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def rag_with_context(environ: Mapping[str, str] | None = None) -> bool:
    """The prefix setting from RAG_WITH_CONTEXT; DEFAULT_WITH_CONTEXT if unset or blank.

    Accepts 1/true/yes/on and 0/false/no/off in any capitalization. Anything
    else raises ValueError, because a typo silently falling back to the
    default would make a store look like it used a setting it did not.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_NAME, "").strip()
    if not raw:
        return DEFAULT_WITH_CONTEXT
    value = raw.casefold()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{ENV_NAME}={raw!r} is not a yes/no value. Use one of: "
        f"{', '.join(sorted(_TRUE))} or {', '.join(sorted(_FALSE))}."
    )


def resolve_with_context(cli_choice: bool | None, environ: Mapping[str, str] | None = None) -> bool:
    """A command-line choice (--context / --no-context) beats the environment, which beats the default."""
    return cli_choice if cli_choice is not None else rag_with_context(environ)