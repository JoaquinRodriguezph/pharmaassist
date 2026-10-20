"""Answer a question from the knowledge base with section-id citations.

`answer()` retrieves the top k chunks, builds the prompt (app/rag/prompting.py), makes one
LLM call, and checks the reply (app/rag/citations.py). It does not change, retry or repair
the reply: invalid citations and refusals are reported in the returned `Answer`.
Decisions: docs/answer_notes.md.

Retrieval goes through `store.retrieve` (and so through its embedder guard) unless a
`retrieve_fn` is passed; tests pass a fake. The store is imported only when it is used, so
importing this module does not need chromadb.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.llm.base import LLMClient
from app.rag.citations import (
    REFUSAL_SENTENCE,
    InvalidCitation,
    is_refusal,
    kb_section_ids,
    parse_citations,
    says_not_covered,
)
from app.rag.prompting import SYSTEM_PROMPT, build_user_message, group_sources

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "docs" / "knowledge_base"
DEFAULT_CONTEXT_LIMIT = 4096  # Ollama's running context on this machine (docs/local_model_notes.md)
CONTEXT_MARGIN = 300  # flag a call this close to the limit; my choice, see docs/answer_notes.md


@dataclass(frozen=True)
class Answer:
    question: str
    text: str  # the model's reply, stripped, otherwise unchanged
    retrieved: tuple  # the retrieved chunks, rank order, with scores
    sources: tuple[str, ...]  # section ids shown to the model, in block order
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[InvalidCitation, ...]
    refusal: bool  # the whole reply is the fixed sentence
    says_not_covered: bool  # the fixed sentence appears anywhere
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    latency_seconds: float | None
    model: str | None
    near_context_limit: bool | None  # None when the backend reported no token counts
    system: str
    prompt: str  # the user message sent


def _near_limit(prompt_tokens, completion_tokens, context_limit: int) -> bool | None:
    if prompt_tokens is None or completion_tokens is None:
        return None
    return prompt_tokens + completion_tokens >= context_limit - CONTEXT_MARGIN


def answer(
    question: str,
    k: int = 5,
    *,
    client: LLMClient,
    db_path: Path,
    embedder,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    kb_dir: Path = DEFAULT_KB_DIR,
    retrieve_fn: Callable | None = None,
) -> Answer:
    """Retrieve `k` chunks, ask `client` to answer only from them, and check the citations.

    Raises ValueError for an empty question. If nothing is retrieved (an empty store), no LLM
    call is made: the text is the fixed sentence and both refusal flags are True. Errors
    from retrieval or from the LLM propagate unchanged.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if retrieve_fn is None:
        from app.rag.store import retrieve as retrieve_fn  # imported here so tests need no chromadb

    retrieved = tuple(retrieve_fn(db_path, embedder, question, k=k))
    blocks = group_sources(retrieved)
    prompt = build_user_message(question, blocks)
    sources = tuple(b.section_id for b in blocks)

    if not retrieved:
        return Answer(
            question=question, text=REFUSAL_SENTENCE, retrieved=(), sources=(),
            valid_citations=(), invalid_citations=(), refusal=True, says_not_covered=True,
            prompt_tokens=None, completion_tokens=None, reasoning_tokens=None,
            latency_seconds=None, model=None, near_context_limit=None,
            system=SYSTEM_PROMPT, prompt=prompt,
        )

    response = client.generate(
        [{"role": "user", "content": prompt}],
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.text.strip()
    citations = parse_citations(text, sources, kb_section_ids(kb_dir))
    return Answer(
        question=question,
        text=text,
        retrieved=retrieved,
        sources=sources,
        valid_citations=citations.valid,
        invalid_citations=citations.invalid,
        refusal=is_refusal(text),
        says_not_covered=says_not_covered(text),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
        latency_seconds=response.latency_seconds,
        model=response.model,
        near_context_limit=_near_limit(response.prompt_tokens, response.completion_tokens, context_limit),
        system=SYSTEM_PROMPT,
        prompt=prompt,
    )