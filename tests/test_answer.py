"""Tests for app.rag.answer with a fake LLM client and a fake store. No model, no Chroma."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.llm.base import LLMClient, LLMConnectionError, LLMResponse
from app.rag.answer import CONTEXT_MARGIN, Answer, answer
from app.rag.citations import NOT_IN_KB, NOT_RETRIEVED, REFUSAL_SENTENCE, InvalidCitation
from app.rag.prompting import SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent

TEMP = "storage_sop.md#Temperature-Monitoring"
RETURNS = "returns_and_expiry_policy.md#Customer-Returns"
FEFO = "storage_sop.md#Receiving-And-FEFO"


def chunk(section_id, part, text, score):
    return SimpleNamespace(id=f"{section_id}::{part}", text=text, section_id=section_id,
                           section=section_id.split("#")[1].replace("-", " "), source=section_id.split("#")[0],
                           score=score, distance=1 - score)


DEFAULT_CHUNKS = [
    chunk(TEMP, 1, "Check and record temperatures twice daily, at 8:00 AM and at 4:00 PM.", 0.71),
    chunk(RETURNS, 2, "Cold-chain items are excluded.", 0.52),
    chunk(RETURNS, 1, "Unopened OTC items may be returned within 7 calendar days.", 0.49),
]


class FakeStore:
    """Stands in for store.retrieve: records each call and returns fixed chunks."""

    def __init__(self, chunks=None):
        self.chunks = DEFAULT_CHUNKS if chunks is None else chunks
        self.calls = []

    def __call__(self, db_path, embedder, question, k):
        self.calls.append((db_path, embedder, question, k))
        return list(self.chunks)[:k]


class FakeClient(LLMClient):
    def __init__(self, text="", prompt_tokens=420, completion_tokens=30, error=None):
        self.text, self.prompt_tokens, self.completion_tokens, self.error = text, prompt_tokens, completion_tokens, error
        self.calls = []

    def generate(self, messages, system=None, temperature=0.2, max_tokens=None):
        self.calls.append({"messages": messages, "system": system, "temperature": temperature, "max_tokens": max_tokens})
        if self.error:
            raise self.error
        return LLMResponse(text=self.text, prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                           latency_seconds=0.5, model="fake-llm", reasoning_tokens=None)


def run(text="", store=None, **kwargs):
    client = kwargs.pop("client", None) or FakeClient(text)
    store = store or FakeStore()
    result = answer(kwargs.pop("question", "When do we record temperatures?"), client=client,
                    db_path=Path("fake-db"), embedder="fake-embedder", retrieve_fn=store, **kwargs)
    return result, client, store


# ---------------------------------------------------------------------------
# What goes in
# ---------------------------------------------------------------------------

def test_retrieval_gets_the_question_k_store_and_embedder():
    _, _, store = run(f"Twice daily [{TEMP}].", k=3)

    assert store.calls == [(Path("fake-db"), "fake-embedder", "When do we record temperatures?", 3)]


def test_the_default_k_is_five():
    _, _, store = run(f"Twice daily [{TEMP}].")

    assert store.calls[0][3] == 5


def test_one_llm_call_with_the_system_prompt_the_sources_and_the_settings():
    result, client, _ = run(f"Twice daily [{TEMP}].", temperature=0.0, max_tokens=256)

    (call,) = client.calls
    assert call["system"] == SYSTEM_PROMPT == result.system
    assert call["messages"] == [{"role": "user", "content": result.prompt}]
    assert (call["temperature"], call["max_tokens"]) == (0.0, 256)
    assert result.prompt.startswith("Sources:\n\nSource [storage_sop.md#Temperature-Monitoring]")
    assert result.prompt.endswith("Question: When do we record temperatures?")


def test_parts_of_a_section_are_grouped_and_sources_follow_block_order():
    result, _, _ = run(f"Twice daily [{TEMP}].")

    assert result.sources == (TEMP, RETURNS)
    assert result.prompt.index("Unopened OTC items") < result.prompt.index("Cold-chain items are excluded")
    assert [c.id for c in result.retrieved] == [f"{TEMP}::1", f"{RETURNS}::2", f"{RETURNS}::1"]  # rank order kept


# ---------------------------------------------------------------------------
# What comes out
# ---------------------------------------------------------------------------

def test_valid_and_invalid_citations_are_reported_and_the_text_is_unchanged():
    text = f"  Twice daily [{TEMP}]. Seven days [{RETURNS}]. Inspect deliveries [{FEFO}]. [example.md#Example-Section]  "

    result, _, _ = run(text)

    assert result.text == text.strip()
    assert result.valid_citations == (TEMP, RETURNS)
    assert result.invalid_citations == (
        InvalidCitation(FEFO, NOT_RETRIEVED),
        InvalidCitation("example.md#Example-Section", NOT_IN_KB),
    )


def test_a_pure_refusal_sets_both_flags():
    result, _, _ = run(REFUSAL_SENTENCE)

    assert result.refusal and result.says_not_covered
    assert result.valid_citations == () and result.invalid_citations == ()


def test_a_partial_answer_says_not_covered_but_is_not_a_refusal():
    result, _, _ = run(f"Readings are at 8:00 AM and 4:00 PM [{TEMP}]. {REFUSAL_SENTENCE}")

    assert result.says_not_covered and not result.refusal
    assert result.valid_citations == (TEMP,)


def test_a_full_answer_sets_neither_flag():
    result, _, _ = run(f"Twice daily [{TEMP}].")

    assert not result.refusal and not result.says_not_covered


def test_token_counts_latency_and_model_are_copied_from_the_llm_response():
    result, _, _ = run(f"Twice daily [{TEMP}].", client=FakeClient(f"[{TEMP}]", prompt_tokens=612, completion_tokens=44))

    assert (result.prompt_tokens, result.completion_tokens, result.reasoning_tokens) == (612, 44, None)
    assert (result.latency_seconds, result.model) == (0.5, "fake-llm")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Readings are taken twice daily.", id="no-brackets"),
        pytest.param("Readings are taken twice daily [Temperature-Monitoring].", id="section-name-without-file"),
        pytest.param(f"Readings are taken twice daily. {REFUSAL_SENTENCE}", id="partial-answer-without-citations"),
    ],
)
def test_an_answer_with_no_citation_of_any_kind_is_flagged_uncited(text):
    result, _, _ = run(text)

    assert result.uncited_answer


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(REFUSAL_SENTENCE, id="pure-refusal"),
        pytest.param(f"Twice daily [{TEMP}].", id="valid-citation"),
        pytest.param("Twice daily [returns_and_expiry_policy.md].", id="only-an-invalid-citation"),
    ],
)
def test_a_refusal_or_any_citation_is_not_uncited(text):
    result, _, _ = run(text)

    assert not result.uncited_answer


def test_an_empty_store_is_not_an_uncited_answer():
    result, _, _ = run("never used", store=FakeStore(chunks=[]))

    assert not result.uncited_answer


# ---------------------------------------------------------------------------
# The context-limit flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt_tokens, completion_tokens, expected",
    [
        pytest.param(600, 50, False, id="far-below"),
        pytest.param(4096 - CONTEXT_MARGIN - 51, 50, False, id="just-below-the-margin"),
        pytest.param(4096 - CONTEXT_MARGIN - 50, 50, True, id="at-the-margin"),
        pytest.param(4096, 0, True, id="at-the-limit"),
        pytest.param(None, 50, None, id="no-prompt-count"),
        pytest.param(600, None, None, id="no-completion-count"),
    ],
)
def test_a_call_near_the_context_limit_is_flagged(prompt_tokens, completion_tokens, expected):
    client = FakeClient(f"[{TEMP}]", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    result, _, _ = run(client=client)

    assert result.near_context_limit is expected


def test_a_larger_context_limit_can_be_passed_for_another_backend():
    client = FakeClient(f"[{TEMP}]", prompt_tokens=5000, completion_tokens=100)

    result, _, _ = run(client=client, context_limit=1_000_000)

    assert result.near_context_limit is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_an_empty_question_is_an_error_and_nothing_is_called(question):
    client, store = FakeClient("x"), FakeStore()

    with pytest.raises(ValueError, match="question"):
        answer(question, client=client, db_path=Path("fake-db"), embedder="e", retrieve_fn=store)
    assert client.calls == [] and store.calls == []


def test_an_empty_store_makes_no_llm_call_and_returns_the_fixed_sentence():
    result, client, _ = run("never used", store=FakeStore(chunks=[]))

    assert client.calls == []
    assert result.text == REFUSAL_SENTENCE and result.refusal and result.says_not_covered
    assert result.sources == () and result.retrieved == ()
    assert result.prompt_tokens is None and result.near_context_limit is None
    assert result.prompt.startswith("Sources: (none)")


def test_an_llm_error_propagates_unchanged():
    client = FakeClient(error=LLMConnectionError("Ollama is not running"))

    with pytest.raises(LLMConnectionError, match="not running"):
        run(client=client)


def test_the_answer_is_a_frozen_record():
    result, _, _ = run(f"[{TEMP}]")

    assert isinstance(result, Answer)
    with pytest.raises(Exception):
        result.text = "changed"


def test_importing_answer_does_not_import_chromadb():
    code = (
        "import sys\n"
        "sys.modules['chromadb'] = None\n"  # makes `import chromadb` fail
        "import app.rag.answer\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout