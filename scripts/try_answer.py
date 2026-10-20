"""Try answer() on single questions: a dry run, the "insurance plan" demonstration, or one question.

Run from the project root. The store must be built without the prefix (the default), e.g.
`py scripts/ingest_kb.py --db chroma_db/noctx --no-context`.

    py scripts/try_answer.py --dry-run --id Q05     # real retrieval, fake LLM: prints the prompt, no Ollama call
    py scripts/try_answer.py --insurance            # conditions A, B, C, two runs each (docs/answer_notes.md 5a)
    py scripts/try_answer.py --id Q05               # one test-set question through answer() on Ollama
    py scripts/try_answer.py "any question"         # free text

The model comes from OLLAMA_MODEL and OLLAMA_HOST (.env). Temperature 0. Gemini is never used.
Each run is judged only by mechanical checks, printed as PASS or FAIL.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app.llm.base import LLMClient, LLMError, LLMResponse
from app.rag.answer import answer
from app.rag.citations import is_refusal, kb_section_ids, parse_citations, says_not_covered
from app.rag.config import ENV_NAME, rag_with_context
from app.rag.prompting import SYSTEM_PROMPT, build_user_message

INSURANCE_QUESTION = "In two sentences, what does a pharmacy reorder level mean?"
KB_DIR = REPO_ROOT / "docs" / "knowledge_base"


class PromptOnlyClient(LLMClient):
    """For --dry-run: makes no call and returns an empty reply."""

    def generate(self, messages, system=None, temperature=0.2, max_tokens=None):
        return LLMResponse(text="", model="dry-run (no LLM call)")


def open_store(db: Path):
    """Check the store exists and matches the RAG_WITH_CONTEXT setting; return the embedder."""
    from app.rag.embedding import SentenceTransformerEmbedder
    from app.rag.store import store_info

    info = store_info(db)
    wanted = rag_with_context()
    print(f"store {db}: {info['count']} chunks, with_context={info['with_context']} ({ENV_NAME} says {wanted})")
    if info["with_context"] is not wanted:
        raise SystemExit(f"ERROR: this store was built with with_context={info['with_context']}, "
                         f"but {ENV_NAME} says {wanted}. Use the matching store.")
    return SentenceTransformerEmbedder()


def show_answer(result, label: str) -> None:
    print(f"--- {label}")
    for rank, chunk in enumerate(result.retrieved, 1):
        print(f"    retrieved {rank}: {chunk.id}  score {chunk.score:.3f}")
    print(f"    sources shown: {list(result.sources)}")
    print(f"    TEXT: {result.text!r}")
    print(f"    refusal={result.refusal} says_not_covered={result.says_not_covered} uncited_answer={result.uncited_answer}")
    print(f"    valid citations: {list(result.valid_citations)}")
    print(f"    invalid citations: {[(c.raw, c.reason) for c in result.invalid_citations]}")
    print(f"    model={result.model} prompt_tokens={result.prompt_tokens} completion_tokens={result.completion_tokens} "
          f"latency={result.latency_seconds} near_context_limit={result.near_context_limit}")


def verdict(checks: dict) -> str:
    failed = [name for name, ok in checks.items() if not ok]
    return "PASS" if not failed else "FAIL (" + ", ".join(failed) + ")"


def run_insurance(client, db, embedder, runs: int) -> None:
    known = kb_section_ids(KB_DIR)
    texts = {"A": [], "B": [], "C": []}
    print(f"\nQUESTION: {INSURANCE_QUESTION}\n")
    for run in range(1, runs + 1):
        # A. plain: no system prompt, no sources (as in the sampling test)
        a = client.generate([{"role": "user", "content": INSURANCE_QUESTION}], system=None, temperature=0.0)
        texts["A"].append(a.text.strip())
        print(f"--- A plain, run {run}\n    TEXT: {a.text.strip()!r}")
        print(f"    contains 'insurance': {'insurance' in a.text.lower()}  prompt_tokens={a.prompt_tokens} "
              f"completion_tokens={a.completion_tokens}")

        # B. rules only: the system prompt, no sources
        b = client.generate([{"role": "user", "content": build_user_message(INSURANCE_QUESTION, [])}],
                            system=SYSTEM_PROMPT, temperature=0.0, max_tokens=512)
        b_text = b.text.strip()
        b_valid = parse_citations(b_text, (), known).valid
        texts["B"].append(b_text)
        print(f"--- B rules only, run {run}\n    TEXT: {b_text!r}")
        print(f"    refusal={is_refusal(b_text)} says_not_covered={says_not_covered(b_text)} "
              f"contains 'insurance': {'insurance' in b_text.lower()}  valid citations: {list(b_valid)}")

        # C. RAG through answer()
        c = answer(INSURANCE_QUESTION, client=client, db_path=db, embedder=embedder)
        texts["C"].append(c.text)
        show_answer(c, f"C RAG, run {run}")
        print("    C mechanical check: " + verdict({
            "refusal is True": c.refusal,
            "'insurance' absent": "insurance" not in c.text.lower(),
            "valid_citations empty": not c.valid_citations,
        }))
    for condition, seen in texts.items():
        print(f"condition {condition}: runs identical: {len(set(seen)) == 1}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="free-text question")
    parser.add_argument("--id", help="a test-set question id, e.g. Q05")
    parser.add_argument("--insurance", action="store_true", help="the 'insurance plan' demonstration")
    parser.add_argument("--dry-run", action="store_true", help="real retrieval, no LLM call; prints the prompt")
    parser.add_argument("--runs", type=int, default=None, help="runs per question (default 2 for --insurance, else 1)")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "chroma_db" / "noctx")
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    args = parser.parse_args(argv)
    if sum(bool(x) for x in (args.question, args.id, args.insurance)) != 1:
        parser.error("give exactly one of: a question, --id, --insurance")
    runs = args.runs or (2 if args.insurance else 1)

    expected = []
    question = args.question or INSURANCE_QUESTION
    if args.id:
        by_id = {q["id"]: q for q in json.loads(args.test_set.read_text(encoding="utf-8"))}
        if args.id not in by_id:
            parser.error(f"no question {args.id!r} in {args.test_set.name}")
        question, expected = by_id[args.id]["question"], by_id[args.id].get("expected_sources") or []

    from app.rag.store import StoreError

    try:
        embedder = open_store(args.db)
    except StoreError as error:
        print(f"ERROR: {error}")
        return 1
    if args.dry_run:
        result = answer(question, client=PromptOnlyClient(), db_path=args.db, embedder=embedder)
        print("=== SYSTEM ===\n" + result.system + "\n=== PROMPT ===\n" + result.prompt)
        print("=== retrieved ===")
        for rank, chunk in enumerate(result.retrieved, 1):
            print(f"  {rank}: {chunk.id}  score {chunk.score:.3f}")
        return 0

    from app.llm.ollama_client import OllamaClient

    client = OllamaClient()
    try:
        if args.insurance:
            run_insurance(client, args.db, embedder, runs)
            return 0
        print(f"\nQUESTION ({args.id or 'free text'}): {question}")
        for run in range(1, runs + 1):
            result = answer(question, client=client, db_path=args.db, embedder=embedder)
            show_answer(result, f"run {run}")
            if expected:
                print("    mechanical check: " + verdict({
                    "cites an expected section": any(e in result.valid_citations for e in expected),
                    "refusal is False": not result.refusal,
                    "says_not_covered is False": not result.says_not_covered,
                }))
    except LLMError as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())