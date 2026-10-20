"""Ask the agent one question live and print what it did.

    py scripts/try_agent.py --id Q13
    py scripts/try_agent.py --id Q18                     (runs its setup turns first)
    py scripts/try_agent.py --question "which branches are low on paracetamol?"
    py scripts/try_agent.py --id Q17 --backend gemini    (uses Gemini requests)

Needs the tracker running, the Chroma store (default chroma_db/noctx), and Ollama or Gemini.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def print_calls(result) -> None:
    for r in result.tool_calls:
        status = f"ok, {r.rows} rows/blocks" if r.ok else f"ERROR {r.error}"
        print(f"  step {r.step}: {r.name}({json.dumps(r.arguments, ensure_ascii=False)}) -> {status}")
    if not result.tool_calls:
        print("  (none)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", help="a test-set question id, e.g. Q13")
    group.add_argument("--question")
    parser.add_argument("--backend", choices=["ollama", "gemini"], help="overrides LLM_BACKEND")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "chroma_db" / "noctx")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args(argv)

    expected, setup = None, []
    if args.id:
        test_set = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
        expected = next((q for q in test_set if q["id"] == args.id), None)
        if expected is None:
            print(f"ERROR: no question {args.id}")
            return 1
        setup = [t if isinstance(t, str) else t.get("content", "") for t in (expected.get("setup_turns") or [])]
        question = expected["question"]
    else:
        question = args.question

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    from app.agent import Agent
    from app.llm.factory import create_llm_client
    from app.rag.citations import kb_section_ids
    from app.rag.embedding import SentenceTransformerEmbedder
    from app.rag.store import retrieve
    from app.tools.tracker_client import TrackerClient, TrackerError

    tracker = TrackerClient()
    try:
        tracker.health()
    except TrackerError as error:
        print(f"ERROR: the tracker is not reachable: {error}")
        return 1

    embedder = SentenceTransformerEmbedder()
    agent = Agent(
        llm_client=create_llm_client(),
        tracker=tracker,
        kb_search=lambda query: retrieve(args.db, embedder, query, k=5),
        known_section_ids=kb_section_ids(REPO_ROOT / "docs" / "knowledge_base"),
        max_steps=args.max_steps,
    )
    results = agent.run_conversation(setup + [question])

    for prior in results[:-1]:
        print(f"setup turn: {prior.question}")
        print_calls(prior)
        print(f"  -> {prior.text}\n")

    result = results[-1]
    print(f"question: {question}")
    if expected:
        print(f"expected: route={expected['expected_route']} tool={expected.get('expected_tool')} "
              f"args={json.dumps(expected.get('expected_args'))} sources={expected.get('expected_sources')}")
    print("tool calls:")
    print_calls(result)
    print(f"\nanswer ({result.stopped}, {result.steps} steps):\n{result.text}\n")
    print(f"kb sources: {list(result.kb_sources)}")
    print(f"valid citations: {list(result.valid_citations)}")
    print(f"invalid citations: {[(c.raw, c.reason) for c in result.invalid_citations]}")
    print(f"refusal={result.refusal} medical_refusal={result.medical_refusal} scope_refusal={result.scope_refusal} "
          f"says_not_covered={result.says_not_covered}")
    print(f"repair attempted={result.repair_attempted} applied={result.repair_applied}")
    print(f"tokens prompt={result.prompt_tokens} completion={result.completion_tokens} "
          f"reasoning={result.reasoning_tokens} latency={(result.latency_seconds or 0):.1f}s model={result.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())