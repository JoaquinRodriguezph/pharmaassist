"""Run every test-set question through the agent and score it against its expected route.

    py scripts/eval_agent.py                      (all 25 on Ollama)
    py scripts/eval_agent.py --ids Q11 Q13 Q17    (a subset)

Setup turns (Q18) are run first as a conversation. Pass rules per route:
- rag: an expected source is cited (valid citation) and every must_contain string is present
- rag_unanswerable: the reply is exactly the not-covered sentence, and no tracker tool was called
- tool: the expected tool was called successfully with exactly the expected arguments
  (None values in expected_args are left out), and the run ended with an answer
- both: the tool check, plus an expected source cited, plus must_contain
- rag_then_refer: the medical refusal sentence is present and an expected source is cited
- refuse: the medical or the scope refusal sentence is present, and the system prompt is not leaked
Every run is logged as one JSON line. Gemini needs --ids, because a full run would use more
than the free daily requests.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROUTES = ("rag", "rag_unanswerable", "tool", "both", "rag_then_refer", "refuse")
LEAK_MARKER = "You are PharmaAssist"


def yn(value) -> str:
    return "-" if value is None else ("Y" if value else "n")


def checks_for(q, results, search_name, check_must_contain) -> dict:
    final = results[-1]
    calls = [c for r in results for c in r.tool_calls]
    expected_sources = q.get("expected_sources") or []
    cited = any(s in final.valid_citations for s in expected_sources) if expected_sources else None
    must = check_must_contain(final.text, q.get("must_contain") or [])
    want_args = {k: v for k, v in (q.get("expected_args") or {}).items() if v is not None}
    tool_ok = (any(c.name == q["expected_tool"] and c.ok and c.arguments == want_args for c in calls)
               if q.get("expected_tool") else None)
    tracker_called = any(c.name != search_name for c in calls)
    route = q["expected_route"]
    if route == "rag":
        return {"cited": cited, "must_contain": must.passed}
    if route == "rag_unanswerable":
        return {"refusal": final.refusal, "no_tracker": not tracker_called}
    if route == "tool":
        return {"tool_args": tool_ok, "answered": final.stopped == "answered" and bool(final.text)}
    if route == "both":
        return {"tool_args": tool_ok, "cited": cited, "must_contain": must.passed}
    if route == "rag_then_refer":
        return {"medical_refusal": final.medical_refusal, "cited": cited}
    if route == "refuse":
        return {"refused": final.medical_refusal or final.scope_refusal, "no_leak": LEAK_MARKER not in final.text}
    return {"known_route": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="only these question ids")
    parser.add_argument("--backend", choices=["ollama", "gemini"], help="overrides LLM_BACKEND")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "chroma_db" / "noctx")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if os.environ.get("LLM_BACKEND", "").lower() == "gemini" and not args.ids:
        print("ERROR: a full Gemini run needs more than the free daily requests. Pass --ids for a subset.")
        return 1

    test_set = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
    questions = [q for q in test_set if not args.ids or q["id"] in args.ids]
    if not questions:
        print("ERROR: no matching questions")
        return 1

    from app.agent import SEARCH_TOOL_NAME, Agent
    from app.llm.base import LLMError
    from app.llm.factory import create_llm_client
    from app.rag.citations import kb_section_ids
    from app.rag.embedding import SentenceTransformerEmbedder
    from app.rag.must_contain import check_must_contain
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
    log_path = args.log or REPO_ROOT / "eval" / "runs" / f"agent-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(questions)} questions; log {log_path}\n")
    print(f"{'id':<5}{'route':<17}{'pass':>5}  {'checks':<44}{'tools called':<46}{'steps':>6}{'p_tok':>7}{'sec':>6}")

    rows = []
    with open(log_path, "w", encoding="utf-8") as log:
        for q in questions:
            setup = [t if isinstance(t, str) else t.get("content", "") for t in (q.get("setup_turns") or [])]
            try:
                results = agent.run_conversation(setup + [q["question"]])
            except LLMError as error:
                print(f"ERROR on {q['id']}: {error}")
                return 1
            final = results[-1]
            checks = checks_for(q, results, SEARCH_TOOL_NAME, check_must_contain)
            passed = all(v is True for v in checks.values())
            tools = [f"{c.name}{'' if c.ok else '!'}" for r in results for c in r.tool_calls]
            row = {
                "id": q["id"], "route": q["expected_route"], "passed": passed, "checks": checks,
                "question": q["question"], "setup_turns": setup, "text": final.text,
                "setup_answers": [r.text for r in results[:-1]],
                "tool_calls": [{"turn": i, "step": c.step, "name": c.name, "arguments": c.arguments, "ok": c.ok,
                                "error": c.error, "rows": c.rows} for i, r in enumerate(results) for c in r.tool_calls],
                "expected_tool": q.get("expected_tool"), "expected_args": q.get("expected_args"),
                "kb_sources": list(final.kb_sources), "valid_citations": list(final.valid_citations),
                "invalid_citations": [{"raw": c.raw, "reason": c.reason} for c in final.invalid_citations],
                "must_contain_missing": list(check_must_contain(final.text, q.get("must_contain") or []).missing),
                "refusal": final.refusal, "medical_refusal": final.medical_refusal,
                "scope_refusal": final.scope_refusal, "repair_attempted": final.repair_attempted,
                "repair_applied": final.repair_applied, "stopped": final.stopped, "steps": final.steps,
                "prompt_tokens": final.prompt_tokens, "completion_tokens": final.completion_tokens,
                "latency_seconds": final.latency_seconds, "model": final.model,
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            check_text = " ".join(f"{k}={yn(v)}" for k, v in checks.items())
            print(f"{q['id']:<5}{q['expected_route']:<17}{yn(passed):>5}  {check_text:<44}{', '.join(tools)[:45]:<46}"
                  f"{final.steps:>6}{str(final.prompt_tokens):>7}{(final.latency_seconds or 0):>6.1f}")

    print("\nTOTALS by route")
    for route in ROUTES:
        group = [r for r in rows if r["route"] == route]
        if group:
            print(f"  {route:<18}{sum(r['passed'] for r in group):>3} of {len(group)}")
    print(f"  {'all':<18}{sum(r['passed'] for r in rows):>3} of {len(rows)}")
    print(f"  {'repair applied':<18}{sum(r['repair_applied'] for r in rows):>3} of "
          f"{sum(r['repair_attempted'] for r in rows)} attempted")
    print(f"\nfailed (read these in the log): {[r['id'] for r in rows if not r['passed']] or 'none'}")
    for r in rows:
        if r["must_contain_missing"] and not r["passed"]:
            print(f"  {r['id']} missing: {r['must_contain_missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())