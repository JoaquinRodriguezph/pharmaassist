"""Run the pre-registered batch through answer() on Ollama (docs/answer_notes.md, section 5b).

The 11 answerable original questions (Q01 to Q08, Q17, Q18, Q19) and the 6 independent
questions with no expected section. With --with-extra, also Q09 and Q10, reported in a
separate row. Run from the project root:

    py scripts/eval_answers.py [--db chroma_db/noctx] [--with-extra] [--log eval/runs/answers.jsonl]

Prints one row per question and the totals that section 5b predicts. Every call is logged as
one JSON line (question, system, prompt, reply, retrieved chunks, citations, flags, tokens),
so a bad answer can be inspected. Temperature 0, k = 5. Gemini is never used.
"""

import argparse
import json
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

from app.llm.base import LLMError
from app.rag.answer import answer
from app.rag.config import ENV_NAME, rag_with_context
from app.rag.evalsets import load_independent, original_questions, split_answerable

EXTRA_IDS = ("Q09", "Q10")


def yn(value) -> str:
    return "-" if value is None else ("Y" if value else "n")


def record(q, group, result) -> dict:
    return {
        "id": q.id, "group": group, "route": q.route, "question": q.question,
        "expected_sources": list(q.expected), "related_sections": list(q.related),
        "system": result.system, "prompt": result.prompt, "text": result.text,
        "retrieved": [{"id": c.id, "section_id": c.section_id, "score": round(c.score, 4)} for c in result.retrieved],
        "sources": list(result.sources),
        "valid_citations": list(result.valid_citations),
        "invalid_citations": [{"raw": c.raw, "reason": c.reason} for c in result.invalid_citations],
        "refusal": result.refusal, "says_not_covered": result.says_not_covered, "uncited_answer": result.uncited_answer,
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        "latency_seconds": result.latency_seconds, "model": result.model, "near_context_limit": result.near_context_limit,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "chroma_db" / "noctx")
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    parser.add_argument("--independent", type=Path, default=REPO_ROOT / "eval" / "retrieval_independent.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--with-extra", action="store_true", help="also Q09 and Q10, in a separate row")
    parser.add_argument("--log", type=Path, default=None, help="JSON-lines log (default eval/runs/answers-<time>.jsonl)")
    args = parser.parse_args(argv)

    test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    originals = original_questions(test_set)
    answerable, score_only = split_answerable(originals)
    _, independent = split_answerable(load_independent(args.independent))
    batch = [(q, "answerable") for q in answerable] + [(q, "independent") for q in independent]
    if args.with_extra:
        batch += [(q, "extra") for q in score_only if q.id in EXTRA_IDS]

    from app.llm.ollama_client import OllamaClient
    from app.rag.embedding import SentenceTransformerEmbedder
    from app.rag.store import StoreError, store_info

    try:
        info = store_info(args.db)
    except StoreError as error:
        print(f"ERROR: {error}")
        return 1
    if info["with_context"] is not rag_with_context():
        print(f"ERROR: store with_context={info['with_context']} but {ENV_NAME} says {rag_with_context()}.")
        return 1
    embedder = SentenceTransformerEmbedder()
    client = OllamaClient()
    log_path = args.log or REPO_ROOT / "eval" / "runs" / f"answers-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"store {args.db} ({info['count']} chunks, with_context={info['with_context']}); model {client.model}; "
          f"k={args.k}; temperature 0; log {log_path}\n")

    header = (f"{'id':<6}{'group':<12}{'route':<17}{'refusal':>8}{'notcov':>7}{'uncited':>8}{'valid':>6}"
              f"{'exp/rel cited':>14}{'rel in top5':>12}{'invalid':>8}{'p_tok':>7}{'c_tok':>6}{'sec':>6}")
    print(header)
    rows = []
    with open(log_path, "w", encoding="utf-8") as log:
        for q, group in batch:
            try:
                result = answer(q.question, args.k, client=client, db_path=args.db, embedder=embedder)
            except LLMError as error:
                print(f"ERROR on {q.id}: {error}")
                return 1
            row = record(q, group, result)
            if group == "answerable":
                row["expected_cited"] = any(e in result.valid_citations for e in q.expected)
                row["all_expected_cited"] = all(e in result.valid_citations for e in q.expected)
                cited, in_top = row["expected_cited"], None
            elif q.related:
                row["related_cited"] = any(r in result.valid_citations for r in q.related)
                row["related_in_top5"] = any(r in result.sources for r in q.related)
                cited, in_top = row["related_cited"], row["related_in_top5"]
            else:
                cited, in_top = None, None
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"{q.id:<6}{group:<12}{(q.route or '-'):<17}{yn(result.refusal):>8}{yn(result.says_not_covered):>7}"
                  f"{yn(result.uncited_answer):>8}{len(result.valid_citations):>6}{yn(cited):>14}{yn(in_top):>12}"
                  f"{len(result.invalid_citations):>8}{str(result.prompt_tokens):>7}{str(result.completion_tokens):>6}"
                  f"{(result.latency_seconds or 0):>6.1f}")

    ans = [r for r in rows if r["group"] == "answerable"]
    ind = [r for r in rows if r["group"] == "independent"]
    rel = [r for r in ind if r["route"] == "rag_unanswerable"]  # IND03 has a related section too, but it is a tool question
    main_rows = ans + ind
    counts = [
        ("answerable: at least one valid citation", sum(bool(r["valid_citations"]) for r in ans), len(ans)),
        ("answerable: cited an expected section", sum(r["expected_cited"] for r in ans), len(ans)),
        ("Q18: cited both expected sections", sum(r["all_expected_cited"] for r in ans if r["id"] == "Q18"), 1),
        ("answerable: pure refusal", sum(r["refusal"] for r in ans), len(ans)),
        ("all 17: at least one invalid citation", sum(bool(r["invalid_citations"]) for r in main_rows), len(main_rows)),
        ("independent: says_not_covered", sum(r["says_not_covered"] for r in ind), len(ind)),
        ("independent: pure refusal", sum(r["refusal"] for r in ind), len(ind)),
        ("rag_unanswerable: cited the related section", sum(r["related_cited"] for r in rel), len(rel)),
        ("rag_unanswerable: related section in the top 5", sum(r["related_in_top5"] for r in rel), len(rel)),
        ("all 17: near_context_limit", sum(bool(r["near_context_limit"]) for r in main_rows), len(main_rows)),
        ("all 17: uncited_answer (not predicted)", sum(r["uncited_answer"] for r in main_rows), len(main_rows)),
    ]
    print("\nTOTALS (compare with docs/answer_notes.md 5b)")
    for label, value, n in counts:
        print(f"  {label:<48}{value:>3} of {n}")
    tokens = [r["prompt_tokens"] for r in main_rows if r["prompt_tokens"] is not None]
    print(f"  {'all 17: largest prompt_tokens':<48}{max(tokens) if tokens else 'n/a':>3}")
    extra = [r for r in rows if r["group"] == "extra"]
    if extra:
        print(f"  {'extra (Q09, Q10): pure refusal':<48}{sum(r['refusal'] for r in extra):>3} of {len(extra)}")

    to_read = [r["id"] for r in ind if not r["says_not_covered"]]
    to_read += [r["id"] for r in main_rows if r["uncited_answer"] and r["id"] not in to_read]
    to_read += [r["id"] for r in main_rows if r["invalid_citations"] and r["id"] not in to_read]
    print(f"\nread by hand (independent without the sentence, uncited, or with invalid citations): {to_read or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())