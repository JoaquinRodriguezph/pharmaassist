"""Measure how often the right section is in the top k, with and without the context prefix.

First build two stores from the same knowledge base (800 characters, no overlap):

    py scripts/ingest_kb.py --db chroma_db/ctx
    py scripts/ingest_kb.py --db chroma_db/noctx --no-context

Then, from the project root:

    py scripts/eval_retrieval.py [--sets original variants independent] [--k 5]

For each question set it prints, side by side for both stores, every question's
rank (of the first chunk from an expected section) and whether it is a hit at
1, 3 and 5; then counts ("9 of 11") for hit@k, all-hit@k, MRR, and how many of
the top slots are taken by repeated chunks of one section. Questions the
knowledge base cannot answer are reported by their top scores only; no
threshold is derived from them. Metric definitions: app/rag/metrics.py.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.evalsets import load_independent, load_variants, original_questions, split_answerable
from app.rag.metrics import collapse_to_sections, duplicate_slots, first_hit_rank, hit_at_k, summarize
from app.rag.store import StoreError, retrieve, store_info

HIT_KS = (1, 3, 5)
STORES = ("ctx", "noctx")


def short(section_id: str) -> str:
    return section_id.split("#", 1)[-1]


def rank_text(rank, k) -> str:
    return str(rank) if rank is not None else f">{k}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx-db", type=Path, default=REPO_ROOT / "chroma_db" / "ctx")
    parser.add_argument("--noctx-db", type=Path, default=REPO_ROOT / "chroma_db" / "noctx")
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    parser.add_argument("--variants", type=Path, default=REPO_ROOT / "eval" / "retrieval_variants.json")
    parser.add_argument("--independent", type=Path, default=REPO_ROOT / "eval" / "retrieval_independent.json")
    parser.add_argument("--sets", nargs="+", choices=["original", "variants", "independent"], default=["original", "variants"])
    parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve (default 5)")
    args = parser.parse_args(argv)
    if args.k < max(HIT_KS):
        print(f"ERROR: --k must be at least {max(HIT_KS)}")
        return 1
    dbs = {"ctx": args.ctx_db, "noctx": args.noctx_db}

    try:
        test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
        question_sets = {}
        for name in args.sets:
            if name == "original":
                question_sets[name] = original_questions(test_set)
            elif name == "variants":
                question_sets[name] = load_variants(args.variants, test_set)
            else:
                question_sets[name] = load_independent(args.independent)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    try:
        from app.rag.embedding import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except ImportError:
        print("ERROR: sentence-transformers is not installed. See docs/retrieval_notes.md.")
        return 1

    try:
        for label, wanted in (("ctx", True), ("noctx", False)):
            info = store_info(dbs[label])
            print(f"{label:<6} {dbs[label]} | {info['count']} chunks | {info['embedder_name']} | with_context={info['with_context']}")
            if info["with_context"] is not wanted:
                print(f"ERROR: the '{label}' store must be built {'with' if wanted else 'without'} the prefix "
                      f"(scripts/ingest_kb.py{'' if wanted else ' --no-context'} --db <path>).")
                return 1
        print(f"retrieving the top {args.k} chunks per question\n")

        for name, questions in question_sets.items():
            answerable, unanswerable = split_answerable(questions)
            results = {
                label: {q.id: retrieve(dbs[label], embedder, q.question, k=args.k) for q in questions}
                for label in STORES
            }
            rankings = {
                label: {qid: [r.section_id for r in found] for qid, found in per_question.items()}
                for label, per_question in results.items()
            }
            print_set(name, answerable, unanswerable, results, rankings, args.k)
    except StoreError as error:
        print(f"ERROR: {error}")
        return 1
    return 0


def print_set(name, answerable, unanswerable, results, rankings, k) -> None:
    print(f"=== {name}: {len(answerable)} question(s) with an expected section, top {k} retrieved")
    if answerable:
        print(f"{'id':<7}{'about':<36}{'ctx rank':>9}{'1/3/5':>7}{'noctx rank':>12}{'1/3/5':>7}{'repeats ctx/noctx':>20}")
        for q in answerable:
            cells = {}
            for label in STORES:
                ranking = rankings[label][q.id]
                rank = first_hit_rank(ranking, q.expected)
                cells[label] = (
                    rank_text(rank, k),
                    "".join("Y" if hit_at_k(ranking, q.expected, hk) else "N" for hk in HIT_KS),
                    duplicate_slots(ranking, k),
                )
            about = q.style or (", ".join(short(s) for s in q.expected))
            print(f"{q.id:<7}{about[:34]:<36}{cells['ctx'][0]:>9}{cells['ctx'][1]:>7}{cells['noctx'][0]:>12}{cells['noctx'][1]:>7}"
                  f"{str(cells['ctx'][2]) + ' / ' + str(cells['noctx'][2]):>20}")

        summary = {
            label: summarize([(rankings[label][q.id], q.expected) for q in answerable], HIT_KS)
            for label in STORES
        }
        n = summary["ctx"]["n"]
        print(f"\ncounts out of {n}{'':<20}{'ctx':>10}{'noctx':>10}")
        for hk in HIT_KS:
            print(f"  hit@{hk:<28}{summary['ctx']['hits'][hk]:>3} of {n:<3}{summary['noctx']['hits'][hk]:>3} of {n}")
        if any(len(q.expected) > 1 for q in answerable):
            for hk in HIT_KS:
                print(f"  all-hit@{hk:<24}{summary['ctx']['all_hits'][hk]:>3} of {n:<3}{summary['noctx']['all_hits'][hk]:>3} of {n}")
        print(f"  MRR (chunk ranks){'':<12}{summary['ctx']['mrr']:>10.3f}{summary['noctx']['mrr']:>10.3f}")
        print(f"  MRR (distinct sections){'':<6}{summary['ctx']['mrr_distinct']:>10.3f}{summary['noctx']['mrr_distinct']:>10.3f}")
        print(f"  top-{k} slots taken by repeats{'':<2}{summary['ctx']['repeat_slots']:>10}{summary['noctx']['repeat_slots']:>10}"
              f"   (in {summary['ctx']['questions_with_repeats']} / {summary['noctx']['questions_with_repeats']} questions)")

        for label in STORES:
            misses = []
            for q in answerable:
                ranking = rankings[label][q.id]
                rank = first_hit_rank(ranking, q.expected)
                if rank != 1:
                    top = results[label][q.id][0]
                    misses.append(f"    {q.id}: rank {rank_text(rank, k)}; rank 1 was {top.section_id} (score {top.score:.3f})")
            print(f"\n  not at rank 1 ({label}): {len(misses)} of {n}")
            print("\n".join(misses))

    if unanswerable:
        print("\nQuestions the knowledge base cannot answer (top scores only; no threshold is drawn from these):")
        for q in unanswerable:
            print(f"  {q.id}: {q.question[:78]}")
            for label in STORES:
                found = results[label][q.id]
                print(f"    {label:<6} top-1 {found[0].score:.3f} ({found[0].section_id})   top-3 "
                      + " ".join(f"{r.score:.3f}" for r in found[:3]))
    print()


if __name__ == "__main__":
    sys.exit(main())