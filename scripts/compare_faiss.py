"""Compare the Chroma store with a FAISS index over the same chunks, on the real model.

Needs the two Chroma stores from scripts/ingest_kb.py (see docs/retrieval_eval.md):

    py scripts/ingest_kb.py --db chroma_db/ctx --context
    py scripts/ingest_kb.py --db chroma_db/noctx --no-context

Then, from the project root:

    py scripts/compare_faiss.py [--settings noctx ctx] [--k 5] [--repeats 500] [--skip-timing] [--from-chroma-vectors]

For each prefix setting:

1. **Vectors.** Chunk the knowledge base the way ingest_kb does (800 characters, no
   overlap) and embed every chunk's text once. Read the stored vectors back from
   Chroma (`collection.get(include=["embeddings"])`, page by page) and print count,
   dtype, shape and the min and max norm of both sets, and the largest absolute
   difference after normalizing both. FAISS is built from **exactly the re-embedded
   vectors that were checked** (the default), or with `--from-chroma-vectors` from
   Chroma's stored vectors, which takes the embedding step out of the comparison
   altogether. The index is saved to a temporary folder, loaded back, and the loaded
   index is searched.
2. **Questions.** Each question is embedded **once** and the same vector is passed to
   both backends (`collection.query(query_embeddings=...)` and `search_vector`). For
   every question it reports whether the ranked chunk ids are identical and, if not,
   whether the first difference is a tie under the rule in app/rag/rank_compare.py
   (written in docs/vector_store_comparison.md before the first run); then the largest
   score difference, and hit@k and MRR side by side from app/rag/metrics.py.
3. **Timing** (each setting). The search alone, on the pre-embedded queries. The three
   rows do different amounts of work; see the note printed with them.

This script reaches into private helpers on purpose: `store._open_for_reading` and
`store._check_embedder` (Chroma) and `index._index` (the bare FAISS index, for the
timing row). They are not part of any public interface.

Exit status 1 if any question's ranked ids differ between the backends, tie or not.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag import store as store_module
from app.rag.chunking import chunk_kb
from app.rag.evalsets import load_independent, load_variants, original_questions, split_answerable
from app.rag.faiss_index import FaissChunkIndex, normalized_max_abs_difference, vector_stats
from app.rag.metrics import summarize
from app.rag.rank_compare import SCORE_TOLERANCE, compare_results
from app.rag.store import Retrieved, StoreError, store_info

HIT_KS = (1, 3, 5)
MAX_CHARS, OVERLAP = 800, 0  # what ingest_kb uses by default
PAGE = 500


def describe(hits) -> str:
    return "  ".join(f"{h.id.split('#', 1)[-1]} {h.score:.6f}" for h in hits)


def show_stats(label: str, stats: dict) -> None:
    print(f"    {label:<30} count {stats['count']}, dtype {stats['dtype']}, shape {stats['shape']}, "
          f"norm min {stats['min_norm']:.7f} max {stats['max_norm']:.7f}")


def stored_embeddings(collection) -> dict:
    """{chunk id: stored vector}, read page by page. Uses len(), never the truth value of an array."""
    found = {}
    offset = 0
    while True:
        got = collection.get(include=["embeddings"], limit=PAGE, offset=offset)
        ids, embeddings = got["ids"], got["embeddings"]
        if embeddings is None or len(ids) != len(embeddings):
            raise StoreError(f"Chroma returned {len(ids)} ids but no matching embeddings")
        found.update(zip(ids, embeddings))
        if len(ids) < PAGE:
            return found
        offset += PAGE


def chroma_search_vector(collection, vector, k: int) -> list:
    """Chroma's top k for an already-embedded query, built the way store.retrieve builds them."""
    result = collection.query(
        query_embeddings=[list(vector)],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        Retrieved(id=i, text=t, section_id=m["section_id"], section=m["section"], source=m["source"],
                  score=1.0 - d, distance=d)
        for i, t, m, d in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def check_vectors(collection, chunks, fresh) -> list:
    """Print the stored-versus-re-embedded check. Return Chroma's stored vectors in chunk order."""
    import numpy as np

    stored = stored_embeddings(collection)
    missing = [c.id for c in chunks if c.id not in stored]
    extra = sorted(set(stored) - {c.id for c in chunks})
    if missing or extra:
        raise StoreError(f"Chroma and the chunker disagree: missing from Chroma {missing}, only in Chroma {extra}")
    stored_in_order = np.asarray([stored[c.id] for c in chunks])
    show_stats("Chroma stored vectors", vector_stats(stored_in_order))
    show_stats("re-embedded vectors", vector_stats(np.asarray(fresh)))
    difference, row = normalized_max_abs_difference(stored_in_order, fresh)
    print(f"    max abs difference after normalizing both: {difference:.3e} (at {chunks[row].id})")
    return stored_in_order


def run_setting(label, with_context, chroma_db, embedder, question_sets, query_vectors, k, from_chroma):
    """Print the comparison for one setting. Return (number of questions that differ, collection, FAISS index)."""
    info = store_info(chroma_db)
    print(f"=== setting {label}: Chroma {chroma_db} ({info['count']} chunks, with_context={info['with_context']})")
    if info["with_context"] is not with_context:
        raise StoreError(f"the Chroma store for '{label}' must be built "
                         f"{'with --context' if with_context else 'with --no-context'}")
    collection = store_module._open_for_reading(chroma_db)
    store_module._check_embedder(collection, embedder)

    chunks = chunk_kb(REPO_ROOT / "docs" / "knowledge_base", MAX_CHARS, OVERLAP)
    fresh = embedder.embed([c.embed_text(with_context) for c in chunks])
    stored_in_order = check_vectors(collection, chunks, fresh)

    source = stored_in_order if from_chroma else fresh
    built = FaissChunkIndex.from_vectors(chunks, source, embedder, with_context=with_context)
    show_stats("FAISS vectors (normalized)", built.build_stats)
    with tempfile.TemporaryDirectory() as folder:
        built.save(Path(folder))
        index = FaissChunkIndex.load(Path(folder), embedder)
    print(f"    FAISS built from {'Chroma stored' if from_chroma else 're-embedded'} vectors, "
          f"saved, reloaded: {len(index)} vectors")

    differing = 0
    all_hits = {}
    for name, questions in question_sets.items():
        per_question = {}
        worst, worst_at, identical, ties = 0.0, None, 0, 0
        for q in questions:
            vector = query_vectors[(name, q.id)]
            chroma_hits = chroma_search_vector(collection, vector, k)
            faiss_hits = index.search_vector(vector, k)
            per_question[q.id] = (chroma_hits, faiss_hits)
            result = compare_results(chroma_hits, faiss_hits)
            if result.max_score_difference > worst:
                worst, worst_at = result.max_score_difference, q.id
            if result.identical:
                identical += 1
                continue
            differing += 1
            ties += bool(result.is_tie)
            print(f"    DIFFERENT {q.id}: first difference at rank {result.first_difference + 1}; {result.reason}")
            print(f"        Chroma: {describe(chroma_hits)}")
            print(f"        FAISS:  {describe(faiss_hits)}")
        all_hits[name] = per_question
        where = f" (at {worst_at})" if worst_at else ""
        flag = "" if worst <= SCORE_TOLERANCE else f"  <-- above {SCORE_TOLERANCE:.0e}"
        tie_note = f", {ties} of them ties" if len(questions) - identical else ""
        print(f"    {name:<12} {len(questions)} questions: ranked ids identical for {identical} of {len(questions)}"
              f"{tie_note}; largest score difference {worst:.2e}{where}{flag}")

    print("\n    hit@k and MRR, Chroma / FAISS (questions with an expected section only)")
    print(f"    {'set':<12}{'n':>3}" + "".join(f"{'hit@' + str(hk):>12}" for hk in HIT_KS) + f"{'MRR':>16}")
    for name, questions in question_sets.items():
        answerable, _ = split_answerable(questions)
        if not answerable:
            continue
        summaries = {
            side: summarize(
                [([h.section_id for h in all_hits[name][q.id][position]], q.expected) for q in answerable], HIT_KS
            )
            for side, position in (("chroma", 0), ("faiss", 1))
        }
        cells = "".join(
            f"{summaries['chroma']['hits'][hk]:>5} /{summaries['faiss']['hits'][hk]:>3}   " for hk in HIT_KS
        )
        print(f"    {name:<12}{summaries['chroma']['n']:>3}{cells}"
              f"{summaries['chroma']['mrr']:>8.3f} /{summaries['faiss']['mrr']:>6.3f}")
    print()
    return differing, collection, index


def time_searches(collection, index, vectors, k, repeats):
    """{row name: sorted seconds}, the search alone on pre-embedded queries."""
    import faiss
    import numpy as np

    raw = np.array(vectors, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(raw)
    n = collection.count()

    rows = (
        ("Chroma query", lambda i: collection.query(
            query_embeddings=[list(vectors[i])], n_results=min(k, n), include=["documents", "metadatas", "distances"])),
        ("FAISS search_vector", lambda i: index.search_vector(vectors[i], k)),
        ("FAISS bare search", lambda i: index._index.search(raw[i : i + 1], k)),
    )
    timings = {}
    for name, call in rows:
        for i in range(len(vectors)):  # warm-up: one pass over every query
            call(i)
        samples = []
        for r in range(repeats):
            i = r % len(vectors)
            start = time.perf_counter()
            call(i)
            samples.append(time.perf_counter() - start)
        timings[name] = sorted(samples)
    return timings


def print_timings(label, timings, n_queries, repeats, n_vectors) -> None:
    print(f"=== timing, setting {label}: search alone, {n_queries} pre-embedded queries cycled, "
          f"{repeats} timed searches per row, {n_vectors} vectors")
    print(f"    {'':<22}{'median':>12}{'min':>12}{'95th pct':>12}")
    for name, samples in timings.items():
        median = statistics.median(samples) * 1e6
        p95 = samples[int(0.95 * (len(samples) - 1))] * 1e6
        print(f"    {name:<22}{median:>9.1f} us{samples[0] * 1e6:>9.1f} us{p95:>9.1f} us")
    print("    The rows do different work. Chroma query: the search plus returning documents, metadata and")
    print("    distances. FAISS search_vector: copying and normalizing the query, the search, building hit objects.")
    print("    FAISS bare search: the search only, on a query normalized beforehand. Do not read a ratio between")
    print("    rows as 'FAISS is N times faster'. With this few vectors it says nothing about scale either.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx-db", type=Path, default=REPO_ROOT / "chroma_db" / "ctx")
    parser.add_argument("--noctx-db", type=Path, default=REPO_ROOT / "chroma_db" / "noctx")
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    parser.add_argument("--variants", type=Path, default=REPO_ROOT / "eval" / "retrieval_variants.json")
    parser.add_argument("--independent", type=Path, default=REPO_ROOT / "eval" / "retrieval_independent.json")
    parser.add_argument("--settings", nargs="+", choices=["ctx", "noctx"], default=["noctx", "ctx"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=500, help="timed searches per row")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--from-chroma-vectors", action="store_true",
                        help="build FAISS from Chroma's stored vectors instead of the re-embedded ones")
    args = parser.parse_args(argv)
    if args.k < max(HIT_KS):
        print(f"ERROR: --k must be at least {max(HIT_KS)}")
        return 1
    dbs = {"ctx": args.ctx_db, "noctx": args.noctx_db}

    try:
        test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
        question_sets = {
            "original": original_questions(test_set),
            "variants": load_variants(args.variants, test_set),
            "independent": load_independent(args.independent),
        }
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
        import faiss
    except ImportError:
        print("ERROR: faiss is not installed. Install the faiss-cpu package.")
        return 1

    counts = ", ".join(
        f"{name} {len(qs)} ({len(split_answerable(qs)[0])} with an expected section)" for name, qs in question_sets.items()
    )
    total = sum(len(qs) for qs in question_sets.values())
    print(f"faiss {faiss.__version__}; top {args.k} per question; tie tolerance {SCORE_TOLERANCE:.0e}")
    print(f"questions: {counts}; {total} in all\n")

    # Each question embedded once; the same vector goes to both backends, in every setting
    # (queries get no prefix, so the setting does not change them).
    query_vectors = {
        (name, q.id): embedder.embed([q.question])[0] for name, qs in question_sets.items() for q in qs
    }
    ordered_vectors = list(query_vectors.values())

    total_differing = 0
    try:
        for label in args.settings:
            differing, collection, index = run_setting(
                label, label == "ctx", dbs[label], embedder, question_sets, query_vectors, args.k,
                args.from_chroma_vectors,
            )
            total_differing += differing
            if not args.skip_timing:
                timings = time_searches(collection, index, ordered_vectors, args.k, args.repeats)
                print_timings(label, timings, len(ordered_vectors), args.repeats, len(index))
    except StoreError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"questions whose ranked ids differ between the backends: {total_differing}")
    return 1 if total_differing else 0


if __name__ == "__main__":
    sys.exit(main())