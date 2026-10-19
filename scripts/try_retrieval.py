"""Ask Q01, Q06 and Q08 of the Chroma store and compare with the Step 4a table.

Run from the project root, after scripts/ingest_kb.py:

    py scripts/try_retrieval.py [--db chroma_db]

The top-3 scores from the store should match the scores from
scripts/try_embeddings.py (the Step 4a table) to about three decimals. This
checks the whole path: chunking, the prefix setting, Chroma's cosine distance
and `score = 1 - distance`. Exit status 1 if any chunk or score differs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.store import StoreError, retrieve, store_info

CR = "returns_and_expiry_policy.md#Customer-Returns"
RETURNS_PURPOSE = "returns_and_expiry_policy.md#Purpose-And-Scope"
RETURNS_EXPIRED = "returns_and_expiry_policy.md#Expired-Medications"
PARA = "paracetamol_leaflet.md"
IBU = "ibuprofen_leaflet.md"

# The Step 4a table (scripts/try_embeddings.py, measured on this machine): (chunk id, score).
EXPECTED = {
    True: {  # with the context prefix
        "Q01": [(f"{CR}::1", 0.619), (f"{CR}::2", 0.423), (f"{RETURNS_PURPOSE}::1", 0.386)],
        "Q06": [(f"{PARA}#Dosage-And-Administration::1", 0.689), (f"{PARA}#About-This-Medicine::1", 0.622), (f"{PARA}#Warnings-And-Precautions::1", 0.479)],
        "Q08": [(f"{IBU}#About-This-Medicine::1", 0.762), (f"{IBU}#Drug-Interactions::1", 0.693), (f"{IBU}#Contraindications::1", 0.671)],
    },
    False: {  # without it
        "Q01": [(f"{CR}::1", 0.640), (f"{CR}::2", 0.452), (f"{RETURNS_EXPIRED}::1", 0.364)],
        "Q06": [(f"{PARA}#Dosage-And-Administration::1", 0.705), (f"{PARA}#About-This-Medicine::1", 0.566), (f"{PARA}#Warnings-And-Precautions::1", 0.369)],
        "Q08": [(f"{IBU}#About-This-Medicine::1", 0.709), (f"{IBU}#Drug-Interactions::1", 0.632), (f"{IBU}#Contraindications::1", 0.603)],
    },
}
TOLERANCE = 0.001


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "chroma_db")
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    args = parser.parse_args(argv)

    try:
        from app.rag.embedding import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except ImportError:
        print("ERROR: sentence-transformers is not installed. See docs/retrieval_notes.md.")
        return 1

    questions = {q["id"]: q["question"] for q in json.loads(args.test_set.read_text(encoding="utf-8"))}
    try:
        info = store_info(args.db)
        with_context = bool(info["with_context"])
        print(f"store: {info['count']} chunks | embedder {info['embedder_name']} | with_context={with_context}\n")

        agree = total = 0
        first_pass = []
        for qid, expected in EXPECTED[with_context].items():
            start = time.perf_counter()
            results = retrieve(args.db, embedder, questions[qid], k=3)
            first_pass.append(time.perf_counter() - start)
            print(f"=== {qid}: {questions[qid]}")
            for rank, (result, (want_id, want_score)) in enumerate(zip(results, expected), start=1):
                same = result.id == want_id and abs(round(result.score, 3) - want_score) <= TOLERANCE
                agree += same
                total += 1
                print(f"  {rank}  {result.score:6.3f} (distance {result.distance:.3f})  {result.id}"
                      f"   | step 4a: {want_score:.3f} {'OK' if same else 'DIFFERENT: ' + want_id}")
            print()

        second_pass = []
        for qid in EXPECTED[with_context]:
            start = time.perf_counter()
            retrieve(args.db, embedder, questions[qid], k=3)
            second_pass.append(time.perf_counter() - start)
    except StoreError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"agree with the Step 4a table (same chunk, score within {TOLERANCE}): {agree} of {total}")
    print("retrieve() time per question, including embedding the query and opening the store:")
    print(f"  first pass:  {', '.join(f'{t * 1000:.0f} ms' for t in first_pass)}")
    print(f"  second pass: {', '.join(f'{t * 1000:.0f} ms' for t in second_pass)}")
    return 0 if agree == total else 1


if __name__ == "__main__":
    sys.exit(main())