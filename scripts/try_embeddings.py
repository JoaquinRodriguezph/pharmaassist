"""Embed the knowledge base chunks and a few questions; look at the nearest chunks.

Run from the project root (the first run downloads the model from Hugging Face):

    py scripts/try_embeddings.py

Chunks the KB at max_chars=800, overlap=0, embeds every chunk twice (with and
without the document/section prefix from Chunk.embed_text), embeds the exact
text of Q01, Q06, Q08, Q09 and Q22, and prints the top 3 chunks and cosine
scores for each question and each version. It also counts tokens per chunk
with the model's own tokenizer and compares them with the model's limit.
"""

# PREDICTIONS, written before the first run: the section I expect at rank 1.
#
#   Q01  return unopened amoxicillin, has receipt  -> returns_and_expiry_policy.md#Customer-Returns
#   Q06  max paracetamol 500 mg tabs in 24 hours   -> paracetamol_leaflet.md#Dosage-And-Administration
#        (my guess for rank 2: paracetamol_leaflet.md#About-This-Medicine, which also says "500 mg"
#        and "adult"; the question's typo "paracetmol" may weaken the match for this model)
#   Q08  who should NOT take ibuprofen             -> ibuprofen_leaflet.md#Contraindications
#   Q09  employee discount on OTC vitamins         -> no right answer exists (the KB is silent). I expect a
#        false friend at rank 1: returns_and_expiry_policy.md#Customer-Returns, which mentions OTC items
#        and vitamins, with a middling score that is clearly below the answerable questions' rank-1 scores.
#   Q22  write a python scraper for a competitor   -> no right answer exists. No confident guess for rank 1;
#        I expect the lowest scores of the five.
#
# Token counts: I expect no chunk to exceed the 256-token limit at 800/0, with the longest chunk (756
# characters) at roughly 150 to 220 tokens, and the prefix adding roughly 15 to 25 tokens to each chunk.
#
# Context prefix: I expect it to raise the score of the right leaflet for Q06 and Q08 (the title names the
# drug) and to change little for Q01. I have no prediction about whether it helps overall.

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.chunking import chunk_kb
from app.rag.similarity import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_IDS = ["Q01", "Q06", "Q08", "Q09", "Q22"]
MAX_CHARS = 800
OVERLAP = 0
TOP_K = 3


def label(chunk) -> str:
    part = f" (part {chunk.part}/{chunk.n_parts})" if chunk.n_parts > 1 else ""
    return f"{chunk.section_id}{part}"


def load_questions(test_set_path: Path, ids: list[str]) -> list[dict]:
    by_id = {q["id"]: q for q in json.loads(test_set_path.read_text(encoding="utf-8"))}
    return [by_id[i] for i in ids]


def rank(query_vector, chunk_vectors, chunks, k=None):
    """All chunks as (score, chunk), best first; ties keep chunk order. `k` limits the list."""
    scored = [(cosine_similarity(query_vector, v), c) for v, c in zip(chunk_vectors, chunks)]
    scored.sort(key=lambda pair: -pair[0])
    return scored if k is None else scored[:k]


def token_counts(tokenizer, texts: list[str]) -> list[int]:
    """Tokens per text as the model sees them: with the special [CLS]/[SEP] tokens, no truncation."""
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]


def report_tokens(model, chunks) -> None:
    limit = model.max_seq_length
    without = token_counts(model.tokenizer, [c.embed_text(False) for c in chunks])
    with_prefix = token_counts(model.tokenizer, [c.embed_text(True) for c in chunks])

    print("=== Token lengths (the model's own tokenizer, special tokens included) ===")
    print(f"Model limit (model.max_seq_length): {limit}; tokenizer.model_max_length: {model.tokenizer.model_max_length}")
    print(f"{'chunk':<66}{'chars':>6}{'tokens':>8}{'with prefix':>13}")
    for chunk, n, m in zip(chunks, without, with_prefix):
        print(f"{label(chunk):<66}{len(chunk.text):>6}{n:>8}{m:>13}")
    print(f"max tokens: {max(without)} without prefix, {max(with_prefix)} with prefix (limit {limit})")
    print(f"chunks over the limit: {sum(n > limit for n in without)} without prefix, "
          f"{sum(m > limit for m in with_prefix)} with prefix, of {len(chunks)}")
    print("Text beyond the limit is cut off (the first max_seq_length tokens are kept) before embedding.\n")


def report_queries(model, chunks, questions) -> list[dict]:
    vectors = {
        with_context: model.encode(
            [c.embed_text(with_context) for c in chunks], show_progress_bar=False
        ).tolist()
        for with_context in (False, True)
    }
    query_vectors = model.encode([q["question"] for q in questions], show_progress_bar=False).tolist()

    summary = []
    for question, query_vector in zip(questions, query_vectors):
        expected = set(question.get("expected_sources") or [])
        print(f"=== {question['id']} ({question['category']}): {question['question']}")
        print(f"    expected section: {', '.join(sorted(expected)) if expected else '(none: not answerable from the KB)'}")
        row = {"id": question["id"], "category": question["category"]}
        for with_context in (False, True):
            name = "with context prefix" if with_context else "without prefix"
            ranked = rank(query_vector, vectors[with_context], chunks)
            print(f"  {name}:")
            for position, (score, chunk) in enumerate(ranked[:TOP_K], start=1):
                mark = "  <- expected" if chunk.section_id in expected else ""
                print(f"    {position}  {score:6.3f}  {label(chunk)}{mark}")
            row[name] = {
                "top1": ranked[0][0],
                "expected_rank": next(
                    (i for i, (_, c) in enumerate(ranked, start=1) if c.section_id in expected), None
                ),
            }
        summary.append(row)
        print()
    return summary


def report_summary(summary) -> None:
    print("=== Summary: top-1 score, and the rank of the first chunk of the expected section ===")
    print(f"{'question':<9}{'category':<26}{'top-1 without':>14}{'top-1 with':>12}{'expected rank (without / with)':>34}")
    for row in summary:
        a, b = row["without prefix"], row["with context prefix"]
        ranks = f"{a['expected_rank'] or '-'} / {b['expected_rank'] or '-'}"
        print(f"{row['id']:<9}{row['category']:<26}{a['top1']:>14.3f}{b['top1']:>12.3f}{ranks:>34}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "docs" / "knowledge_base")
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    args = parser.parse_args(argv)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is not installed. See docs/retrieval_notes.md for the install steps.")
        return 1

    chunks = chunk_kb(args.kb_dir, MAX_CHARS, OVERLAP)
    questions = load_questions(args.test_set, QUERY_IDS)

    print(f"Loading {args.model} (the first run downloads it from Hugging Face)...")
    model = SentenceTransformer(args.model, device="cpu")
    print(f"Model: {args.model}")
    dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"Embedding dimension: {dimension()}")
    print(f"Chunks: {len(chunks)} at max_chars={MAX_CHARS}, overlap={OVERLAP}\n")

    report_tokens(model, chunks)
    summary = report_queries(model, chunks, questions)
    report_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())