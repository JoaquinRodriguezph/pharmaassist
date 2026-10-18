"""Chunk the knowledge base and check the chunks against eval/test_set.json.

Run from the project root:

    py scripts/chunk_stats.py [--max-chars 800] [--overlap 0]

Prints the number of chunks and their sizes in characters (min, median, max).
Then, for each question's cited section and its `must_contain` strings, checks:

  (a) every string is found in at least one chunk of that section
      (exit status 1 if not: chunking has lost a fact);
  (b) all the strings sit in the same chunk (reported only).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_kb, stray_preamble_lines
from app.rag.kb import (
    is_kb_document,
    kb_documents,
    normalize_for_match,
    question_requirements,
    split_sections,
)


def analyze(kb_dir: Path, test_set: list[dict], max_chars: int, overlap: int) -> dict:
    """Chunk the KB and run checks (a) and (b). Returns everything main() prints."""
    chunks = chunk_kb(kb_dir, max_chars, overlap)
    sizes = [len(c.text) for c in chunks]
    section_sizes = [
        len(section.body.strip())
        for path in kb_documents(kb_dir)
        for section in split_sections(path.read_text(encoding="utf-8"))
        if section.body.strip()
    ]

    pairs = []  # one entry per (question, cited section) that has must_contain strings
    for question in test_set:
        for source, strings in question_requirements(question).items():
            section_chunks = [c for c in chunks if c.section_id == source]
            found_in = {
                s: [c.part for c in section_chunks if normalize_for_match(s) in normalize_for_match(c.text)]
                for s in strings
            }
            missing = [s for s, parts in found_in.items() if not parts]
            same_chunk = not missing and any(
                all(normalize_for_match(s) in normalize_for_match(c.text) for s in strings)
                for c in section_chunks
            )
            pairs.append(
                {
                    "id": question.get("id", "?"),
                    "source": source,
                    "n_parts": section_chunks[0].n_parts if section_chunks else 0,
                    "found_in": found_in,
                    "missing": missing,
                    "same_chunk": same_chunk,
                }
            )

    return {
        "chunks": len(chunks),
        "sections": len({c.section_id for c in chunks}),
        "split_sections": len({c.section_id for c in chunks if c.n_parts > 1}),
        "min": min(sizes) if sizes else 0,
        "median": statistics.median(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "section_min": min(section_sizes) if section_sizes else 0,
        "section_median": statistics.median(section_sizes) if section_sizes else 0,
        "section_max": max(section_sizes) if section_sizes else 0,
        "pairs": pairs,
        "facts_found": all(not p["missing"] for p in pairs),
        "same_chunk_pairs": sum(1 for p in pairs if p["same_chunk"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "docs" / "knowledge_base")
    args = parser.parse_args(argv)

    test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    try:
        result = analyze(args.kb_dir, test_set, args.max_chars, args.overlap)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Chunking: max_chars={args.max_chars}, overlap={args.overlap}")
    skipped = sorted(p.name for p in args.kb_dir.iterdir() if p.is_file() and not is_kb_document(p))
    print(f"Not indexed (not documents): {', '.join(skipped) if skipped else 'none'}")
    for path in kb_documents(args.kb_dir):
        stray = stray_preamble_lines(path.read_text(encoding="utf-8"))
        if stray:
            print(f"WARNING: {path.name}: {len(stray)} line(s) before the first '##' heading are not indexed")
    print(f"Chunks: {result['chunks']} from {result['sections']} sections "
          f"({result['split_sections']} section(s) split into more than one chunk)")
    print(f"Chunk size in characters: min={result['min']}, median={result['median']:g}, max={result['max']}")
    print(f"Section text size in characters (before splitting): min={result['section_min']}, "
          f"median={result['section_median']:g}, max={result['section_max']}")

    print("\nmust_contain, per question and cited section:")
    for pair in result["pairs"]:
        label = f"  {pair['id']} {pair['source']}:"
        if pair["missing"]:
            print(f"{label} MISSING {pair['missing']} in every chunk of the section")
        elif pair["same_chunk"]:
            print(f"{label} all {len(pair['found_in'])} found, in one chunk")
        else:
            spread = "; ".join(f"{s!r} in part {','.join(map(str, parts))}" for s, parts in pair["found_in"].items())
            print(f"{label} all found, but NOT in one chunk ({spread})")

    total = len(result["pairs"])
    print(f"\n(a) every string found in a chunk of its section: {'PASS' if result['facts_found'] else 'FAIL'}")
    print(f"(b) all of a question's strings in one chunk: {result['same_chunk_pairs']} of {total} (report only)")
    return 0 if result["facts_found"] else 1


if __name__ == "__main__":
    sys.exit(main())