"""Embed the knowledge base into the Chroma store. Safe to run again.

Run from the project root:

    py scripts/ingest_kb.py [--db chroma_db] [--max-chars 800] [--overlap 0] [--no-context] [--rebuild]

Unchanged chunks are left alone, edited ones are embedded again, removed ones
are deleted. `--no-context` embeds chunk text without the document/section
prefix; changing it re-embeds every chunk. `--rebuild` deletes the collection
first (needed to switch to a different embedding model).
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.store import StoreError, ingest_kb, store_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "docs" / "knowledge_base")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "chroma_db")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--no-context", action="store_true", help="embed the text without the document/section prefix")
    parser.add_argument("--rebuild", action="store_true", help="delete the collection first")
    args = parser.parse_args(argv)

    try:
        from app.rag.embedding import SentenceTransformerEmbedder

        print("Loading the embedding model...")
        embedder = SentenceTransformerEmbedder()
    except ImportError:
        print("ERROR: sentence-transformers is not installed. See docs/retrieval_notes.md.")
        return 1

    start = time.perf_counter()
    try:
        counts = ingest_kb(
            args.kb_dir,
            args.db,
            embedder,
            max_chars=args.max_chars,
            overlap=args.overlap,
            with_context=not args.no_context,
            rebuild=args.rebuild,
        )
    except StoreError as error:
        print(f"ERROR: {error}")
        return 1
    seconds = time.perf_counter() - start

    print(f"added={counts.added} updated={counts.updated} unchanged={counts.unchanged} deleted={counts.deleted}")
    print(f"ingest took {seconds:.2f}s (max_chars={args.max_chars}, overlap={args.overlap}, "
          f"with_context={not args.no_context})")
    info = store_info(args.db)
    print(f"store: {args.db} | {info['count']} chunks | embedder {info['embedder_name']} "
          f"(dimension {info['embedder_dimension']}) | with_context={info['with_context']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())