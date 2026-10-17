"""Check the knowledge base against eval/test_set.json.

For every `expected_sources` entry: the file must exist, it must have the
section heading, and the section's text must contain each of the question's
`must_contain` strings (case-insensitive). Run from the project root:

    py scripts/check_kb.py

Prints each problem and exits with status 1 if there are any.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.kb import check_sources, expected_sources, required_strings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "docs" / "knowledge_base")
    args = parser.parse_args(argv)

    test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    sources = expected_sources(test_set)
    try:
        required = required_strings(test_set)
    except ValueError as error:
        print(f"ERROR in {args.test_set.name}: {error}")
        return 1
    problems = check_sources(args.kb_dir, sources, required)

    for problem in problems:
        print(f"MISSING: {problem}")
    facts = sum(len(strings) for strings in required.values())
    print(f"Checked {len(sources)} cited section(s) and {facts} required string(s): {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())