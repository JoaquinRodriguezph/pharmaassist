"""Check that every section cited in eval/test_set.json exists in the knowledge base.

Run from the project root:  py scripts/check_kb.py
Exit status 1 if any cited file or section is missing.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.rag.kb import check_sources, expected_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set.json")
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "docs" / "knowledge_base")
    args = parser.parse_args(argv)

    test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    sources = expected_sources(test_set)
    problems = check_sources(args.kb_dir, sources)

    for problem in problems:
        print(f"MISSING: {problem}")
    print(f"Checked {len(sources)} cited section(s): {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())