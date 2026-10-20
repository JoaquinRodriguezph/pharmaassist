"""Write requirements-runtime.txt: the service's direct runtime packages, pinned to the versions in this venv.

    py scripts/write_runtime_requirements.py

Test-only and analysis packages (pytest, faiss-cpu) are left out. torch is left out on purpose:
the Dockerfile installs the CPU build from PyTorch's own index, because the default Linux wheel
pulls in CUDA libraries the container does not use. Transitive dependencies are not pinned.
"""

import importlib.metadata as metadata
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = [
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "chromadb",
    "sentence-transformers",
    "numpy",
    "ollama",
    "google-genai",
    "python-dotenv",
]


def main() -> int:
    lines, missing = [], []
    for name in PACKAGES:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        print(f"ERROR: not installed in this venv: {missing}")
        return 1
    out = REPO_ROOT / "requirements-runtime.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))
    try:
        print(f"(torch in this venv: {metadata.version('torch')}; the image installs the CPU build instead)")
    except metadata.PackageNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())