"""Send one fixed prompt through whichever backend LLM_BACKEND selects.

Run from the project root:  py scripts/try_backend.py
One call, no retries of your own: a failed Gemini call still counts against quota.
"""

import sys
from pathlib import Path

# Make `app` importable when running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # fall back to variables already set in the environment

import os

from app.llm.base import LLMError
from app.llm.factory import create_llm_client

PROMPT = "What does a pharmacy reorder level mean?"
SYSTEM_PROMPT = "You are a pharmacy operations assistant. Answer in one short sentence."


def main() -> None:
    client = create_llm_client()
    result = client.generate(
        [{"role": "user", "content": PROMPT}],
        system=SYSTEM_PROMPT,
        temperature=0.0,
    )

    print(f"Backend:           {os.environ.get('LLM_BACKEND')}")
    print(f"Model:             {result.model}")
    print(f"Prompt:            {PROMPT}")
    print(f"System prompt:     {SYSTEM_PROMPT}")
    print("--- Text ---")
    print(result.text.strip())
    print("--- Numbers ---")
    print(f"Prompt tokens:     {result.prompt_tokens}")
    print(f"Completion tokens: {result.completion_tokens}")
    print(f"Reasoning tokens:  {result.reasoning_tokens}")
    latency = result.latency_seconds
    print(f"Latency:           {latency:.2f}s" if latency is not None else "Latency:           None")


if __name__ == "__main__":
    try:
        main()
    except (LLMError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)