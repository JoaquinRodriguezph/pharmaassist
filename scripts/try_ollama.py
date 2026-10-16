"""Manual check of OllamaClient: determinism, sampling, and system prompt.

Run from the project root:  py scripts/try_ollama.py
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

from app.llm import LLMConnectionError, LLMResponse
from app.llm.ollama_client import OllamaClient

PROMPT = "In two sentences, what does a pharmacy reorder level mean?"
HIGH_TEMPERATURE = 1.5
SYSTEM_PROMPT = (
    "You are a pharmacy operations assistant. Answer in one short sentence."
)


def ask(client: OllamaClient, temperature: float, system: str | None = None) -> LLMResponse:
    return client.generate(
        [{"role": "user", "content": PROMPT}],
        system=system,
        temperature=temperature,
    )


def show(label: str, r: LLMResponse) -> None:
    print(f"--- {label}")
    print(r.text.strip())
    print(
        f"[model={r.model} prompt_tokens={r.prompt_tokens} "
        f"completion_tokens={r.completion_tokens} latency={r.latency_seconds:.2f}s]\n"
    )


def compare(a: LLMResponse, b: LLMResponse) -> None:
    same = a.text.strip() == b.text.strip()
    print(f"==> Identical: {'YES' if same else 'NO'}\n")


def main() -> None:
    client = OllamaClient()

    print(f"PROMPT: {PROMPT}\n")

    print("=== 1. temperature=0.0, same prompt twice ===")
    a, b = ask(client, 0.0), ask(client, 0.0)
    show("run 1", a)
    show("run 2", b)
    compare(a, b)

    print(f"=== 2. temperature={HIGH_TEMPERATURE}, same prompt twice ===")
    a, b = ask(client, HIGH_TEMPERATURE), ask(client, HIGH_TEMPERATURE)
    show("run 1", a)
    show("run 2", b)
    compare(a, b)

    print("=== 3. system prompt (temperature=0.0) ===")
    show("no system prompt", ask(client, 0.0))
    show(f'system: "{SYSTEM_PROMPT}"', ask(client, 0.0, system=SYSTEM_PROMPT))


if __name__ == "__main__":
    try:
        main()
    except LLMConnectionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)