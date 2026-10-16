# Local Model Notes

Labels: **Measured** = read directly from a tool's output. **Derived** = calculated from measured values. **Observed** = something I saw happen, not a tool figure.

## Hardware Configuration
- **GPU (Reported by `nvidia-smi`):** NVIDIA GeForce RTX 5050 ... (8151 MiB Total VRAM) *(`nvidia-smi` truncates the name; the full name was not seen)*
- **Driver Version:** 592.15
- **CUDA Version:** 13.1
- **Active Compute Process:** PID 1032 (`...a\lib\ollama\llama-server.exe`)

---

## Model Specifications (`ollama show llama3.2`, `ollama list`)
- **Model Tag:** `llama3.2` (resolves to `llama3.2:latest`) [Measured: `ollama list`]
- **Model ID:** `a80c4f17acd5` [Measured: `ollama list`]
- **Size on disk:** 2.0 GB [Measured: `ollama list`]
- **Architecture:** `llama`
- **Parameter Count:** `3.2B`
- **Quantization:** `Q4_K_M`
- **Embedding Length:** `3072`

### Context length
- **Model-supported maximum:** `131,072` tokens [Measured: `ollama show`]
- **Running context:** `4096` tokens [Measured: `ollama ps`]
- The 4096 figure is the runtime default in this setup: it is what Ollama actually loaded the model with. The 131,072 figure is only the architectural limit; using it would need a larger context setting and more VRAM for the KV cache.

---

## Memory & VRAM Measurements
- **Total Dedicated VRAM:** `8,151 MiB` (~8 GB) [Measured: `nvidia-smi`]
- **VRAM With Model Loaded:** `3,829 MiB` [Measured: `nvidia-smi`]
  - Includes `llama-server.exe` (PID 1032), the display desktop environment, and open applications (VS Code, Chrome, Edge, DuckDuckGo).
- **Baseline VRAM (Without Model):**
  - `1,348 MiB`: first `nvidia-smi`, before the model loaded [Measured]
  - `1,082 MiB`: `nvidia-smi` on 2026-10-16 at 15:34, model unloaded (`ollama ps` empty) [Measured]
- **Model size reported by Ollama:** `2.6 GB` [Measured: `ollama ps`]
- **Model VRAM footprint:** `3,829 − 1,348 = 2,481 MiB` [Derived], consistent with the 2.6 GB from `ollama ps`. Against the later baseline it would be `3,829 − 1,082 = 2,747 MiB` [Derived].
  - Open apps differed somewhat between the readings, so treat these as approximate. The two baselines themselves differ by ~266 MiB with no model loaded, which is a rough indication of how much non-model usage varies.

---

## Performance & Speed Benchmarks (`--verbose`)

### Run 1:
- **Total Duration:** `804.6034 ms` (0.80 s) [Measured]
- **Load Duration:** `5.3194 ms` [Measured]
- **Prompt Evaluation Count:** `37` tokens (Cached: `20` tokens) [Measured]
- **Prompt Evaluation Duration:** `87.65 ms` [Measured]
- **Prompt Evaluation Speed:** `193.95 tokens/s` [Derived: 17 non-cached tokens ÷ 0.08765 s]
- **Generation (Eval) Count:** `64` tokens [Measured]
- **Generation (Eval) Duration:** `688.494 ms` [Measured]
- **Generation Speed:** `92.96 tokens/s` [Measured]

### Run 2:
- **Total Duration:** `1.1135813 s` (1,113.58 ms) [Measured]
- **Load Duration:** `3.7885 ms` [Measured]
- **Prompt Evaluation Count:** `37` tokens (Cached: `36` tokens) [Measured]
- **Prompt Evaluation Duration:** `200.616 ms` [Measured]
- **Prompt Evaluation Speed:** `4.98 tokens/s` **not meaningful (prompt cache)**. 36 of 37 prompt tokens came from cache, so only 1 token was actually evaluated: 1 ÷ 0.2006 s = 4.98.
- **Generation (Eval) Count:** `81` tokens [Measured]
- **Generation (Eval) Duration:** `891.815 ms` [Measured]
- **Generation Speed:** `90.83 tokens/s` [Measured]

### Reading the numbers
- Generation speed (~91–93 tokens/s) is the comparable figure across the two runs.
- Prompt eval speed is not comparable: Run 1 evaluated 17 tokens, Run 2 only 1. A single throughput number can mislead when the token count behind it differs.

### Cold start
- Not measured. Load durations of 5.3 ms and 3.8 ms show the model was already in memory for both runs, so these are warm-start figures.

---

## Model Outputs

### Prompt:
> *"In two sentences, what does a pharmacy reorder level mean?"*

### Response 1:
> "A pharmacy reorder level refers to the quantity of a medication that a pharmacy will automatically reorder when a certain amount is sold or used. This level is set by the pharmacy or the patient's insurance plan to ensure that the medication is always available and to prevent stockouts, which can lead to inconvenience and decreased patient adherence."

### Response 2:
> "A pharmacy reorder level is a predetermined quantity of a medication that a pharmacy is willing to hold in stock before reordering from the manufacturer. This level can vary depending on the pharmacy and the medication, but it is typically a specific quantity (e.g. 10, 20, or 50 units) that the pharmacy has agreed to maintain in stock to ensure that patients can receive their prescribed medication."

### Observed
- The same prompt produced two different answers. The difference comes from sampling.
- Response 1 is wrong: it says the reorder level is set by "the patient's insurance plan". A reorder level is an inventory threshold the pharmacy sets for itself.
- Gemini answered the same prompt correctly.
- A 3B model answering from memory will be unreliable on pharmacy facts. That is why grounding it with RAG (Steps 3 to 5) and measuring the results (Step 9) are the core of this project.

---

## Sampling test

Run through `OllamaClient` with `scripts/try_ollama.py` (model `llama3.2:latest`). Parts 1 to 3 ran in two separate successful executions. Labels as above.

Prompt: *"In two sentences, what does a pharmacy reorder level mean?"*

### Temperature 0.0
- **[Measured]** Two runs per execution gave identical text (68 completion tokens, 37 prompt tokens each). Counting the "no system prompt" run in part 3 (same settings), all 6 temperature-0 runs across both executions gave the same text.
- **[Measured]** Latency of the first call was 5.51 s in one execution and 2.76 s in the other; every later call took 0.51 to 0.73 s.
- **[Observed]** The slow first call is probably warm-up. I did not isolate it (cold start was not measured).

### Temperature 1.5
- **[Measured]** 4 runs (2 per execution) gave 4 different texts, with 62, 61, 62 and 75 completion tokens.
- **[Observed]** The texts differ in substance, not only wording. One says the level is the quantity a pharmacy "must sell" before reordering; another says reordering happens when the pharmacy "sells out".

### The persistent error
- **[Observed]** The earlier Response 1 said the reorder level is set by "the patient's insurance plan". All 6 temperature-0 runs repeat that claim, and both system-prompt runs repeat it ("insurance provider"). None of the 4 temperature-1.5 runs mention insurance.
- **[Observed]** So the error is not a sampling accident. It is what the model most likely says, and temperature 0 makes it repeatable. Identical answers at temperature 0 show consistency, not correctness. This is another reason for RAG (Steps 3 to 5) and measurement (Step 9).

### System prompt (part 3, temperature 0.0)
- **[Measured]** No system prompt: 68 completion tokens, 2 sentences. With "You are a pharmacy operations assistant. Answer in one short sentence.": 52 completion tokens, still 2 sentences. Both results were the same in both executions.
- **[Observed]** This is not a clean test of the length instruction, because the user prompt says "in two sentences" and the system prompt says "one short sentence". The model produced two sentences, so the user prompt won, though the answer got shorter.
- Part 4 of the script removes the conflict ("What does a pharmacy reorder level mean?", temperature 0, with and without the system prompt). **Not yet run.**

### Prompt tokens
- **[Measured]** `prompt_tokens`: 37 without a system prompt, 50 with one.
- **[Derived]** 50 − 37 = 13 tokens added by the system prompt. This includes any chat-template tokens the system role adds; the two cannot be separated from these numbers.
- **[Observed]** 37 matches the prompt count from Runs 1 and 2 earlier and does not drop on repeated calls, which is consistent with Ollama's count including cached prompt tokens.

### Failure tests
1. **Wrong host.** `$env:OLLAMA_HOST="http://localhost:11999"; py scripts/try_ollama.py`, then `Remove-Item Env:OLLAMA_HOST`.
   - **[Measured]** On the first call: `ERROR: Could not reach the Ollama server at http://localhost:11999. Is it running? Start it (or check OLLAMA_HOST) and try again.` No traceback; exit code 1.
   - **[Observed]** The `LLMConnectionError` message appeared as intended. The real environment variable overrode the value in `.env`.
2. **Wrong model.** `$env:OLLAMA_MODEL="no-such-model"; py scripts/try_ollama.py`, then `Remove-Item Env:OLLAMA_MODEL`.
   - **[Measured]** Raw traceback ending in `ollama._types.ResponseError: model 'no-such-model' not found (status code: 404)`.
   - **Decision:** `OllamaClient` should translate this. Callers should not need to catch an `ollama` package exception, the model name comes from config so this mistake is likely, and the message can name the fix. A 404 is now raised as `LLMModelNotFoundError` ("Ollama has no model '...'. Run `ollama pull ...` or check OLLAMA_MODEL."). Other `ResponseError`s pass through unchanged. **Implemented after this test and not yet re-run against the real server.**

### `.env.example` (Ollama lines only)
```
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_HOST=http://localhost:11434
```

### `scripts/try_ollama.py`
Parts 1 to 3 gave the results above; part 4 was added afterwards.

```python
"""Manual check of OllamaClient: determinism, sampling, and system prompt.

Run from the project root:  py scripts/try_ollama.py
"""

import re
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
PROMPT_NO_LENGTH = "What does a pharmacy reorder level mean?"
HIGH_TEMPERATURE = 1.5
SYSTEM_PROMPT = (
    "You are a pharmacy operations assistant. Answer in one short sentence."
)


def ask(
    client: OllamaClient,
    temperature: float,
    system: str | None = None,
    prompt: str = PROMPT,
) -> LLMResponse:
    return client.generate(
        [{"role": "user", "content": prompt}],
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


def length_stats(text: str) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return f"    words={len(text.split())} approx_sentences={len(sentences)}\n"


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

    print("=== 4. length instruction, no conflicting prompt (temperature=0.0) ===")
    print(f"PROMPT: {PROMPT_NO_LENGTH}\n")
    for label, system in [("no system prompt", None), (f'system: "{SYSTEM_PROMPT}"', SYSTEM_PROMPT)]:
        r = ask(client, 0.0, system=system, prompt=PROMPT_NO_LENGTH)
        show(label, r)
        print(length_stats(r.text))


if __name__ == "__main__":
    try:
        main()
    except LLMConnectionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
```