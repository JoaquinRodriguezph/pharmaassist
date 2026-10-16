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
- The same prompt produced two different answers. The difference is probably sampling (tested at temperature 0; see Sampling test).
- Response 1 is wrong: it says the reorder level is set by "the patient's insurance plan". A reorder level is an inventory threshold the pharmacy sets for itself.
- Gemini answered the same prompt correctly.
- A 3B model answering from memory will be unreliable on pharmacy facts. That is why grounding it with RAG (Steps 3 to 5) and measuring the results (Step 9) are the core of this project.

---

## Sampling test

Run through `OllamaClient` with `scripts/try_ollama.py` (model `llama3.2:latest`). Labels as above.

**Executions covered:** one execution of the current script, the output pasted most recently. Figures from earlier executions are not used here. "Run" means one `generate()` call.

### Experiments 1 and 2: temperature (prompt: *"In two sentences, what does a pharmacy reorder level mean?"*)
- **[Measured]** Temperature 0.0, two runs: identical text, 68 completion tokens and 37 prompt tokens each.
- **[Measured]** Temperature 1.5, two runs: different texts, 73 and 87 completion tokens.
- **[Observed]** The two temperature-1.5 texts differ in substance, not only wording. Run 1 says restocking triggers when the level "is reached or exceeded"; run 2 describes demand, not stock, falling below the level.
- **[Measured]** Latency: 5.48 s for the first call, then 0.73 to 0.89 s for the others in these two experiments.
- **[Observed]** The slow first call is probably warm-up. I did not isolate it (cold start was not measured).

### The persistent error
- **[Observed]** The earlier Response 1 said the reorder level is set by "the patient's insurance plan". Both temperature-0 runs here repeat it ("set by the pharmacy or the patient's insurance plan"). Neither temperature-1.5 run mentions insurance.
- **[Observed]** So at temperature 0, for this prompt, the wrong claim is the model's most likely answer, and temperature 0 makes it repeatable. Identical answers at temperature 0 show consistency, not correctness. This is another reason for RAG (Steps 3 to 5) and measurement (Step 9). Two runs is a small sample.

### Experiment 3: length instruction without a conflict (prompt: *"What does a pharmacy reorder level mean?"*, temperature 0.0)

| | No system prompt | System prompt ("You are a pharmacy operations assistant. Answer in one short sentence.") |
|---|---|---|
| Completion tokens [Measured] | 238 | 32 |
| Prompt tokens [Measured] | 33 | 46 |
| Latency [Measured] | 2.38 s | 0.34 s |
| Words [Measured, script count] | 190 | 29 |
| Approx. sentences [script count] | 7 | 1 |

- **[Observed]** The system-prompt answer is one sentence, so the 3B model obeyed the length instruction here. One prompt, one run per condition, temperature 0: this does not show how reliable that is.
- **[Observed]** The sentence count is approximate: the script's splitter counts list items, which inflates the no-system figure of 7.
- **[Observed]** The long answer includes example reorder levels (10-20, 50-100 and 200-500 units) presented as common, with no source. The short answer does not mention insurance.
- **[Derived]** Prompt tokens: 46 − 33 = 13 added by the system prompt, the same difference as in the earlier two-sentence version. It includes any chat-template tokens the system role adds, which cannot be separated here. 37 − 33 = 4 is the extra from "In two sentences, " in the other prompt.

### Prompt tokens
- **[Observed]** The 37 prompt tokens in experiments 1 and 2 match the count from Runs 1 and 2 earlier and do not drop on repeated calls, which is consistent with Ollama's count including cached prompt tokens.

### Failure tests
1. **Wrong host.** `$env:OLLAMA_HOST="http://localhost:11999"; py scripts/try_ollama.py`, then `Remove-Item Env:OLLAMA_HOST`.
   - **[Measured]** On the first call: `ERROR: Could not reach the Ollama server at http://localhost:11999. Is it running? Start it (or check OLLAMA_HOST) and try again.` No traceback. Exit code `1` (`$LASTEXITCODE`), from a re-run of this test.
   - **[Observed]** The `LLMConnectionError` message appeared as intended. The real environment variable overrode the value in `.env`.
   - **Wrong-host re-run [Measured]**, after the handler became `except LLMError`: the same one-line `ERROR: Could not reach the Ollama server at http://localhost:11999. Is it running? Start it (or check OLLAMA_HOST) and try again.`, no traceback, exit code `1`. **Confirmed:** both backend error types print as one line with exit code 1 through the shared `LLMError` handler.
2. **Wrong model.** `$env:OLLAMA_MODEL="no-such-model"; py scripts/try_ollama.py`, then `Remove-Item Env:OLLAMA_MODEL`.
   - **[Measured]** Raw traceback ending in `ollama._types.ResponseError: model 'no-such-model' not found (status code: 404)`.
   - **Decision:** `OllamaClient` should translate this. Callers should not need to catch an `ollama` package exception, the model name comes from config so this mistake is likely, and the message can name the fix. A 404 is now raised as `LLMModelNotFoundError` ("Ollama has no model '...'. Run `ollama pull ...` or check OLLAMA_MODEL."). Other `ResponseError`s pass through unchanged. Both error types now share an `LLMError` base class (confirmed below). The script is meant to catch it so that any backend error prints as one `ERROR:` line; this is confirmed by Re-run 3 below.
   - **Re-run 1 [Measured]:** the traceback ended in `app.llm.base.LLMModelNotFoundError: Ollama has no model 'no-such-model'. Run `ollama pull no-such-model` or check OLLAMA_MODEL.` (chained from the original `ResponseError ... (status code: 404)`), exit code `1`.
   - **[Observed]** The translation in `OllamaClient` works. The error was not caught by the script, so a traceback printed instead of the one-line message. The script's handler still caught only `LLMConnectionError` (inferred from the behavior; the line numbers match the updated script, so only the import and `except` lines were stale).
   - **Fix:** in `scripts/try_ollama.py`, import `LLMError` and use `except LLMError as e:`. Confirmed by Re-run 3.
   - **Re-run 2 [Measured]:** the output was the same traceback as Re-run 1 (same line numbers, ending in `LLMModelNotFoundError`), exit code `1`. **[Observed]** The script still did not catch the error, so the `except LLMError` fix is not in effect in the file that ran (possible causes: edit not saved, or the import/`except` lines not both changed).
   - **File check [Measured]:** `Select-String -Path scripts\try_ollama.py -Pattern "LLMError|LLMConnectionError"` found only `LLMError`: line 20 `from app.llm import LLMError, LLMResponse` and line 97 `except LLMError as e:`. **[Observed]** The edit is now in the file, so Re-run 2 most likely ran before it was saved (inferred, not shown). 
   - **Re-run 3 [Measured]:** after the edit, the same test printed no traceback, only `ERROR: Ollama has no model 'no-such-model'. Run `ollama pull no-such-model` or check OLLAMA_MODEL.`, and exit code `1`. **Confirmed:** a wrong model name now fails with a one-line error and exit code 1, the same as a wrong host.
3. **Class hierarchy [Measured]:** `issubclass(LLMConnectionError, LLMError)` and `issubclass(LLMModelNotFoundError, LLMError)` both print `True`.

### `.env.example` (Ollama lines only)
```
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_HOST=http://localhost:11434
```

The script itself is in the repo (`scripts/try_ollama.py`) and is not copied here.