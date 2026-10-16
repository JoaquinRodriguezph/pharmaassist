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