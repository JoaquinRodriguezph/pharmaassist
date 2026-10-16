# Local Model Notes

## Hardware Configuration
- **GPU (Reported by `nvidia-smi`):** NVIDIA GeForce RTX 5050 Laptop GPU (8151 MiB Total VRAM)
- **Driver Version:** 592.15
- **CUDA Version:** 13.1
- **Active Compute Process:** PID 1032 (`...a\lib\ollama\llama-server.exe`)

---

## Model Specifications (`ollama show llama3.2`)
- **Model Tag:** `llama3.2` (resolves to `llama3.2:latest` / `3b`)
- **Model ID:** `a80c4f17acd5` *(run `ollama list` to retrieve your local image hash)*
- **Architecture:** `llama`
- **Parameter Count:** `3.2B`
- **Quantization:** `Q4_K_M`
- **Context Length:** `131,072` tokens
- **Embedding Length:** `3072`

---

## Memory & VRAM Measurements
- **Total Dedicated VRAM:** `8,151 MiB` (~8 GB) [Measured via `nvidia-smi`]
- **VRAM With Model Loaded:** `3,829 MiB` [Measured via `nvidia-smi`]
  - *Includes `llama-server.exe` (PID 1032), display desktop environment, and open applications (VS Code, Chrome, Edge, DuckDuckGo).*
- **Baseline VRAM (Without Model):** ~`1,600 MiB` – `1,800 MiB` *(Calculated: `3,829 MiB` minus ~`2,100 MiB` model weight and KV cache overhead)*

---

## Performance & Speed Benchmarks (`--verbose`)

### Run 1:
- **Total Duration:** `804.6034 ms` (0.80 s) [Measured]
- **Load Duration:** `5.3194 ms` [Measured]
- **Prompt Evaluation Count:** `37` tokens (Cached: `20` tokens) [Measured]
- **Prompt Evaluation Duration:** `87.65 ms` [Measured]
- **Prompt Evaluation Speed:** `193.95 tokens/s` [Measured]
- **Generation (Eval) Count:** `64` tokens [Measured]
- **Generation (Eval) Duration:** `688.494 ms` [Measured]
- **Generation Speed:** `92.96 tokens/s` [Measured]

### Run 2:
- **Total Duration:** `1.1135813 s` (1,113.58 ms) [Measured]
- **Load Duration:** `3.7885 ms` [Measured]
- **Prompt Evaluation Count:** `37` tokens (Cached: `36` tokens) [Measured]
- **Prompt Evaluation Duration:** `200.616 ms` [Measured]
- **Prompt Evaluation Speed:** `4.98 tokens/s` [Measured]
- **Generation (Eval) Count:** `81` tokens [Measured]
- **Generation (Eval) Duration:** `891.815 ms` [Measured]
- **Generation Speed:** `90.83 tokens/s` [Measured]

---

## Model Outputs

### Prompt:
> *"In two sentences, what does a pharmacy reorder level mean?"*

### Response 1:
> "A pharmacy reorder level refers to the quantity of a medication that a pharmacy will automatically reorder when a certain amount is sold or used. This level is set by the pharmacy or the patient's insurance plan to ensure that the medication is always available and to prevent stockouts, which can lead to inconvenience and decreased patient adherence."

### Response 2:
> "A pharmacy reorder level is a predetermined quantity of a medication that a pharmacy is willing to hold in stock before reordering from the manufacturer. This level can vary depending on the pharmacy and the medication, but it is typically a specific quantity (e.g. 10, 20, or 50 units) that the pharmacy has agreed to maintain in stock to ensure that patients can receive their prescribed medication."