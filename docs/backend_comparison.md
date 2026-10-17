# Backend comparison: first smoke test

One run per backend through `scripts/try_backend.py` (backend chosen by `LLM_BACKEND` via the factory). This is a smoke test, not a benchmark: one run each cannot show how reliable either backend is.

Labels: **Measured** = printed by the script. **Derived** = calculated from measured values. **Observed** = something I saw, not a printed figure. **Judged** = Judged against the tracker’s definition.   

## Setup (same for both)
- Prompt: *"What does a pharmacy reorder level mean?"*
- System prompt: *"You are a pharmacy operations assistant. Answer in one short sentence."*
- Temperature 0.0, no output-token limit.
- Ollama: `llama3.2:latest`, local. Gemini: `gemini-3.6-flash` 
- Ollama: one run. Gemini: one run. 
## Results

| | Ollama | Gemini |
|---|---|---|
| Prompt tokens [Measured] | 46 | 24 |
| Completion tokens, visible answer [Measured] | 32 | 23 |
| Reasoning tokens [Measured] | None (Ollama reports none) | 309 |
| Total output tokens, completion + reasoning [Derived] | 32 | 332 |
| Latency [Measured] | 10.10 s | 2.78 s |
| One sentence, as instructed [Observed] | Yes | Yes |
| Correct? [Judged] | Partly | Yes |

### Answers
- **Ollama:** "A pharmacy reorder level is the minimum quantity of a medication that must be ordered to ensure a steady supply, typically set by the pharmacy or a designated manager."
- **Gemini:** "A pharmacy reorder level is the minimum inventory threshold that triggers a new order of medication to prevent a stockout."

### Correctness [Judged]
- **Ollama, partly correct.** It is right that the pharmacy or a manager sets the level, and it does not repeat the earlier "patient's insurance plan" error. But it describes the reorder level as a quantity to order. A reorder level is the stock level that triggers an order; the amount ordered is the reorder quantity.
- **Gemini, correct.** It gives the standard meaning: a stock threshold that triggers an order.

## Notes
- **[Observed]** Ollama's text and token counts (46 prompt, 32 completion) are identical to the earlier temperature-0 run with the same system prompt, so the output was repeatable across sessions here.
- **[Observed]** Latency is not comparable between the two. The same Ollama call took 0.34 s earlier ([Measured] in the sampling test, warm model), so the 10.10 s here is consistent with a cold start (model loading). I did not isolate this.
- **[Not comparable]** Prompt tokens differ (46 vs 24) because the two models use different tokenizers.
- **[Derived]** Gemini spent 309 of its 332 output tokens (93%) on reasoning, against 23 visible. A report that showed only completion tokens would hide almost all of its output; this is why `reasoning_tokens` is a separate field.

## What this run does and does not verify about the live Gemini API
- **Accepted by the live API:** a `system_instruction`, `temperature=0.0` and a single user turn through the real SDK. The usage fields `prompt_token_count`, `candidates_token_count` and `thoughts_token_count` were all populated and mapped as expected.
- **Not exercised:** the `"model"` role for assistant turns (this call had only one user message), and `max_output_tokens` (no limit was set). Both are still unconfirmed against the live API.

Unverified as of now: the assistant role "model", max_output_tokens together with thinking tokens, and tool calling.    