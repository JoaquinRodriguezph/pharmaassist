# Model comparison: Q4 vs Q8 quantization, local vs hosted (Step 8)

Same agent, same code (commit 3571280), same tracker and knowledge base, temperature 0, prefetch on, k = 5.
Command: `py scripts/eval_agent.py`, then `py scripts/compare_runs.py`. Logs: `eval/runs/agent-q4.jsonl`,
`agent-q8.jsonl`, `agent-gemini.jsonl`. Machine: Windows 11, RTX 5050 (8 GB). Labels as in the other notes.

## Models

| Label | Model | File size | VRAM loaded, 8,192-token context [Measured, `ollama ps`] |
|---|---|---|---|
| q4 | `llama3.2:latest` (3.2B, Q4_K_M) | 2.0 GB | 3.1 GB, 100% GPU |
| q8 | `llama3.2:3b-instruct-q8_0` (3.2B, Q8_0) | 3.4 GB | 4.5 GB, 100% GPU |
| gemini | `gemini-3.6-flash`, free tier | - | hosted |

## Results [Measured]

| | q4 | q8 | gemini |
|---|---|---|---|
| Questions | 25 | 25 | 6 (one per route) |
| Passed | 19 of 25 | 19 of 25 | 4 of 6 |
| Passed on the 6 shared questions | 5 of 6 | 5 of 6 | 4 of 6 |
| tool route | 6 of 6 | 6 of 6 | 1 of 1 |
| refuse route | 6 of 6 | 6 of 6 | 1 of 1 |
| Median latency per question | 2.8 s | 3.3 s | 26.0 s |
| Total latency | 67 s (25 questions) | 98 s (25 questions) | 151 s (6 questions) |
| Largest single prompt | 3,763 tokens | 3,761 tokens | 3,684 tokens |

- **q4 and q8 fail different questions.** q4: Q01, Q02, Q04, Q07, Q18, Q19. q8: Q02, Q03, Q04, Q06, Q18, Q19.
  Every difference is a `must_contain` string (for example "tamper seal" in Q03, "hepatotoxicity" in Q07).
  On 25 questions with one run each, a swap of two questions either way is not evidence that either
  quantization answers better.
- **Gemini** failed Q17 (called the tracker, never stated the policy half) and Q19 (as both local models did).
  One request was retried after an HTTP 503. Its latency includes the network, free-tier queuing and
  hidden reasoning; it is not a measure of a paid tier.
- **Sending tool results back to Gemini now works live** (Q13: tool call, result sent back, correct answer).
  This was the last unverified part of the Gemini tool-calling path.

## Decision

**The service defaults to q4 on Ollama.** Same pass count as q8 on this set, about 1.4 GB less VRAM, lower
median latency, and no per-request quota. Gemini stays available as a backend (`LLM_BACKEND=gemini`) but is
not the default: the free tier allows about 20 requests a day, an agent question uses 2 to 3 of them, and its
median latency here was about nine times the local one.

## Limits

- One run per model; temperature 0 makes runs repeatable, not representative of variance across prompts.
- The 25 questions are the ones the agent was tuned on (see `agent_eval.md`), so absolute pass counts are
  optimistic for all three models. The comparison between them is fairer than the absolute numbers.
- Gemini was measured on 6 questions only, to stay inside the free daily quota.
- **Untested:** other quantizations (Q5_K_M, FP16), larger local models, Gemini on all 25 questions, cost per
  question on a paid tier.