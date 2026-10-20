# Agent evaluation on the 25-question test set (Steps 6 to 8)

All runs: temperature 0, store `chroma_db/noctx`, k = 5, tracker running locally. Command:
`py scripts/eval_agent.py`. Logs: `eval/runs/agent-*.jsonl`. Labels: **Measured** = printed by the script;
**Untested** = not run. Model comparison: `model_comparison.md`.

## Runs on llama3.2 Q4_K_M [Measured]

| Run (log) | Change before the run | Passed |
|---|---|---|
| agent-20261020-183257 | First agent: tool loop, branch list, fixed refusals in the prompt, citation repair | 16 of 25 |
| agent-20261020-184227 | Input guard, output-leak guard, text-only repair, "null" strings | 16 of 25 |
| agent-20261020-185119 | Pre-retrieval, medical mode, hedge and made-up-reference guards, num_ctx 8192 | 16 of 25 |
| agent-20261020-191301 | Refusal sentences owned by code, branch aliases, product filter, JSON-line and prompt-echo guards | 20 of 25 |
| agent-q4 | No citation repair on tracker-only answers (bug found on a live query, see below) | 19 of 25 |

Last run, by route: rag 4 of 8, rag_unanswerable 2 of 2, tool 6 of 6, both 1 of 2, rag_then_refer 0 of 1,
refuse 6 of 6. Largest single prompt 3,763 tokens (context window 8,192).

## What changed the result

The same total (16) in the first three runs hid different failures each time. The step that moved it was
taking the medical and scope refusal sentences away from the model: a 3B model used them on the wrong
questions (a returns question got the medical refusal, a payments question got the scope refusal). Code now
decides both, from patterns on the question (`app/guards.py`). The model is asked to use only one fixed
sentence, the not-covered one.

## Found on live queries, not in the test set

- **Misattribution (fixed).** On "what batches are expiring in the next 30 days at Quezon City Hub?" the repair
  turn pasted document citations onto every tracker row, claiming the batch data came from the SOP. Repair
  now runs only when the model searched the documents itself or no tracker data came back.
- **Valid but irrelevant citation (not fixed).** In the chat page, "For the first item on that list, how
  should we receive and store it?" (Latex Surgical Gloves) got a correct, cited receiving procedure, plus a
  paragraph on prescription returns cited to `Customer-Returns`. The citation is valid; the claim does not
  apply to gloves.

## Remaining failures (agent-q4) [Measured]

| Id | Route | What happened |
|---|---|---|
| Q01 | rag | Correct rule, cited, but "antibiotics" absent |
| Q02 | rag | Cited, but merges two rules (90-day items "to the quarantine bin") and omits the 30-day and batch-number rule |
| Q04 | rag | Cited steps are right, but omits the 2°C–8°C range and the 30-minute threshold |
| Q07 | rag | Correct and cited, but says "liver toxicity" where the test set requires "hepatotoxicity" |
| Q18 | both | Tool call and receiving steps right, but no citation of `Temperature-Monitoring` |
| Q19 | rag_then_refer | Medical refusal correct; the model cited none of its leaflet sentences, so the medical filter removed them |

## Limits of these numbers

- **The fixes were designed against these same 25 questions.** 19 or 20 of 25 is a tuned result, not an
  estimate of quality on new questions. A held-out set is needed for that. **Untested.**
- **must_contain is a substring check.** It fails correct paraphrases (Q07) and cannot see a wrong fact that
  contains the right words.
- **A valid citation shows the source was retrieved, not that it supports the sentence** (the gloves example).
- **The guards are regular expressions,** tested against every test-set question and a few hand-written ones,
  not against a broad set of real staff questions or adversarial prompts. **Untested** at scale.
- One run per change at temperature 0. The drop from 20 to 19 between the last two runs came with a code
  change, so it is not pure noise, but one question either way is too small to read.

## Known fix, not applied

Q19: when the question names a drug that has a leaflet in the knowledge base and is not a dose request, append
that leaflet's retrieved section verbatim with its citation after the medical refusal. **Untested.**