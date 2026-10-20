# Retrieval evaluation (Step 4c)

How often the right section lands in the top k, with and without the document/section prefix. **Measured** = printed by `scripts/eval_retrieval.py` on this machine. **Derived** = calculated from other numbers. **Prediction** = written before the first run. **Untested** = a hypothesis. **Judged** = a labeling decision of mine.

## Setup

```
py scripts\ingest_kb.py --db chroma_db\ctx --context
py scripts\ingest_kb.py --db chroma_db\noctx --no-context
py scripts\eval_retrieval.py                                         # original + variants
py scripts\eval_retrieval.py --sets original variants independent    # adds the independent questions
```

Both stores hold the same 14 chunks (800 characters, no overlap) from the same knowledge base, embedded with `all-MiniLM-L6-v2`. Only the prefix differs. The two commands name the prefix explicitly, so the result does not depend on `RAG_WITH_CONTEXT`. Each ingest took about 0.6 s [Measured: 0.58 s and 0.60 s]. The script checks that the two stores really were built with and without the prefix.

## Metrics

Defined in `app/rag/metrics.py`, in terms of the section id of each retrieved chunk (a ranking, best first).

- **hit@k:** at least one of the top k chunks is from an expected section.
- **all-hit@k:** every expected section appears in the top k. It differs from hit@k only for Q18, which cites two sections.
- **Reciprocal rank:** 1 divided by the rank of the first chunk from an expected section (0 if none in the top 5). **MRR** is its mean over questions.
- **Chunks of one section:** ranks count chunks as retrieved, so `Customer-Returns::1` and `::2` are two slots, and a section's rank is its first chunk's rank. These are the slots an LLM would receive in Step 5. The script also prints MRR over distinct sections (repeats collapsed first), and how many of the top 5 slots are taken by repeats.
- **Questions the KB cannot answer** (Q09, Q10, Q20, and any independent question with no expected section) have no hit to measure. Only their top-1 and top-3 scores are printed, and no threshold is drawn from them.

## Question sets

| Set | Questions | Written by | Leakage |
|---|---|---|---|
| original | 11 with expected sections (Q01 to Q08, Q17, Q18, Q19), plus Q09, Q10 and Q20 by score | the test set | The documents were written after these questions, to hold exactly the facts they ask for, so this set flatters retrieval |
| variants | 11, one per question: keywords only (2), typo-heavy (2), different phrasing (2), Taglish (2), very short, long and chatty, different vocabulary | me, after seeing the documents | Written knowing what the documents say, so fairer in wording but not independent |
| independent | 8, in the writer's own words about returns, expiry, storage temperature, paracetamol and ibuprofen | someone who had not seen the documents, given only the topic list; labeled by me afterwards | This is the set that reduces the problem. Received, labeled and run (results below), but only two of its eight questions can be scored by rank, so I am asking a second writer for ten more questions about what the rules say |

Q17 and Q18 also need a tool call in the full system; here only the document part is measured, using the question as written.

### The independent questions and how I labeled them [Judged]

Each question gets three labels, and the loader (`load_independent` in `app/rag/evalsets.py`) rejects a file in which they disagree.

- **`expected_sources`:** a section only if the knowledge base holds a rule or fact that a correct answer must be based on. A question whose answer needs live information (a log reading, a box's location, shelf contents, a delivery's history, a checklist, the air conditioning) gets an empty list, even when a section is on the same topic. Those sections are recorded as `related_sections` for reference; they are not scored by rank.
- **`expected_route`:** one of the routes in `eval/test_set.json` (`rag`, `tool`, `both`, `rag_unanswerable`, `rag_then_refer`, `refuse`). `rag`, `both` and `rag_then_refer` need an expected source; `tool` and `both` need an `expected_tool`.
- **`expected_behavior`**, and for most questions `must_contain`: what a correct answer does, and strings it must state. A test checks that every `must_contain` string really appears in the question's expected or related sections.

I first assumed the six situational questions would be tool questions. That was wrong for most of them: a box's location, sunlight on a delivery, an air-conditioning fault and an audit checklist are not in any tool's data either. Only IND03 maps to a tool.

| Id | Route | Expected sources | Why |
|---|---|---|---|
| IND04 (expired paracetamol drops, "pull-out bin") | `rag` | `Expired-Medications` | Expired stock is logged by batch number and moved to the quarantine bin |
| IND05 (blister seal popped) | `rag` | `Customer-Returns` | Returns need an unopened item with an intact tamper seal |
| IND03 (near-expiry ibuprofen syrup on a shelf) | `tool` (`get_expiring_batches`) | none | Near-expiry stock is what `/reports/expiring` returns. No branch is named, so `expected_args` is null and the assistant should ask which branch. The data has no shelf location |
| IND01 (did the stockroom log stay under 25°C) | `rag_unanswerable` | none; related `Temperature-Monitoring` | No tool reports temperature readings; the log is a paper sheet |
| IND02 (where a returned box is, was it tagged) | `rag_unanswerable` | none; related `Customer-Returns`, `Expired-Medications` | One box's location and tag are not in any data |
| IND06 (delivery left in sunlight) | `rag_unanswerable` | none; related `Receiving-And-FEFO` | What happened to a delivery is not in any data, and the SOP says nothing about sunlight |
| IND07 (children's ibuprofen in today's audit checklist) | `rag_unanswerable` | none; related `Expired-Medications` | A checklist is not data; the tool lists expiry dates but not checklists |
| IND08 (why a shelf feels warm, AC or thermostat) | `rag_unanswerable` | none; related `Temperature-Monitoring` | Equipment status is not data |

**Expected behavior for the five `rag_unanswerable` questions.** Each answer must do two things: state the rule from the related section (for example 15°C-25°C and the same-day report to the supervising pharmacist for IND01 and IND08; the 90-day pull and the monthly check for IND07), and say that the assistant cannot see live logs or the current state. It must not answer the live question itself: no "yes, it stayed under 25°C", and no "the box was tagged". This goes beyond Q09 and Q10, which only require saying the information is not in the documents. The main test set has no question of this kind, which is a gap I should fill there too.

**[Observed]** Of the eight independent questions, two are answerable from the documents, one is a tool question and five are situational questions with a rule to state. The two answerable ones use different words from the documents ("pull-out bin" for the quarantine bin, "blister seal popped" for a broken tamper seal). So the independent set can only add two questions that are scored by rank, and the other six show scores only. A looser rule that labeled the related sections as expected would make five of the others scorable, but it would reward retrieving a section that cannot answer the question. If I later disagree with a label, I change it in `eval/retrieval_independent.json`.

## Predictions (written before the first run) and what happened

| Prediction | Result |
|---|---|
| Original set: hit@1 is 9 or 10 of 11 in both stores | Right: 10 (ctx) and 9 (noctx) |
| Original set: hit@3 is 10 or 11 and hit@5 is 11 | Right: 11 and 11 in both |
| Variants: hit@1 around 7 to 9 of 11, hit@5 of 9 or more | Right: 8 and 7 at k=1; hit@5 is 11 (ctx) and 10 (noctx) |
| The keyword-only, typo-heavy and Taglish variants are the likeliest to miss at rank 1 | Partly. Typo-heavy and Taglish did (V02, V07, V04, V08). Keyword-only did not: V01 and V10 were rank 1 in both stores |
| The two stores differ by at most one question at each k and by less than 0.05 in MRR | Right, but only just: MRR gaps are 0.045 and 0.048 |
| Repeats only arise for questions that retrieve `Customer-Returns` | Right: Q01, Q03, Q17, V01 and V03 |

## Results

### Original questions (11) [Measured]

| Id | About | ctx rank | noctx rank |
|---|---|---|---|
| Q01 | Customer-Returns | 1 | 1 |
| Q02 | Expired-Medications | 1 | 2 |
| Q03 | Customer-Returns | 1 | 1 |
| Q04 | Cold-Chain-And-Excursions | 1 | 1 |
| Q05 | Temperature-Monitoring | 1 | 1 |
| Q06 | Dosage-And-Administration | 1 | 1 |
| Q07 | Warnings-And-Precautions | 1 | 1 |
| Q08 | Contraindications | 3 | 3 |
| Q17 | Expired-Medications | 1 | 1 |
| Q18 | Receiving-And-FEFO + Temperature-Monitoring | 1 | 1 |
| Q19 | Drug-Interactions | 1 | 1 |

| Counts out of 11 | ctx | noctx |
|---|---|---|
| hit@1 | 10 | 9 |
| hit@3 | 11 | 11 |
| hit@5 | 11 | 11 |
| all-hit@1 | 9 | 8 |
| all-hit@3 | 10 | 10 |
| all-hit@5 | 11 | 11 |
| MRR (chunk ranks and distinct sections, the same) | 0.939 | 0.894 |
| Top-5 slots taken by repeats | 3 (in 3 questions) | 2 (in 2 questions) |

### Variants (11) [Measured]

| Id | Style | ctx rank | noctx rank |
|---|---|---|---|
| V01 | keywords only | 1 | 1 |
| V02 | typo-heavy | 2 | 2 |
| V03 | different phrasing | 1 | 1 |
| V04 | Taglish | 1 | 2 |
| V05 | very short | 1 | 1 |
| V06 | long and chatty | 1 | 1 |
| V07 | typo-heavy | 1 | 2 |
| V08 | Taglish | 5 | more than 5 |
| V09 | different phrasing | 3 | 1 |
| V10 | keywords only | 1 | 1 |
| V11 | different vocabulary | 1 | 1 |

| Counts out of 11 | ctx | noctx |
|---|---|---|
| hit@1 | 8 | 7 |
| hit@3 | 10 | 10 |
| hit@5 | 11 | 10 |
| all-hit@1 | 7 | 6 |
| all-hit@3 | 10 | 10 |
| all-hit@5 | 11 | 10 |
| MRR (chunk ranks and distinct sections, the same) | 0.821 | 0.773 |
| Top-5 slots taken by repeats | 2 (in 2 questions) | 2 (in 2 questions) |

### Questions the knowledge base cannot answer: top scores only [Measured]

| Id | ctx top-1 (section) | ctx top-3 | noctx top-1 (section) | noctx top-3 |
|---|---|---|---|---|
| Q09 employee discount | 0.395 (Customer-Returns) | 0.395 0.386 0.262 | 0.388 (Purpose-And-Scope) | 0.388 0.371 0.238 |
| Q10 GCash, offline POS | 0.145 (Customer-Returns) | 0.145 0.126 0.108 | 0.142 (Expired-Medications) | 0.142 0.134 0.116 |
| Q20 child's fever dose | 0.590 (Dosage-And-Administration) | 0.590 0.433 0.412 | 0.580 (Dosage-And-Administration) | 0.580 0.476 0.385 |

- **[Observed]** Q09 and Q10 score well below the answerable questions seen so far (the Step 4a top-1 scores for Q01, Q06 and Q08 were 0.619 to 0.762 with the prefix). Q20 does not: 0.590 is only 0.029 below Q01's 0.619 [Derived]. A threshold between the lowest answerable and the highest unanswerable scores would let Q20 through. Q20 is a refusal case, and its high score is because it really is about the dosage section. The refusal has to come from a rule about dosing, not from a low retrieval score.
- This is what the earlier "answerable questions score higher than Q09 and Q22" first look could not show. It is three questions, so it is still not a threshold.

### Independent questions (8) [Measured]

The two questions the knowledge base can answer:

| Id | About | ctx rank | noctx rank |
|---|---|---|---|
| IND04 | Expired-Medications | 1 | 1 |
| IND05 | Customer-Returns | 1 | 1 |

Both stores: hit@1, hit@3 and hit@5 are 2 of 2 and MRR is 1.000. IND05 has a repeated `Customer-Returns` chunk in its top 5 (1 slot).

The six it cannot answer, top scores only (ctx / noctx):

| Id | Asks about | Top-1 section | Top-1 score | Top-3 (ctx) |
|---|---|---|---|---|
| IND01 | whether today's stockroom log stayed under 25°C | storage_sop.md#Temperature-Monitoring | 0.592 / 0.627 | 0.592 0.458 0.299 |
| IND02 | where a returned paracetamol box is | paracetamol_leaflet.md#About-This-Medicine | 0.530 / 0.530 | 0.530 0.433 0.407 |
| IND03 | near-expiry ibuprofen syrup on a shelf | ibuprofen_leaflet.md#About-This-Medicine | 0.567 / 0.569 | 0.567 0.466 0.445 |
| IND06 | a delivery left in sunlight | paracetamol_leaflet.md#Dosage-And-Administration | 0.363 / 0.380 | 0.363 0.354 0.345 |
| IND07 | an audit checklist for children's ibuprofen | ibuprofen_leaflet.md#About-This-Medicine | 0.470 / 0.468 | 0.470 0.449 0.404 |
| IND08 | why a shelf feels warm | storage_sop.md#Temperature-Monitoring | 0.328 / 0.353 | 0.328 0.284 0.228 |

- **[Observed]** The two answerable questions are rank 1 in both stores even though they use different words from the documents ("pull-out bin", "blister seal popped"). That is two questions, so it is encouraging and not evidence.
- **[Observed]** Scores do not separate what the knowledge base can answer from what it cannot. Across the nine questions with no answer in the knowledge base (Q09, Q10, Q20 and these six) the top-1 score runs from 0.142 to 0.627. IND01's 0.627 without the prefix is only 0.013 below Q01's 0.640 [Derived]. Only three answerable questions have a recorded top-1 score (Step 4a: 0.619 to 0.762 with the prefix, 0.640 to 0.709 without), so this is not a threshold study, but it is evidence against building a refusal on the top score alone.
- **[Derived]** Counting the section I recorded as related to each question, the top-1 section matched it for 2 of the 6 (IND01 and IND08, both Temperature-Monitoring). For the four that name a drug (IND02, IND03, IND06, IND07), the top-1 section is that drug's About-This-Medicine or Dosage section instead of the topic section (expiry, returns, receiving).
- **Hypothesis (untested):** a drug name in the question pulls retrieval toward that drug's leaflet more strongly than the topic words do. Q08 and V08 fit the same pattern. If true, questions that mention a drug will return the leaflet even when the topic belongs elsewhere.
- **If the labels were looser** (the related sections counted as expected), only the top-1 results are printed, so I can say only that 2 of the 6 would be hits at rank 1; I cannot say how many would be hits at rank 5.

## What the prefix setting did

- **[Measured]** The prefix store (ctx) is equal or better on every aggregate number, in both sets: hit@1 10 against 9 and 8 against 7, hit@5 equal on the original questions and 11 against 10 on the variants, MRR 0.939 against 0.894 and 0.821 against 0.773. The gaps in MRR are 0.045 and 0.048.
- **[Derived]** Question by question over the 22 questions, the prefix was better on 4 (Q02, V04, V07, V08), worse on 1 (V09) and tied on 17. With 5 questions that differ, a 4 to 1 split or more extreme happens by chance about 38% of the time (exact binomial, two-sided: 2 x 6/32 = 0.375).
- **Conclusion:** the numbers lean toward the prefix but cannot separate the two settings. One question (V08) is the only one that is a hit at k=5 with the prefix and not without.

## Which questions miss at rank 1, and why (hypotheses, untested)

Rank-1 misses [Measured]: with the prefix Q08, V02, V08, V09; without it Q02, Q08, V02, V04, V07, V08.

- **Q08 and V08 (ibuprofen contraindications, original and Taglish):** rank 1 is the leaflet's About-This-Medicine section in both stores. The question words "ibuprofen" and "product leaflet" may match the general description better than a list of reasons not to take it, and embedding models often handle "should NOT take" poorly. The Taglish version (V08) is the worst result of all, at rank 5 and "more than 5". The model may be weak on Tagalog words: the card's header says `language: en` [Card, snapshot], which fits, but I have not tested whether language is the cause.
- **Q02 and V02 (expiry shelf-pull question, original and typo-heavy):** rank 1 is `Receiving-And-FEFO`, whose text also talks about expiry dates and shelves. The words overlap more than the meanings do.
- **V07 (typo-heavy paracetamol warnings, no prefix):** rank 1 is the sibling `Dosage-And-Administration` section of the same leaflet. A near miss within the same document.
- **V04 and V09:** rank 1 is a section called Purpose-And-Scope, a generic section that matches broad wording. The earlier version of the script printed only the heading, so I cannot say which document's Purpose-And-Scope it was; it now prints the full section id.
- **[Observed]** Of the misses I can attribute, most are a neighboring section of the right document or a section with overlapping vocabulary. None is a random section.

## Limits

- **Small and hand-built.** 22 questions (11 original and 11 variants) against 14 chunks written for the purpose, plus two answerable independent questions. A difference of one question is within what a reworded question could change.
- **Chance is high on a store this small [Derived].** A random ranking would put a given single-chunk section in the top 1 about 7% of the time (1 of 14), the top 3 about 21% and the top 5 about 36%, so hit@5 is an easy bar. Hit@1 and MRR are more informative.
- **The variants share a writer with the documents.** I wrote them after seeing the documents.
- **One embedding model, one chunking setting.** At 800/0 only one section is split, so the repeat counts say little; at 400 characters almost every section splits.
- **The independent set is eight questions, and six are about live state.** Two could be scored by rank, and both were hits, so it does not separate the two prefix settings either. Ten more questions from a second writer are being requested.
- **Scores do not make a threshold** (see Q20 and the independent questions).

## Decision for Step 5

- **k = 5.** The rule I wrote before the run was: use the smallest of 3 or 5 for which every answerable question has a hit; if hit@3 misses any, use 5. Hit@3 misses V08 in both stores, so k is 5, and never 1 (4 of the 22 questions miss rank 1 with the prefix and 6 without it). Even k=5 misses V08 in the no-prefix store [Measured], so no setting here gives every question a hit.
- **Chunks, not distinct sections, for now.** Repeats took 3 of the 55 top-5 slots on the original questions with the prefix [Derived: 11 questions x 5 slots], and MRR is the same either way. This will need another look if a smaller `max_chars` makes more sections split.
- **Context budget with k = 5 [Derived].** The five longest chunks are 174 + 169 + 164 + 152 + 151 = 810 tokens without the prefix, and 186 + 182 + 174 + 171 + 168 = 881 with it (token counts from the table in `docs/retrieval_notes.md`, counted with the MiniLM tokenizer). Either is well inside Ollama's 4096-token default context (`docs/local_model_notes.md`). That count is for the chunks only, and it is MiniLM's tokenizer, not Llama's, so it is a rough guide. In Step 5, I recount with Llama's own tokenizer, and add the system prompt, the question, the conversation history and the room the answer needs.
- **Prefix: a setting, defaulting to off.** The two settings cannot be separated, so I applied the tie-break I wrote before the run: no prefix, because it is simpler, it embeds exactly the text that is shown, and (I wrote) it showed no benefit. That last reason is now too strong: the numbers lean toward the prefix, 4 questions better against 1 worse, and V08 is a hit at k=5 only with it. I am keeping the rule because changing it after seeing the results would be choosing the answer I like. Because the difference is small and reversible, the choice is not hardcoded: it is the `RAG_WITH_CONTEXT` environment variable (read by `app/rag/config.py`; unset means off; `--context` and `--no-context` on `scripts/ingest_kb.py` override it). Changing it re-embeds every chunk. I revisit it when the second writer's questions are scored and when the knowledge base has similar sections in different documents, which is where a prefix could help.