# Retrieval notes

Step 4, first look: which embedding model, how to install it, and what the first embeddings show. Labels: **Card** = stated on the model's Hugging Face card; items marked **pending** have not yet been read from the card in the local snapshot (the copy of the card in the Hugging Face cache on this machine). **Docs** = stated in the sentence-transformers documentation. **Measured** = printed by `scripts/try_embeddings.py`. **Derived** = calculated from other numbers. **Prediction** = written before running anything.

## Embedding model

**Choice: `sentence-transformers/all-MiniLM-L6-v2`**, run on CPU through the `sentence-transformers` library.

| | Value | Source |
|---|---|---|
| Name | `sentence-transformers/all-MiniLM-L6-v2` | |
| Embedding dimension | 384 | Card ("384 dimensional dense vector space"); **Measured**: the script printed 384. |
| Maximum sequence length | 256 word pieces; longer input is truncated | Card ("By default, input text longer than 256 word pieces is truncated"); Docs (longer texts are truncated to the first `max_seq_length` tokens) |
| License | Apache-2.0, **pending**: not yet read from the card in the local snapshot | Hugging Face metadata of models derived from it |
| Size | 22.7 million parameters, stored as 32-bit floats; `model.safetensors` is 90.9 MB | Parameters from the Hugging Face metadata of derived models. **Measured**: the download log showed `model.safetensors` at 90.9 MB, which matches the Derived figure (22.7 million x 4 bytes = 90.8 MB). |
| Prefixes ("query:" / "passage:") | No mention of prefixes in the card; the card's own example encodes plain sentences. **Pending**: a search of the local snapshot for `prefix`, `query:` and `passage:` has not yet been recorded. | Card |
| Training | Fine-tuned from `nreimers/MiniLM-L6-H384-uncased` on over 1 billion sentence pairs; intended as a sentence and short-paragraph encoder | Card |

**A gotcha with the length limit, and a correction.** A discussion thread on the model's page reported that its tokenizer configuration had been changed to 512, while the 256 limit lives in `sentence_bert_config.json`, so plain `transformers` would truncate at 512 and give different embeddings. In the run below the tokenizer reported 256 as well (**Measured**: `tokenizer.model_max_length: 256`), so that configuration seems to have been reverted, and my earlier note that it said 512 is out of date. The advice stands: load the model through `sentence-transformers`, and read `max_seq_length` from the model rather than assuming.

**Why this one first.**
- It covers the Hugging Face requirement and needs no API key or quota.
- It is small enough for CPU.
- It is the model Chroma's chunking report used for its second set of results, so that report is a useful comparison.
- The card mentions no prefix convention, so there is one less thing to get wrong in a first run.
- Our chunks are 132 to 756 characters, and the limit is measured in tokens, so I expect them to fit (checked by the script, see Token length).

**Limits of the choice.** It is an old model (2021) and newer small models may retrieve better, especially on specialist text such as medicine; one third-party page claims a large gap on a domain benchmark, which I have not verified. Other small models (for example the E5 family) use "query:" and "passage:" prefixes, and some others use a query instruction; that is another variable to test later, not part of a first run. The script takes the model name as a parameter (`--model`), so swapping is cheap.

**Fallback.** If PyTorch will not install on Windows with Python 3.12, paste the error and we switch to an embedding model served by Ollama.

## Install

Python 3.12.10 is within what the sentence-transformers README recommends (Python 3.10+, PyTorch 1.11.0+, transformers 4.41.0+). The README's install command is `pip install -U sentence-transformers`, and for a GPU it says to install PyTorch with matching CUDA support first. A CPU build is enough here. A third-party install guide shows installing the CPU wheel first, so that `sentence-transformers` reuses it (I did not read the PyTorch selector page itself):

```
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
py -m pip install sentence-transformers
```

Add two lines to `requirements.txt`, by hand:

```
sentence-transformers
torch
```

This is a large dependency (the same guide shows the CPU `torch` download at about 143 MiB), so the requirements files will be split at Docker time.

**Versions: pending.** Not yet recorded; `py -m pip show sentence-transformers torch chromadb` gives them.

**Result (Measured):** the packages installed and the script ran on Python 3.12.10 on Windows, so the Ollama fallback is not needed. The versions of `sentence-transformers` and `torch`, and which `torch` build was installed, were not recorded; `py -m pip show sentence-transformers torch` would fix that. The run printed three warnings, none an error: unauthenticated requests to the Hugging Face Hub (a `HF_TOKEN` is optional and gives higher rate limits; keep it out of the repository), a symlinks warning for the Hugging Face cache on Windows (harmless; `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences it), and a deprecation notice for `get_sentence_embedding_dimension`, which the script no longer calls.

## Cosine similarity

`app/rag/similarity.py`, `cosine_similarity(a, b)`. Mismatched lengths, empty vectors, zero vectors and NaN or infinite values raise `ValueError`; the result is clamped to [-1, 1]. A zero vector raises instead of returning 0.0 because a broken embedding would otherwise rank as "unrelated" without anyone noticing. The reasons are in the module docstring.

## First look at the embeddings

Run `py scripts/try_embeddings.py`. It chunks the knowledge base at 800 characters with no overlap (14 chunks), embeds each chunk with and without the context prefix, embeds the exact text of Q01, Q06, Q08, Q09 and Q22, and prints the top 3 chunks and scores for each query and each version.

### Predictions (written before the first run)

| Question | Expected rank 1 |
|---|---|
| Q01 (return unopened amoxicillin, has receipt) | `returns_and_expiry_policy.md#Customer-Returns` |
| Q06 (max paracetamol 500 mg tablets in 24 hours) | `paracetamol_leaflet.md#Dosage-And-Administration`; guess for rank 2: `paracetamol_leaflet.md#About-This-Medicine`; the typo "paracetmol" may weaken it |
| Q08 (who should not take ibuprofen) | `ibuprofen_leaflet.md#Contraindications` |
| Q09 (employee discount, not in the KB) | No right answer. I expect `returns_and_expiry_policy.md#Customer-Returns` anyway, because it mentions OTC items and vitamins, with a middling score below the answerable questions |
| Q22 (write a scraper, out of scope) | No right answer, no confident guess; I expect the lowest scores of the five |

Also predicted: no chunk over the 256-token limit at 800/0, the longest (756 characters) at roughly 150 to 220 tokens, and the prefix adding roughly 15 to 25 tokens. The prefix should raise the right leaflet's score for Q06 and Q08 and change little for Q01.

### Token length and truncation

The script counts tokens per chunk with the model's own tokenizer (special tokens included, no truncation), with and without the prefix, and compares the maximum with `model.max_seq_length`. Per the docs, text beyond the limit is truncated: only the first 256 tokens are embedded, so a chunk that is too long would be embedded without its ending and nothing would warn you. The prefix counts toward the limit.

### Measured (first run of `py scripts\try_embeddings.py`)

**Model.** Dimension 384; `model.max_seq_length` 256; `tokenizer.model_max_length` 256; 14 chunks at `max_chars=800`, `overlap=0`.

**Token lengths** (the model's own tokenizer, special tokens included). "Prefix adds" is Derived.

| Chunk | Chars | Tokens | Tokens with prefix | Prefix adds |
|---|---|---|---|---|
| `ibuprofen_leaflet.md#About-This-Medicine` | 518 | 116 | 132 | 16 |
| `ibuprofen_leaflet.md#Contraindications` | 594 | 151 | 168 | 17 |
| `ibuprofen_leaflet.md#Drug-Interactions` | 608 | 138 | 153 | 15 |
| `paracetamol_leaflet.md#About-This-Medicine` | 522 | 116 | 134 | 18 |
| `paracetamol_leaflet.md#Dosage-And-Administration` | 632 | 152 | 171 | 19 |
| `paracetamol_leaflet.md#Warnings-And-Precautions` | 651 | 137 | 155 | 18 |
| `returns_and_expiry_policy.md#Purpose-And-Scope` | 614 | 121 | 134 | 13 |
| `returns_and_expiry_policy.md#Customer-Returns` (part 1/2) | 687 | 149 | 161 | 12 |
| `returns_and_expiry_policy.md#Customer-Returns` (part 2/2) | 132 | 30 | 42 | 12 |
| `returns_and_expiry_policy.md#Expired-Medications` | 756 | 174 | 186 | 12 |
| `storage_sop.md#Purpose-And-Scope` | 573 | 113 | 124 | 11 |
| `storage_sop.md#Temperature-Monitoring` | 697 | 164 | 174 | 10 |
| `storage_sop.md#Cold-Chain-And-Excursions` | 742 | 169 | 182 | 13 |
| `storage_sop.md#Receiving-And-FEFO` | 607 | 148 | 160 | 12 |

- **[Measured]** The longest chunk is 174 tokens (186 with the prefix) against a limit of 256, and no chunk is over the limit in either version, so nothing is truncated at 800/0.
- **[Derived]** The longest chunk with its prefix leaves 70 tokens of headroom (256 - 186). The prefix adds 10 to 19 tokens. The longest chunk is about 4.3 characters per token (756 / 174).

**Top 3 per question** (cosine score, chunk; a check mark means the expected section).

| Q01, return unopened amoxicillin | Without prefix | With prefix |
|---|---|---|
| 1 | 0.640 Customer-Returns (part 1/2) ✓ | 0.619 Customer-Returns (part 1/2) ✓ |
| 2 | 0.452 Customer-Returns (part 2/2) ✓ | 0.423 Customer-Returns (part 2/2) ✓ |
| 3 | 0.364 returns Expired-Medications | 0.386 returns Purpose-And-Scope |

| Q06, max paracetamol tablets in 24 hours | Without prefix | With prefix |
|---|---|---|
| 1 | 0.705 Dosage-And-Administration ✓ | 0.689 Dosage-And-Administration ✓ |
| 2 | 0.566 paracetamol About-This-Medicine | 0.622 paracetamol About-This-Medicine |
| 3 | 0.369 paracetamol Warnings-And-Precautions | 0.479 paracetamol Warnings-And-Precautions |

| Q08, who should not take ibuprofen | Without prefix | With prefix |
|---|---|---|
| 1 | 0.709 ibuprofen About-This-Medicine | 0.762 ibuprofen About-This-Medicine |
| 2 | 0.632 ibuprofen Drug-Interactions | 0.693 ibuprofen Drug-Interactions |
| 3 | 0.603 Contraindications ✓ | 0.671 Contraindications ✓ |

| Q09, employee discount (not in the KB) | Without prefix | With prefix |
|---|---|---|
| 1 | 0.388 returns Purpose-And-Scope | 0.395 returns Customer-Returns (part 1/2) |
| 2 | 0.371 returns Customer-Returns (part 1/2) | 0.386 returns Purpose-And-Scope |
| 3 | 0.238 storage Purpose-And-Scope | 0.262 storage Purpose-And-Scope |

| Q22, write a scraper (out of scope) | Without prefix | With prefix |
|---|---|---|
| 1 | 0.208 paracetamol Dosage-And-Administration | 0.223 returns Expired-Medications |
| 2 | 0.203 storage Purpose-And-Scope | 0.209 paracetamol About-This-Medicine |
| 3 | 0.199 storage Receiving-And-FEFO | 0.208 paracetamol Dosage-And-Administration |

**Summary** (top-1 score, and the rank of the first chunk of the expected section).

| Question | Category | Top-1 without | Top-1 with | Expected rank (without / with) |
|---|---|---|---|---|
| Q01 | rag_answerable | 0.640 | 0.619 | 1 / 1 |
| Q06 | rag_answerable | 0.705 | 0.689 | 1 / 1 |
| Q08 | rag_answerable | 0.709 | 0.762 | 3 / 3 |
| Q09 | rag_unanswerable | 0.388 | 0.395 | none |
| Q22 | medical_or_out_of_scope | 0.208 | 0.223 | none |

### Predictions against results

| Prediction | Result |
|---|---|
| Q01 rank 1 is Customer-Returns | Right. Part 1 is rank 1 and part 2 is rank 2. |
| Q06 rank 1 is Dosage, rank 2 is About-This-Medicine | Right on both, despite the typo "paracetmol". |
| Q08 rank 1 is Contraindications | **Wrong.** It is rank 3 in both versions; rank 1 is the leaflet's About-This-Medicine section. |
| Q09 rank 1 is Customer-Returns, with a middling score below the answerable questions | Half right. The returns policy did take the top two places, but without the prefix rank 1 was its Purpose-And-Scope section (0.388), with Customer-Returns second (0.371). The score is about 0.39, well below the answerable questions. |
| Q22: no guess; the lowest scores of the five | Right on the scores (0.208 and 0.223). The top three are within 0.009 of each other without the prefix and 0.015 with it [Derived], so the order is noise. |
| No chunk over 256 tokens; longest 150 to 220 tokens | Right: 174. |
| The prefix adds roughly 15 to 25 tokens | Wrong: it adds 10 to 19 [Derived]. |
| The prefix raises the right leaflet's score for Q06 and Q08 and changes little for Q01 | Mixed. Q06's right chunk went down (0.705 to 0.689); Q08's went up (0.603 to 0.671) but its siblings went up too and its rank stayed 3; Q01 changed by -0.021 [Derived]. |

### What the first look can and cannot say

- **[Observed] Answerable questions do score visibly higher than Q09 and Q22.** Their top-1 scores are 0.640 to 0.709 without the prefix (0.619 to 0.762 with it), against 0.388 (0.395) for Q09 and 0.208 (0.223) for Q22. The lowest answerable top-1 is 0.252 above the highest unanswerable one without the prefix, and 0.224 with it [Derived]. Q09 is the closest, which fits it being built to look like the returns policy.
- **This is not a threshold.** It is three answerable and two unanswerable questions against 14 hand-written chunks. Scores in between exist (Q01's second chunk is 0.452, Q06's is 0.566), and paraphrases, typos and other phrasings of an unanswerable question could land anywhere. A threshold needs many more questions.
- **[Observed] A high score does not mean the right section.** For Q08 the right chunk is third, behind two other ibuprofen sections with scores of 0.709 and 0.632. A top-3 retrieval still contains it; a top-1 retrieval would miss it. This is why top-3 or top-5 retrieval is safer than top-1, and why hit at k will be measured across all the questions, including reworded and misspelled variants, in Step 4c. I do not know why Q08 ranks where it does. Two guesses, neither tested: the question's words ("ibuprofen", "product leaflet") match the general description better than the contraindication list, and embedding models often handle "should NOT take" poorly.
- **[Observed] The prefix shows no benefit in this sample.** It raised the scores of whole documents (all three ibuprofen sections went up by 0.053 to 0.068 [Derived]) but did not change any expected rank. Every question here named its document, so none could show the prefix helping to pick the right leaflet. Keep both versions for the larger test.
- **[Observed]** The two typos ("amoxicilin", "paracetmol") still found the right chunk; that is two examples.
- **Next, for Step 4:** measure hit at 1 and hit at 3 across all the RAG questions, with and without the prefix, and add a few paraphrased and misspelled versions of each question, before choosing a threshold or a prefix setting.