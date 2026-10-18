# Chunking notes

How `app/rag/chunking.py` turns the knowledge base into chunks, why, and a first experiment with `max_chars`. All sizes are in characters. **Measured** = printed by `scripts/chunk_stats.py` on the real knowledge base. **Derived** = calculated from measured numbers.

## Decisions

### Which files are documents
A knowledge-base document is a `*.md` file directly inside `docs/knowledge_base/` whose name is not `README.md` (any capitalization) and does not start with `_` or `.`. The rule is one function, `is_kb_document` in `app/rag/kb.py`. The chunker (`kb_documents`) and the citation checker both use it, so a loop over the folder cannot index the README, and a test-set entry such as `README.md#Heading-Rule` is reported as "not a knowledge-base document". Tests cover README in three capitalizations, `_draft.md`, `.hidden.md`, a `.txt` file and a sub-folder.

### Text before the first `##` heading
It is **not indexed**. It holds the banner, the title and the document ID, which carry no facts, and indexing the banner would make "fictional document" a retrievable answer. The title is kept only as `Chunk.doc_title`, for the optional context prefix. The risk is silent loss: a fact placed before the first heading would never be indexed. `stray_preamble_lines` finds preamble text that is not a banner, title or document ID line, and `chunk_stats.py` prints a warning for it.

### Splitting
One chunk per section when its text fits in `max_chars`. Otherwise: paragraphs and list items first (packed together up to the limit), then sentences, and only if one sentence alone exceeds the limit, word boundaries; a single word longer than the limit is cut at exactly `max_chars`. The limit applies to `Chunk.text`, not to the optional prefix. Every part keeps the same `source`, `section` and `section_id`. A section with no text makes no chunk, and two headings in one file with the same fragment raise an error. The sentence splitter is a heuristic (it would split after an abbreviation such as "e.g." when a capital letter follows).

### What `overlap` means here
A character budget for **whole sentences**. A part after the first begins with the last whole sentence or sentences of the previous part that fit in `overlap` characters, never the whole previous part. Carried sentences count toward the new part's `max_chars`, carried list items keep their own lines, and overlap never crosses a section boundary. `overlap=0` (the default) means none. I chose this before looking at how other tools define it.

After deciding, I read how others describe it. LangChain's recursive splitter takes a `chunk_size` (a maximum, measured by its length function, characters by default) and a `chunk_overlap` described as the target overlap between chunks, there to reduce information loss when context is split; its default separators go paragraph, line, word, character, and its docs note that words can be split. Chroma's chunking report (Smith and Troynikov, July 2024) states sizes and overlaps in tokens. Its tables show overlap lowering token-level efficiency, because overlapping chunks return the same tokens twice. Recall depended on the embedding model: with text-embedding-3-large, 400-token recursive chunks scored 89.5 with no overlap and 88.1 with 200 tokens of overlap, while with the small all-MiniLM model the token splitter at 250 tokens scored 82.4 with overlap and 77.1 without. Those are large general corpora, not this knowledge base. The difference from my version: theirs is a character or token window that can start mid-sentence, mine never starts mid-sentence, stays inside one section and is capped so a part always adds new text.

### The context prefix (design question 1)
`Chunk.embed_text(with_context=True)` returns `Document: <title>`, `Section: <heading>`, a blank line and then the text; `with_context=False` returns just the text. The stored and cited `text` never includes the prefix. The prefix adds 63 to 90 characters in this knowledge base (measured with a one-off check, not in the repo). My expectation, **not yet tested**, is that it helps because a chunk such as "Items within 30 days of expiry go to the quarantine bin" does not name its document, and the two leaflets have similar sections and wording. The title and heading may also pull queries toward a chunk on the title's words alone. The default is `True` only so there is something to compare: Step 4 should embed both versions and compare retrieval on the test set.

### Ids (design question 2)
- **Scheme:** `<file>.md#<Fragment>::<n>`, with `n` counting from 1 within the section, for example `returns_and_expiry_policy.md#Customer-Returns::2`. The part before `::` is the section id used by the test set, so an unsplit section is `...::1`.
- **Why unique:** Step 4's vector database uses the id as the key, so two chunks with one id would collide. File name, fragment and part number cannot repeat; the chunker raises an error if two headings in one file give the same fragment.
- **Why stable:** the id comes only from the file name, the heading and the position, never from a random value, a timestamp or the content. Re-running ingestion gives the same ids, so it can replace chunks instead of duplicating them, and a retrieval hit can be compared with `expected_sources` run after run.
- **If a document is edited:**
  - Text changed, heading unchanged: the ids stay the same. `Chunk.text_hash` changes, so ingestion can re-embed only chunks whose hash changed.
  - A section gets more or fewer parts: ids appear or disappear. Ingestion must delete stored ids that the new run does not produce, or a stale `::3` keeps its old text and can still be retrieved.
  - Text inserted early in a long section: part boundaries move, so every part of that section gets new text and a new hash, with the same ids.
  - A heading or file is renamed: the section id changes. The old chunks must be deleted, and `check_kb.py` fails for any test-set entry that still cites the old name, which is the intended loud failure.
  - Sections reordered: no effect, because ids do not depend on order.
  - Not implemented: storing `text_hash` in the database and the delete-stale step. They belong to Step 4.

## Experiment: `max_chars`

`py scripts/chunk_stats.py --max-chars N --overlap M` on the 13 sections of the real knowledge base. The section texts are 518 to 821 characters, median 614 [Measured]. Chunk sizes are the size of `Chunk.text`. "Extra chunks" is chunks minus 13 sections [Derived].

| `max_chars` | `overlap` | Chunks | Extra chunks vs. one per section | Min | Median | Max | Facts found in a chunk of their section (a) | Question/section pairs with all facts in one chunk (b) |
|---|---|---|---|---|---|---|---|---|
| 400 | 0 | 27 | 14 | 132 | 319 | 399 | all | 9 of 12 |
| 800 | 0 | 14 | 1 | 132 | 611 | 756 | all | 12 of 12 |
| 1500 | 0 | 13 | 0 | 518 | 614 | 821 | all | 12 of 12 |
| 400 | 100 | 28 | 15 | 163 | 323.5 | 399 | all | 10 of 12 |

Check (a): every `must_contain` string is found in a chunk of its cited section. Check (b): all of a question's strings sit in one chunk, counted per question and cited section (12 pairs from 11 questions, because Q18 cites two sections).

What the table shows:
- **[Measured]** At 1500 nothing splits, so chunks are exactly the sections. At 800 only the 821-character Customer Returns section splits, into 687 and 132 characters; the 132-character part is its closing paragraph on declined returns, a small orphan chunk. At 400 every section splits.
- **[Measured]** Check (a) passes in all four runs, so no setting lost a fact.
- **[Measured]** At 400, three pairs have their strings in different chunks: Q04 (`Cold-Chain-And-Excursions`), Q07 (`Warnings-And-Precautions`) and Q18 (`Receiving-And-FEFO`). With 100 characters of overlap, Q18 is healed because the carried sentence brings `FEFO` and the range into one chunk, and Q04 and Q07 are not. Overlap added one chunk (28 against 27).
- **[Derived]** Settings 800 and 1500 give the same result on both checks (12 of 12), so this test cannot tell them apart.

Limits: the knowledge base is tiny (13 sections), hand-written, and was written after the questions were seen, so every fact sits in its own short section. The setting only matters for the longest section, and these checks only show that facts are present, not that a retriever finds the right chunk. This is a check that the chunker works, not a measurement of retrieval quality.

### Where to start Step 4
Start with `max_chars=800` and `overlap=0`: it keeps every question's facts in one chunk (12 of 12) while still exercising the splitter on the one section that is longer than the limit, and real SOP and leaflet sections will be longer than these hand-written ones, so a limit that is not "never split" keeps chunks specific. This is a starting point chosen on a 13-section, hand-written knowledge base where 800 and 1500 behave identically on every check, so revisit it when Step 4 measures real retrieval, along with the context prefix and overlap.