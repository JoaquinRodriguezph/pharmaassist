# Vector store comparison: Chroma and FAISS (Step 4d)

Why I did this: I used Chroma for the pipeline and wanted to know what a plain FAISS index would change, on the same chunks and the same embedded text. Labels: **Docs** = stated on the page named in "Sources" (read on 20 October 2026). **Measured** = printed by a command run on this machine. **Prediction** = written before the run. **Untested** = a guess I have not run. **Pending** = a command exists for it, and I have not pasted its output yet.

The code is `app/rag/faiss_index.py` (index, save, load), `app/rag/rank_compare.py` (the tie rule), `tests/test_faiss_index.py`, `tests/test_rank_compare.py` and `scripts/compare_faiss.py`. I installed FAISS with `pip install faiss-cpu`, version 1.15.1 [Measured].

`compare_faiss.py` uses private helpers on purpose: `store._open_for_reading` and `store._check_embedder` for Chroma, and `index._index` for the bare FAISS timing row. They are fine in a script and are not part of any interface.

## What the FAISS docs say

| Question | Answer | Source |
|---|---|---|
| What does `IndexFlatIP` do? | Exact, exhaustive search by inner product: "all the indexed vectors are decoded sequentially and compared to the query vectors". The table row says "also for cosine (normalize vectors beforehand)". It uses `4*d` bytes per vector | Faiss indexes |
| dtype and shape | Vectors are rows of a `float32` matrix: "The data type `dtype` must be `float32`", row-major, `nb`-by-`d` for the database and `nq`-by-`d` for queries | Getting started |
| Cosine similarity | Inner product "is not by itself cosine similarity, unless the vectors are normalized". Normalize the database vectors before adding and the queries before searching, with `faiss.normalize_L2` in Python | MetricType and distances |
| What `search` returns | `D, I = index.search(xq, k)`: two `nq`-by-`k` matrices, the scores and the integer ids (positions). For the L2 index on that page, rows are "sorted by increasing distance". The page I read for inner product says only that one searches for "a *larger* similarity", not how rows are ordered | Getting started; MetricType and distances |
| Id when fewer than `k` results exist | "If there are not enough elements to fill the result list, the missing results are set to -1." That sentence is in the FAQ in the context of `IndexIVF` and `IndexHNSW`. **I found nothing about `IndexFlatIP`**, and nothing about what the score is for a missing result | FAQ |
| Ids on a flat index | Flat indexes "do not store vector ids, since in many cases sequential numbering is enough". No `add_with_id` (wrap in `IndexIDMap`). Removal with `remove` "shrinks the index and changes the numbering" | Faiss indexes |
| Save and load | `write_index(index, "large.index")` and `read_index("large.index")`. GPU indexes cannot be written. "It is the user's responsibility to verify that a file ... was not altered before reading" | Index IO |
| Threads | Concurrent searches are supported on CPU; concurrent search and add, or add and add, are not, and there is no locking | FAQ |

**What I did about the gaps.** Where the docs do not say, the code does not rely on a guess. `search_vector` limits `k` to the number of vectors, so no missing result should occur, and it also drops any id below zero if one appears. Either guard alone should be enough, and the test for `k` larger than the vector count will show it on the real library. Whether `IndexFlatIP` returns rows in decreasing similarity is checked by the ranking test against a brute-force ranking on the real library. I have not run that yet (see "Results"). This one-line check shows what a real flat index returns when `k` exceeds the vector count:

```
py -c "import faiss, numpy as np; i = faiss.IndexFlatIP(4); i.add(np.eye(4, dtype='float32')[:2]); print(i.search(np.ones((1, 4), dtype='float32'), 5))"
```

**Pending:** the output of that command.

## Keeping vectors and chunk records together

FAISS knows only positions 0, 1, 2... Everything else (chunk id, text, section, source) is in a list I keep, and position `i` of the index is `chunks[i]`. Chroma has no such problem: the chunk id is the key to its vector and its text and metadata, so a record cannot drift away from its vector.

**Where the mapping can break**

- The list is built in a different order from the order the vectors were added.
- Vectors are added or removed without the list changing the same way. FAISS renumbers after a removal.
- On disk, `index.faiss` and `records.json` come from different builds, for example after an interrupted save, a copied file, or a restored backup.
- Someone edits or reorders `records.json`. The counts still match.
- The index file is changed or truncated. The docs say they do not check it on load.

**What I did.** `FaissChunkIndex` never exposes `add` or `remove`: changing the knowledge base means building a new index. `save` writes both files under temporary names and renames them, the index first. `records.json` stores the SHA-256 of the index file, the vector count, the dimension, the embedder name and the prefix setting. Each record carries a fingerprint, and `load` checks all of it, so these cases are caught:

| What went wrong | How it is caught |
|---|---|
| A file is missing | `IndexNotFoundError` naming it |
| `index.faiss` from another build, or changed | SHA-256 stored in `records.json` differs |
| Crash between the two renames (new index, old records) | The same SHA-256 mismatch |
| Records dropped or added | Vector count differs from the number of records |
| Records reordered, even if their position numbers are tidied afterwards | The fingerprint of each record includes its position, so it stops matching |
| A record's text, section, source, part, `n_parts` or document title edited | Its fingerprint stops matching; `n_parts` and the title are covered even with the prefix off |
| A different embedder, or a different prefix setting | Embedder name or dimension differs; the prefix is part of every fingerprint |

**Compared with the Chroma fingerprint.** The shared fields are embedder, dimension, prefix setting, embedded text, text, source, section, section id and part. Two differences. First, the FAISS fingerprint also covers `n_parts` and the document title, because `records.json` stores them. The Chroma fingerprint covers the title only through the embedded text (so only with the prefix on) and does not cover `n_parts`. Chroma does not store either field, so there is nothing stale to catch there, but its fingerprint would not notice a title change with the prefix off. Second, and the main one, the **position**. In Chroma the fingerprint answers "does this id need to be embedded again?" and the id is its key. Here the fingerprint also has to answer "is this record still at the position of its own vector?", so it includes the position. I added a file hash because FAISS leaves the check on the index file to me, and Chroma keeps vectors and records in one database.

**What I did not add, and what it would miss.** A check that re-embeds the records and compares the results with the stored vectors. It would catch two files that are consistent with each other and wrong together (for example both written from a build with a different embedder version under the same name). That is the same limit the Chroma fingerprint has ("a model whose weights change under the same name"), and the fix is the same: build again. FAISS can return stored vectors with `reconstruct_n`, which is the likely way to do the re-embedding check, but that function is not on the pages I read, so I did not use it. **Untested.**

## Pre-registration (written before the first run)

I commit this section before running `compare_faiss.py` for the first time, so the git history shows the rules came first.

**Route: re-embed, then check against Chroma.** FAISS is built from the same chunks (`chunk_kb`, 800 characters, no overlap) and the same embedded text (`Chunk.embed_text` with the store's prefix setting). The script embeds the chunk texts once, compares those vectors with the ones Chroma stored (`collection.get(include=["embeddings"])`), and builds FAISS from exactly the vectors it checked. Re-embedding adds a second possible source of difference, the embedding step, alongside the index itself. The vector check measures that source directly. `--from-chroma-vectors` builds FAISS from Chroma's stored vectors instead, which removes the embedding step from the comparison. I use it if any rank differs, to tell an index difference from an embedding difference.

**Same query vector.** Each question is embedded once, and that vector goes to both backends (`collection.query(query_embeddings=...)` and `search_vector`), so a difference cannot come from embedding the query twice.

**What the script prints for the vectors.** For Chroma's stored vectors, the re-embedded vectors and the normalized vectors FAISS holds: count, dtype, shape, smallest and largest norm. Then the largest absolute difference between stored and re-embedded vectors after scaling both to length 1, and the chunk where it occurs. The re-embedded dtype will show float64: the embedder returns Python lists (`encode(...).tolist()`), which numpy reads as float64. That is a property of the list, not of the model.

**Questions.** 33 per setting, as the loaders return them: original 14 (11 with an expected section, plus Q09, Q10 and Q20 reported by score only), variants 11, independent 8 (2 with an expected section). Hit@k and MRR use the 24 with an expected section. The script prints these counts at the top; if they differ from these, its numbers are the ones to use.

**Rules, applied as written in `app/rag/rank_compare.py`:**

1. **Identical:** same chunk ids, same order, in the top 5.
2. **Tie:** at the first rank where the ids differ, call Chroma's chunk X and FAISS's chunk Y. It is a tie only if X and Y are in both top-5 lists and, in each backend, the scores of X and Y are within 1e-5 of each other. Otherwise it is **not a tie**, including when X or Y is missing from the other list.
3. **Score agreement:** for chunks at the same rank in both lists, scores within 1e-5.
4. Any difference that is not a tie, or scores outside 1e-5, I explain before writing any conclusion in this file. A tie is reported as a tie and is not counted against either backend.

The tolerance, 1e-5, is about float32 rounding on cosine scores, and is the tolerance the store tests already use. I chose it before seeing any output.

**Predictions.**

- The stored and re-embedded vectors differ by less than 1e-6 after normalizing (CPU embedding should be deterministic, but this is the check that shows it).
- All norms are within 1e-5 of 1 for the FAISS vectors. Chroma's stored norms are about 1 too, if the model normalizes its output; I have not checked that.
- In both settings, all 33 questions have identical top-5 ids, with no ties, and the largest score difference is below 1e-5.
- Hit@k and MRR are therefore identical for the two backends.
- Median search time with 14 vectors is well under 1 ms for every timing row. I do not predict a ratio, because the rows do different work (below).

## Results of `scripts/compare_faiss.py`

**Measured: Pending.** Run `py scripts\compare_faiss.py` with the `ctx` and `noctx` stores built, and paste the output here. To fill in from that output: the route (re-embedded, unless I had to use `--from-chroma-vectors`), the largest vector difference for each setting, the norm and dtype lines, how many questions were identical, any ties or other differences, the largest score difference, the hit@k and MRR table, and the timing rows.

**Reading the timing rows.** The script times each setting, on queries embedded beforehand, so the numbers are the search alone. The three rows do different work:

- **Chroma query** searches and returns documents, metadata and distances.
- **FAISS search_vector** copies and normalizes the query, searches, and builds hit objects.
- **FAISS bare search** searches a query normalized beforehand.

A ratio between rows is not a finding that one backend is faster. Embedding a query is much slower than any of them: `retrieve()` took 33 to 52 ms in the Step 4b cross-check, including embedding and opening the store [Measured]. With 14 vectors, none of this says anything about scale.

**At 100,000 vectors, all Untested.**

- A flat index compares the query with every vector, so its time should grow in proportion to the count: 100,000 vectors of 384 numbers is about 154 MB [Derived from `4*d` bytes per vector] and about 38 million multiplications per query. I guess a few milliseconds to a few tens of milliseconds on a CPU.
- Chroma's HNSW search should grow much more slowly with the count, but it is approximate, so some true neighbors may be missed. Building the index takes longer and uses more memory.
- Query embedding (tens of milliseconds) may still dominate at that size.

## Comparison

| | FAISS (`IndexFlatIP`) | Chroma |
|---|---|---|
| Install | **Docs:** "The supported way to install Faiss is through conda" (`faiss-cpu=1.15.1`); that page does not mention pip. **Measured:** installed with `pip install faiss-cpu`, version 1.15.1 | **Measured:** `chromadb` 1.5.9 installed in the venv and used by the whole pipeline |
| Persistence | **Docs:** `write_index` / `read_index`; files are not checked on load. **Pending:** the round-trip tests and the reload inside `compare_faiss.py` | **Docs:** `PersistentClient(path=...)`. **Measured:** the store tests pass on this machine; the current total is the one `py -m pytest` prints, **Pending** for this step |
| Storing text and metadata | **Docs:** vectors only; flat indexes do not even store ids. The text, metadata and the position link are my `records.json` | **Docs:** documents, metadatas (string, integer, float, boolean) and ids are stored with the embeddings. **Measured:** `store.py` uses all three |
| Updating and deleting | **Docs:** flat index supports `remove`, which "changes the numbering"; no `add_with_id` without `IndexIDMap`. My module does neither: it rebuilds | **Docs:** `upsert` (creates or updates by id), `delete(ids=...)`. **Measured:** `ingest_kb` adds, updates and deletes chunks by fingerprint, tested |
| Metadata filtering | **Docs:** the FAQ says filtering must be based on vector ids; there is no metadata in the index. I would filter the results in Python. **Untested** | **Docs:** `where` on `get` and `query` (`$eq`, `$gt`, `$gte`, `$lte`, `$and`, `$or`, `$in`, `$nin`, `$contains`). The page does not say whether the filter is applied before or after the nearest-neighbor search. **Untested** by me |
| Exact or approximate search | **Docs:** exact; every vector is compared | **Docs:** approximate (HNSW, `ef_search` default 100). **Pending:** whether it matches the exact result on my 33 questions per setting (14 vectors) |
| What a search returns | **Docs:** scores and integer positions. Cosine distance is my `1 - score`; ordering for inner product is checked by my test, not stated in the pages I read | **Docs:** a distance; cosine distance is `1 - cosine similarity`, so `score = 1 - distance`. **Measured:** the scores matched my own cosine to 0.00001 |
| Index types for scale | **Docs:** IVF partitions the vectors into `nlist` cells; "`IndexIVFPQ` is probably the most useful indexing structure for large-scale search"; `IndexHNSW` also exists (FAQ). **Untested** by me | **Docs:** HNSW on a single node (`space`, `ef_construction`, `ef_search`, `max_neighbors`); a different index (SPANN) in the distributed and cloud versions that cannot be configured. **Untested** beyond cosine space |
| Threads | **Docs:** concurrent CPU searches are fine; search during add is not, and there is no locking | Not read |

## Decision

**The service uses Chroma.**

- It keeps vectors, text and metadata together under one id, so the position-to-chunk problem above does not exist. My FAISS version needs a file hash and position-aware fingerprints to prevent it, and it still cannot be updated in place.
- I already have incremental ingest, deletes and the model-mixing guard working and tested on Chroma, and they would need to be rebuilt for FAISS.
- At 14 chunks, or a few thousand, exact versus approximate search and search speed are not a reason to choose: embedding the query is far slower than either.

What would change my mind: a knowledge base of hundreds of thousands of chunks or more, where flat search becomes slow and I would want to choose the index type (IVF or HNSW with quantization); Chroma's approximate search missing a chunk that exact search finds (the compare script checks this on my questions, at 14 vectors); or needing a feature FAISS has and Chroma does not.

**What I can claim afterwards.** I used Chroma in the pipeline (real chromadb 1.5.9, with the test totals that `py -m pytest` prints), built a FAISS index from the same chunks and text, and compared the two on a 14-chunk knowledge base with 33 questions per setting [after the run]. I have no experience with large-scale search. Everything I wrote about 100,000 vectors is a guess from the documentation.

## Sources

- [FAISS: Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS: Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS: MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [FAISS: Index IO, cloning and hyper parameter tuning](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning)
- [FAISS: FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)
- [FAISS: INSTALL.md](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
- [Chroma: metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [Chroma: configure collections](https://docs.trychroma.com/docs/collections/configure)
- Chroma persistence, updates and deletes, and distance: `docs/store_notes.md`