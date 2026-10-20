# Store notes (Step 4b)

The Chroma store in `app/rag/store.py`: what the Chroma documentation says, the decisions made, and what has and has not been checked. Labels: **Docs** = stated in the Chroma documentation (docs.trychroma.com and cookbook.chromadb.dev, read on 19 October 2026). **Stand-in** = checked only against a stand-in for Chroma that I wrote from those docs, which is not the real library. **Pending** = needs a run on this machine.

## What the Chroma docs say

| Question | Answer | Source |
|---|---|---|
| Persistent client | `chromadb.PersistentClient(path=...)` keeps the data on disk | Docs |
| Creating a collection | `create_collection` and `get_or_create_collection`. If the collection already exists, `get_or_create_collection` ignores the metadata passed to it, so `store.py` creates the collection explicitly and compares metadata itself | Docs (collection reference) |
| Default distance | `l2`, the squared L2 norm | Docs (collection configuration) |
| Setting cosine | At creation, `configuration={"hnsw": {"space": "cosine"}}` (Chroma 1.x). Older versions used `metadata={"hnsw:space": "cosine"}`. The space cannot be changed afterwards | Docs; cookbook for the "cannot change" part |
| What a query returns | A **distance**, not a similarity. For cosine space the documented distance is 1 - cosine similarity, so `score = 1 - distance` | Docs |
| Adding and updating | `add` needs `documents`, `embeddings` or both; with only documents, the collection's embedding function embeds them. `upsert` creates or updates by id | Docs |
| Metadata values | Strings, integers, floats and booleans | Docs |
| `get` and `query` | Results come as parallel lists. By default `get` returns metadatas and documents, and `query` returns metadatas, documents and distances; embeddings are left out unless `include` asks for them. A query vector of the wrong length raises an error | Docs |
| Deleting | `collection.delete(ids=[...])` | Docs |
| Default embedding function | When documents or query texts are given without embeddings, Chroma embeds them with its own default function, which is `all-MiniLM-L6-v2` run locally and downloaded automatically | Docs (embedding functions) |

**The silent-default trap.** Chroma's default function is the same model name as ours but a separate copy running through a different runtime. A call that forgot to pass embeddings would use it silently, and the scores would come from a different pipeline. So `store.py` passes `embeddings=` on every write and `query_embeddings=` on every read, and creates the collection with `embedding_function=None`. I assumed `None` switches the default off in the Python client; I did not see that stated for Python. If your Chroma rejects it, the first run will say so, and the fix is a one-line change.

## Decisions

**Score.** `score = 1 - distance`, with the collection created in cosine space. It equals `cosine_similarity(query_vector, stored_vector)` up to the float32 precision of the index; the test allows 0.00001.

**What the fingerprint covers.** The fingerprint is a hash of everything that determines a stored record: the embedder's name and dimension, the prefix setting, the exact text that was embedded (so a change to the document title or section heading counts when the prefix is on), the stored text, and the stored metadata (source, section, section id, part). The reasoning is that "unchanged" must mean "the stored record is exactly what a new run would write". If it left out the model, switching models would keep old vectors with a new name; if it left out the prefix setting, switching the setting would silently reuse vectors made the other way; if it left out the metadata, a renamed section would keep a stale label. It holds nothing random or time-based, so the same input always gives the same fingerprint.

**What it cannot see.** A model whose weights change under the same name. Use `rebuild=True` after such a change. The same goes for a change of library version that changes the vectors.

**Mixing models.** The collection records the embedder's name, dimension and the prefix setting. `retrieve` and `ingest_kb` refuse an embedder whose name or dimension differs, with an error that names both. The prefix setting is not part of the guard, because it describes how the stored chunks were embedded, not how queries are. Queries get no prefix. Changing it at ingest re-embeds every chunk and updates the record. `rebuild=True` is the way to switch models.

**Order of writes.** New and changed chunks are upserted first, and stale ids are deleted last. An interrupted run leaves some stale chunks and never missing ones, and running again finishes the job.

**Nothing is embedded twice.** Unchanged chunks are not sent to the embedder, so a second run with nothing changed does no embedding.

**What counts as "no collection".** Only a name that `list_collections()` does not list. The first version treated any error from `get_collection` as "not found", which would have read a corrupt database as an empty one and started a new store on top of it. Now any other error propagates. `list_collections()` returns `Collection` objects in Chroma 1.x and names in 0.6.x (Chroma's migration guide), so the code reads each entry's name with `getattr` and works with both. Tests cover a missing name, both return types, and errors while listing and while fetching a listed collection; the last test also checks that ingest and retrieve stop and create nothing.

**[Measured, chromadb 1.5.9]** `get_collection('nope')` raises `chromadb.errors.NotFoundError: Collection [nope] does not exist`, so that is the exception a missing collection gives; the store does not rely on it. `list_collections()` on an empty store returns a plain `list`. A non-empty store gives `[Collection(name=abc)]`: Collection objects, as the 1.x documentation says (the first command I gave you failed because it used a collection name that is too short; Chroma requires 3 to 512 characters).

**Smaller choices.** Stored ids are read page by page, because I did not want to rely on `get` returning everything by default. `k` larger than the collection returns everything. A missing store is reported and not created, because `PersistentClient` would otherwise create an empty one. The collection name is `pharmaassist_kb`. `retrieve` opens the store on every call, which is simple, and the timing in `try_retrieval.py` includes it.

## Known limits

- **Switching the prefix setting is not atomic.** `ingest_kb` updates the collection's recorded `with_context` (`collection.modify`) before it writes the new vectors. If a run is interrupted in between, the collection can say `with_context=True` while some stored vectors were made without the prefix (or the other way round). The next ingest repairs this, because the fingerprint of every affected chunk includes the prefix setting and so differs from the stored one, and those chunks are embedded again. Until then, retrieval from such a store mixes two kinds of vectors.
- **Interrupted runs leave stale chunks, never missing ones.** New and changed chunks are written first and stale ids are deleted last. Until the next run finishes the job, old chunks can still be retrieved.
- **The fingerprint cannot see changed model weights** under the same name, or a library change that alters the vectors. Use `rebuild=True`.
- **No protection against two ingests running at the same time.** Both would read the same stored state and could write conflicting results. Run one at a time.
- **Paging in `get` past one page** (500 chunks) has not been exercised; the store has 14 chunks.

## Checked and not checked

Real run on this machine: Python 3.12.10, chromadb 1.5.9, sentence-transformers 6.1.0.

- **[Measured] The tests pass against real Chroma.** `py -m pytest -v` gave 231 passed in 19.47 s, including all 26 store and embedding tests; after the not-found fix and the Step 4c code it gave 296 passed in 19.68 s, including the six new lookup tests, and 297 passed in 19.20 s once the independent-questions test was added. Of the assumptions listed before the run: `embedding_function=None` was accepted; `configuration={"hnsw": {"space": "cosine"}}` was accepted; `get` with `limit` and `offset` worked; returned embeddings could be compared with our own cosine function; a boolean in the collection metadata came back as a boolean (`with_context is False`); and on a fresh folder a missing collection was treated as "not found" and created. The trap test passing on real Chroma shows the collection really uses cosine space: the stored scores matched our own cosine to within 0.00001.
- **Not exercised:** paging in `get` beyond one page (the store has 14 chunks and the page size is 500), and a store with an existing folder but no collection.
- **Stand-in:** earlier, the same tests passed against my stand-in for Chroma, and the trap test failed on a copy without the cosine configuration. That showed the test could catch the mistake; the real library has now passed it.

## Cross-check with the real model

`py scripts\ingest_kb.py`, then `py scripts\try_retrieval.py`, with the prefix on (the default).

- **[Measured]** The first ingest added 14 chunks and took 0.80 s; the second run reported `unchanged=14` and took 0.16 s, because nothing was embedded again.
- **[Measured]** The top 3 for Q01, Q06 and Q08 match the Step 4a "with prefix" table in every position: 9 of 9, the same chunk and the same score to three decimals (for example Q08: 0.762, 0.693, 0.671, with Contraindications third). So the whole path agrees: chunking, the prefix setting, Chroma's cosine distance and `score = 1 - distance`.
- **[Measured]** `retrieve()` took 52, 36 and 37 ms on the first pass and 35, 34 and 33 ms on the second. That includes embedding the query and opening the store, on 14 chunks on this CPU.
- **Not yet done:** the same cross-check with `--no-context` against the "without prefix" table.
- **[Observed]** The first `py scripts\try_retrieval.py` in the pasted session printed nothing and the second worked. I do not know why; if it happens again, send me the output.