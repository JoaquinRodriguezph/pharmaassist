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

**Smaller choices.** Stored ids are read page by page, because I did not want to rely on `get` returning everything by default. `k` larger than the collection returns everything. A missing store is reported and not created, because `PersistentClient` would otherwise create an empty one. The collection name is `pharmaassist_kb`. `retrieve` opens the store on every call, which is simple, and the timing in `try_retrieval.py` includes it.

## Checked and not checked

- **Stand-in only:** all 26 tests in `tests/test_store.py` and `tests/test_embedding.py` pass against my stand-in for Chroma. They cover the counts, edits, deletions, the prefix change, the stale-part deletion, the guard and the trap test. I also removed the cosine configuration from a copy of `store.py` and confirmed that the trap test fails, so the test does catch a collection left on the default distance. That was against the stand-in's idea of the default.
- **Not checked against real Chroma:** that `embedding_function=None` is accepted; that your Chroma version takes `configuration=`; the `limit` and `offset` arguments of `get`; the type of returned embeddings (numpy arrays in some versions); booleans in collection metadata; and that a missing collection raises something `store.py` treats as "not found". These need your `py -m pytest` output.
- **Pending:** the cross-check against the Step 4a table (see below).

## Cross-check with the real model

**Pending.** Run `scripts/ingest_kb.py`, then `scripts/try_retrieval.py`. The second compares the top-3 for Q01, Q06 and Q08 with the Step 4a table (same chunk, score within 0.001) and prints the retrieval time. To be filled in from the output.