"""A persistent Chroma store for knowledge-base chunks, with ingestion that is safe to re-run.

**Distance and score.** Chroma's default distance is squared L2, so every
collection here is created with cosine space. For cosine space Chroma's
`distance` is 1 - cosine similarity, so `score = 1 - distance`; a higher score is
better, and `score` matches `app.rag.similarity.cosine_similarity` up to the
float32 precision of the index.

**Embeddings are always passed in.** Nothing here ever lets Chroma embed text
itself (its default embedding function would silently run its own copy of a
model), so the collection is created with `embedding_function=None`.

**Fingerprint.** Each stored chunk carries a fingerprint of everything that
determines the record: the embedder's name and dimension, the prefix setting,
the exact text that was embedded, the stored text and the stored metadata. An
unchanged fingerprint means the stored record is exactly what a new run would
write; any change (an edit, a different model, a different prefix setting)
makes the chunk "updated" and it is embedded again. It cannot see a model whose
weights changed under the same name; use `rebuild=True` after such a change.

**Mixing models.** The collection records the embedder's name, dimension and
the prefix setting. `retrieve` and `ingest_kb` refuse an embedder whose name or
dimension differs (vectors from different models are not comparable).
`ingest_kb(rebuild=True)` deletes the collection first, to switch models.

**Re-running.** `ingest_kb` upserts new and changed chunks first and deletes
chunks that the new run no longer produces last, so an interrupted run leaves
extra stale chunks, never missing ones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.rag.chunking import Chunk, chunk_kb
from app.rag.embedding import Embedder

COLLECTION_NAME = "pharmaassist_kb"
_BATCH = 100


class StoreError(Exception):
    """Base class for store errors."""


class StoreNotFoundError(StoreError):
    """There is no store, or no collection, to read from."""


class EmbedderMismatchError(StoreError):
    """The embedder differs from the one the store was built with."""


@dataclass(frozen=True)
class IngestCounts:
    added: int
    updated: int
    unchanged: int
    deleted: int


@dataclass(frozen=True)
class Retrieved:
    id: str
    text: str
    section_id: str
    section: str
    source: str
    score: float  # cosine similarity, higher is better: 1 - distance
    distance: float  # Chroma's cosine distance, lower is better


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client(db_path: Path):
    return chromadb.PersistentClient(path=str(db_path))


def _existing_collection(client):
    """The collection, or None if it does not exist.

    Chroma raises different exception classes across versions for a missing
    collection, so any lookup failure counts as "not found" here; a real problem
    surfaces on the next call.
    """
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=None)
    except Exception:
        return None


def _collection_metadata(embedder: Embedder, with_context: bool) -> dict:
    return {
        "embedder_name": embedder.name,
        "embedder_dimension": embedder.dimension,
        "with_context": with_context,
    }


def _check_embedder(collection, embedder: Embedder) -> None:
    meta = collection.metadata or {}
    stored = (meta.get("embedder_name"), meta.get("embedder_dimension"))
    if stored != (embedder.name, embedder.dimension):
        if stored == (None, None):
            was = "no embedder recorded"
        else:
            was = f"embedder {stored[0]!r} (dimension {stored[1]})"
        raise EmbedderMismatchError(
            f"This store has {was}, but the embedder in use is {embedder.name!r} "
            f"(dimension {embedder.dimension}). Vectors from different models are not "
            "comparable. Use the original embedder, or re-ingest with rebuild=True to switch models."
        )


def _check_vectors(vectors: list[list[float]], expected_count: int, embedder: Embedder) -> None:
    if len(vectors) != expected_count:
        raise ValueError(f"The embedder returned {len(vectors)} vectors for {expected_count} texts")
    for vector in vectors:
        if len(vector) != embedder.dimension:
            raise ValueError(
                f"The embedder {embedder.name!r} says its dimension is {embedder.dimension} "
                f"but returned a vector of length {len(vector)}"
            )


def _fingerprint(chunk: Chunk, embedder: Embedder, with_context: bool) -> str:
    record = {
        "embedder": embedder.name,
        "dimension": embedder.dimension,
        "with_context": with_context,
        "embedded_text": chunk.embed_text(with_context),
        "text": chunk.text,
        "source": chunk.source,
        "section": chunk.section,
        "section_id": chunk.section_id,
        "part": chunk.part,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _stored_fingerprints(collection, page_size: int = 500) -> dict[str, str | None]:
    """Every stored id with its fingerprint, read page by page."""
    stored: dict[str, str | None] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=page_size, offset=offset)
        ids = page["ids"]
        if not ids:
            break
        for chunk_id, meta in zip(ids, page["metadatas"]):
            stored[chunk_id] = (meta or {}).get("fingerprint")
        offset += len(ids)
        if len(ids) < page_size:
            break
    return stored


# ---------------------------------------------------------------------------
# Ingest and retrieve
# ---------------------------------------------------------------------------

def ingest_kb(
    kb_dir: Path,
    db_path: Path,
    embedder: Embedder,
    max_chars: int = 800,
    overlap: int = 0,
    with_context: bool = True,
    rebuild: bool = False,
) -> IngestCounts:
    """Chunk the knowledge base and bring the store in line with it; safe to run again.

    New chunks are added, chunks whose fingerprint changed are updated (and
    embedded again), unchanged chunks are left alone and not embedded again, and
    stored chunks the new run does not produce are deleted.
    """
    chunks = chunk_kb(kb_dir, max_chars, overlap)
    client = _client(db_path)
    collection = _existing_collection(client)
    if collection is not None and rebuild:
        client.delete_collection(COLLECTION_NAME)
        collection = None

    metadata = _collection_metadata(embedder, with_context)
    if collection is None:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=metadata,
            embedding_function=None,
        )
    else:
        _check_embedder(collection, embedder)
        if (collection.metadata or {}).get("with_context") != with_context:
            collection.modify(metadata=metadata)

    stored = _stored_fingerprints(collection)
    fingerprints = {chunk.id: _fingerprint(chunk, embedder, with_context) for chunk in chunks}
    added = [c for c in chunks if c.id not in stored]
    updated = [c for c in chunks if c.id in stored and stored[c.id] != fingerprints[c.id]]
    stale = sorted(set(stored) - set(fingerprints))

    to_write = added + updated
    for start in range(0, len(to_write), _BATCH):
        batch = to_write[start : start + _BATCH]
        vectors = embedder.embed([c.embed_text(with_context) for c in batch])
        _check_vectors(vectors, len(batch), embedder)
        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "source": c.source,
                    "section": c.section,
                    "section_id": c.section_id,
                    "part": c.part,
                    "fingerprint": fingerprints[c.id],
                }
                for c in batch
            ],
        )
    for start in range(0, len(stale), _BATCH):
        collection.delete(ids=stale[start : start + _BATCH])

    return IngestCounts(
        added=len(added),
        updated=len(updated),
        unchanged=len(chunks) - len(added) - len(updated),
        deleted=len(stale),
    )


def _open_for_reading(db_path: Path):
    if not Path(db_path).exists():
        raise StoreNotFoundError(f"No store at {db_path}. Run scripts/ingest_kb.py first.")
    collection = _existing_collection(_client(db_path))
    if collection is None:
        raise StoreNotFoundError(f"The store at {db_path} has no '{COLLECTION_NAME}' collection. Run scripts/ingest_kb.py first.")
    return collection


def store_info(db_path: Path) -> dict:
    """What the store was built with: embedder name, dimension, prefix setting and chunk count."""
    collection = _open_for_reading(db_path)
    meta = collection.metadata or {}
    return {
        "embedder_name": meta.get("embedder_name"),
        "embedder_dimension": meta.get("embedder_dimension"),
        "with_context": meta.get("with_context"),
        "count": collection.count(),
    }


def retrieve(db_path: Path, embedder: Embedder, query: str, k: int = 3) -> list[Retrieved]:
    """The `k` chunks nearest to `query`, best first, with cosine-similarity scores.

    Raises EmbedderMismatchError if `embedder` is not the one the store was built with.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        raise ValueError("query must not be empty")
    collection = _open_for_reading(db_path)
    _check_embedder(collection, embedder)
    count = collection.count()
    if count == 0:
        return []

    vector = embedder.embed([query])
    _check_vectors(vector, 1, embedder)
    result = collection.query(
        query_embeddings=vector,
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        retrieved.append(
            Retrieved(
                id=chunk_id,
                text=text,
                section_id=meta["section_id"],
                section=meta["section"],
                source=meta["source"],
                score=1.0 - distance,
                distance=distance,
            )
        )
    return retrieved