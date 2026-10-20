"""A FAISS index over the same chunks and the same embedded text as the Chroma store.

What FAISS gives and what it does not (docs: see docs/vector_store_comparison.md):

- `IndexFlatIP` compares a query with every stored vector by inner product (exact
  search). With vectors scaled to length 1, inner product equals cosine
  similarity, so vectors are normalized with `faiss.normalize_L2` before they are
  added and before they are searched.
- It stores only vectors, numbered 0, 1, 2... in the order they were added. It
  has no ids, no text and no metadata.

**How the link is kept.** This class holds a list of chunks next to the index, and
position `i` of the index is `chunks[i]`. That is the whole mapping. It breaks if
the order of the list differs from the order the vectors were added, if vectors
are added or removed (FAISS renumbers after a removal) without the list changing
the same way, or, on disk, if the two files come from different builds. So:

- The class never exposes `add` or `remove`. Changing the knowledge base means
  building a new index.
- `save` writes `index.faiss` (vectors) and `records.json` (chunk records) into a
  folder. `records.json` stores the index file's SHA-256, the vector count, the
  dimension, the embedder and the prefix setting, and a fingerprint per record
  that includes the record's position. `load` checks all of them and raises
  `IndexOutOfSyncError` on any mismatch.

Importing this module does not import faiss; building, loading or searching an
index does.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.chunking import Chunk
from app.rag.embedding import Embedder

INDEX_FILE = "index.faiss"
RECORDS_FILE = "records.json"
FORMAT_VERSION = 1


class FaissIndexError(Exception):
    """Base class for this module's errors."""


class IndexNotFoundError(FaissIndexError):
    """A saved index (or one of its two files) is missing."""


class IndexEmbedderMismatchError(FaissIndexError):
    """The embedder in use is not the one the index was built with."""


class IndexOutOfSyncError(FaissIndexError):
    """The two saved files do not belong together, or a file was changed."""


@dataclass(frozen=True)
class FaissHit:
    """One search result. Same fields as `app.rag.store.Retrieved`.

    `distance` is the cosine distance, `1 - score`, so that it is comparable with
    Chroma's. It is not a FAISS output: the index returns only the similarity.
    This class is separate from `Retrieved` so that this module does not import
    chromadb.
    """

    id: str
    text: str
    section_id: str
    section: str
    source: str
    score: float  # cosine similarity (float32 precision), higher is better
    distance: float  # 1 - score


def _faiss():
    try:
        import faiss
    except ImportError as error:
        raise ImportError(
            "faiss is not installed. Install the faiss-cpu package (see docs/vector_store_comparison.md)."
        ) from error
    return faiss


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def _matrix(vectors, dimension: int, what: str) -> np.ndarray:
    """A new C-contiguous float32 array of shape (n, dimension), checked.

    FAISS needs float32 and a C-contiguous 2-D array. A copy is always made, because
    `faiss.normalize_L2` changes its argument in place and must not touch the
    caller's data. Zero vectors are refused: their direction, and so their cosine,
    is undefined (the same rule as `similarity.cosine_similarity`).
    """
    array = np.array(vectors, dtype=np.float32, copy=True, order="C")
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"{what}: expected vectors of length {dimension}, got an array of shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{what}: vectors contain NaN or infinite values")
    if array.shape[0] and (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError(f"{what}: cosine similarity is undefined for a zero vector")
    return array


def _embed(embedder: Embedder, texts: list[str], what: str) -> np.ndarray:
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError(f"The embedder returned {len(vectors)} vectors for {len(texts)} texts")
    return _matrix(vectors, embedder.dimension, what) if texts else np.zeros((0, embedder.dimension), np.float32)


def vector_stats(vectors) -> dict:
    """Count, dtype, shape and the smallest and largest vector length (norm) of a set of vectors.

    Uses `len()` and shapes, never truthiness, because a numpy array has no single truth value.
    """
    array = np.asarray(vectors)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D set of vectors, got shape {array.shape}")
    norms = np.linalg.norm(array.astype(np.float64), axis=1) if len(array) else np.zeros(0)
    return {
        "count": int(len(array)),
        "dtype": str(array.dtype),
        "shape": tuple(int(n) for n in array.shape),
        "min_norm": float(norms.min()) if len(norms) else None,
        "max_norm": float(norms.max()) if len(norms) else None,
    }


def normalized_max_abs_difference(a, b) -> tuple[float, int]:
    """Largest absolute difference between two sets of vectors after scaling each vector to length 1.

    Returns (difference, row where it occurs). The scaling is done in float64 so that it adds
    no rounding of its own. Rows must correspond. A zero vector raises ValueError.
    """
    first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or len(first) == 0:
        raise ValueError(f"need two non-empty sets of the same shape, got {first.shape} and {second.shape}")
    norms = [np.linalg.norm(x, axis=1, keepdims=True) for x in (first, second)]
    if (norms[0] == 0).any() or (norms[1] == 0).any():
        raise ValueError("cannot normalize a zero vector")
    gaps = np.abs(first / norms[0] - second / norms[1]).max(axis=1)
    row = int(gaps.argmax())
    return float(gaps[row]), row


# ---------------------------------------------------------------------------
# Records and fingerprints
# ---------------------------------------------------------------------------

def _record(chunk: Chunk) -> dict:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "section": chunk.section,
        "section_id": chunk.section_id,
        "doc_title": chunk.doc_title,
        "part": chunk.part,
        "n_parts": chunk.n_parts,
    }


def _chunk(record: dict) -> Chunk:
    return Chunk(**{key: record[key] for key in (
        "id", "text", "source", "section", "section_id", "doc_title", "part", "n_parts")})


def _fingerprint(position: int, chunk: Chunk, embedder_name: str, dimension: int, with_context: bool) -> str:
    """Hash of everything that decides what this position's vector should be.

    It has the same fields as the Chroma fingerprint (`store._fingerprint`) plus the
    **position**. In Chroma the chunk id is the key to its vector, so a record cannot
    drift away from its vector. Here position is the only key, so a record moved to
    another position must stop matching. Every field kept in `records.json` is covered,
    including `doc_title` and `n_parts` even when the prefix is off (Chroma's
    fingerprint covers `doc_title` only through the embedded text, with the prefix on,
    and does not store `n_parts`).
    """
    record = {
        "position": position,
        "embedder": embedder_name,
        "dimension": dimension,
        "with_context": with_context,
        "embedded_text": chunk.embed_text(with_context),
        "text": chunk.text,
        "source": chunk.source,
        "section": chunk.section,
        "section_id": chunk.section_id,
        "part": chunk.part,
        "n_parts": chunk.n_parts,
        "doc_title": chunk.doc_title,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

class FaissChunkIndex:
    """An exact cosine-similarity index over chunks. Build it, or `load` a saved one."""

    def __init__(self, index, chunks: list[Chunk], embedder: Embedder, with_context: bool) -> None:
        # Internal: use `build` or `load`, which make sure index and chunks agree.
        if index.ntotal != len(chunks):
            raise IndexOutOfSyncError(f"the index has {index.ntotal} vectors but there are {len(chunks)} chunks")
        self._index = index
        self._chunks = list(chunks)
        self._embedder = embedder
        self.with_context = with_context
        self.build_stats: dict | None = None  # set by `build`: stats of the normalized vectors that were added

    # -- building ----------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, with_context: bool = False) -> "FaissChunkIndex":
        """Embed `chunk.embed_text(with_context)` for every chunk and index the normalized vectors.

        Position `i` of the index is `chunks[i]`. Chunk ids must be unique.
        """
        chunks = list(chunks)
        vectors = _embed(embedder, [c.embed_text(with_context) for c in chunks], "chunk vectors")
        return cls.from_vectors(chunks, vectors, embedder, with_context)

    @classmethod
    def from_vectors(
        cls, chunks: list[Chunk], vectors, embedder: Embedder, with_context: bool = False
    ) -> "FaissChunkIndex":
        """Index vectors that were embedded elsewhere: `vectors[i]` belongs to `chunks[i]`.

        The caller vouches that each vector is `embedder`'s embedding of
        `chunks[i].embed_text(with_context)`; only the count and length are checked.
        Used to build from exactly the vectors another check has looked at, for
        example the ones read back from Chroma.
        """
        chunks = list(chunks)
        ids = [c.id for c in chunks]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate chunk id(s): {duplicates}")
        if len(vectors) != len(chunks):
            raise ValueError(f"got {len(vectors)} vectors for {len(chunks)} chunks")
        faiss = _faiss()
        index = faiss.IndexFlatIP(embedder.dimension)
        stats = None
        if len(chunks):
            matrix = _matrix(vectors, embedder.dimension, "chunk vectors")
            faiss.normalize_L2(matrix)
            index.add(matrix)
            stats = vector_stats(matrix)
        built = cls(index, chunks, embedder, with_context)
        built.build_stats = stats
        return built

    # -- searching ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> list[str]:
        """Chunk ids in index order: `chunk_ids[i]` is the chunk at position `i`."""
        return [c.id for c in self._chunks]

    def info(self) -> dict:
        return {
            "embedder_name": self._embedder.name,
            "embedder_dimension": self._embedder.dimension,
            "with_context": self.with_context,
            "count": len(self._chunks),
        }

    def search_vector(self, vector, k: int = 5) -> list[FaissHit]:
        """The `k` chunks most similar to an already-embedded query, best first.

        `k` larger than the number of chunks gives all of them. Any position below
        zero in FAISS's answer (FAISS marks a missing result with -1, in the cases its
        FAQ describes) is dropped; with k limited to the index size none should appear.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if not self._chunks:
            return []
        query = _matrix([vector], self._embedder.dimension, "query vector")
        _faiss().normalize_L2(query)
        scores, positions = self._index.search(query, min(k, len(self._chunks)))
        hits = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            chunk = self._chunks[int(position)]
            similarity = float(score)
            hits.append(
                FaissHit(
                    id=chunk.id,
                    text=chunk.text,
                    section_id=chunk.section_id,
                    section=chunk.section,
                    source=chunk.source,
                    score=similarity,
                    distance=1.0 - similarity,
                )
            )
        return hits

    def search(self, query: str, k: int = 5) -> list[FaissHit]:
        """Embed `query` (no prefix, like the Chroma store) and return the top `k`."""
        if not query.strip():
            raise ValueError("query must not be empty")
        vectors = _embed(self._embedder, [query], "query vector")
        return self.search_vector(vectors[0], k)

    # -- saving and loading ------------------------------------------------

    def save(self, directory: Path) -> None:
        """Write `index.faiss` and `records.json` into `directory` (created if needed).

        Both are written under temporary names first and then renamed, the index
        first. A crash between the two renames leaves a new index beside an old
        records file, which `load` reports because the stored SHA-256 differs.
        """
        faiss = _faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_tmp = directory / (INDEX_FILE + ".tmp")
        records_tmp = directory / (RECORDS_FILE + ".tmp")

        faiss.write_index(self._index, str(index_tmp))
        name, dimension = self._embedder.name, self._embedder.dimension
        payload = {
            "format_version": FORMAT_VERSION,
            "embedder_name": name,
            "embedder_dimension": dimension,
            "with_context": self.with_context,
            "count": len(self._chunks),
            "index_file": INDEX_FILE,
            "index_sha256": _file_sha256(index_tmp),
            "records": [
                {**_record(chunk), "position": i,
                 "fingerprint": _fingerprint(i, chunk, name, dimension, self.with_context)}
                for i, chunk in enumerate(self._chunks)
            ],
        }
        records_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

        os.replace(index_tmp, directory / INDEX_FILE)
        os.replace(records_tmp, directory / RECORDS_FILE)

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "FaissChunkIndex":
        """Read a saved index and check that its two files belong together.

        Raises IndexNotFoundError (a file is missing), IndexEmbedderMismatchError
        (different embedder) or IndexOutOfSyncError (see the checks in the body).
        """
        directory = Path(directory)
        index_path, records_path = directory / INDEX_FILE, directory / RECORDS_FILE
        missing = [p.name for p in (index_path, records_path) if not p.is_file()]
        if missing:
            raise IndexNotFoundError(f"No saved index in {directory}: missing {', '.join(missing)}.")

        try:
            payload = json.loads(records_path.read_text(encoding="utf-8"))
            records = payload["records"]
            header = {key: payload[key] for key in (
                "format_version", "embedder_name", "embedder_dimension", "with_context", "count", "index_sha256")}
        except (ValueError, KeyError, TypeError) as error:
            raise IndexOutOfSyncError(f"{records_path.name} cannot be read as a saved index: {error!r}") from error
        if header["format_version"] != FORMAT_VERSION:
            raise IndexOutOfSyncError(f"{records_path.name} has format version {header['format_version']!r}; expected {FORMAT_VERSION}")

        if (header["embedder_name"], header["embedder_dimension"]) != (embedder.name, embedder.dimension):
            raise IndexEmbedderMismatchError(
                f"This index was built with embedder {header['embedder_name']!r} "
                f"(dimension {header['embedder_dimension']}), but the embedder in use is "
                f"{embedder.name!r} (dimension {embedder.dimension}). Build the index again."
            )

        if _file_sha256(index_path) != header["index_sha256"]:
            raise IndexOutOfSyncError(
                f"{index_path.name} does not match {records_path.name} (different SHA-256): "
                "the files come from different builds, or one was changed."
            )
        index = _faiss().read_index(str(index_path))

        problems = []
        if index.ntotal != header["count"] or header["count"] != len(records):
            problems.append(
                f"the index holds {index.ntotal} vectors, the header says {header['count']}, "
                f"and the records file lists {len(records)}"
            )
        if index.d != embedder.dimension:
            problems.append(f"the index has dimension {index.d}, not {embedder.dimension}")
        if problems:
            raise IndexOutOfSyncError("; ".join(problems))

        chunks = []
        for position, record in enumerate(records):
            try:
                chunk = _chunk(record)
                stored_position, stored_fingerprint = record["position"], record["fingerprint"]
            except (KeyError, TypeError) as error:
                raise IndexOutOfSyncError(f"record {position} is malformed: {error!r}") from error
            expected = _fingerprint(position, chunk, embedder.name, embedder.dimension, bool(header["with_context"]))
            if stored_position != position or stored_fingerprint != expected:
                raise IndexOutOfSyncError(
                    f"record {position} ({chunk.id}) does not match its stored fingerprint: "
                    "it was edited or moved to another position."
                )
            chunks.append(chunk)
        return cls(index, chunks, embedder, bool(header["with_context"]))