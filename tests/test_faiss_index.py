"""Tests for app.rag.faiss_index with a fake embedder, a temporary folder and no model download.

The fake embedder gives every text a fixed pseudo-random vector (seeded from the text's hash),
so two different texts never tie and the expected ranking can be worked out by hand with
`cosine_similarity`. Tests that need faiss are skipped if it is not installed.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.rag.chunking import Chunk
from app.rag.faiss_index import (
    INDEX_FILE,
    RECORDS_FILE,
    FaissChunkIndex,
    FaissHit,
    IndexEmbedderMismatchError,
    IndexNotFoundError,
    IndexOutOfSyncError,
    _matrix,
)
from app.rag.similarity import cosine_similarity

needs_faiss = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss is not installed")

REPO_ROOT = Path(__file__).resolve().parent.parent
DIMENSION = 16
FLOAT32_TOLERANCE = 1e-5

WORDS = ["fruit", "metal", "weather", "storage", "returns", "expiry", "dose", "label", "fridge", "shelf", "batch", "seal"]


def seeded_vector(text, dimension=DIMENSION):
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=dimension).tolist()


class FakeEmbedder:
    def __init__(self, name="fake-v1", dimension=DIMENSION):
        self.name = name
        self.dimension = dimension
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [seeded_vector(t, self.dimension) for t in texts]


def make_chunks(n=12):
    return [
        Chunk(
            id=f"doc{i % 3}.md#{WORDS[i].title()}::1",
            text=f"Rules about {WORDS[i]} number {i}: staff must follow step {i}.",
            source=f"doc{i % 3}.md",
            section=WORDS[i].title(),
            section_id=f"doc{i % 3}.md#{WORDS[i].title()}",
            doc_title=f"Document {i % 3}",
            part=1,
            n_parts=1,
        )
        for i in range(n)
    ]


QUERIES = ["how long is stock kept", "what temperature", "can I return opened items", "max dose", "who signs the log"]


def brute_force(chunks, embedder, query, with_context):
    """(chunk id, cosine similarity) for every chunk, best first, using our own cosine_similarity."""
    query_vector = embedder.embed([query])[0]
    scored = [
        (c.id, cosine_similarity(query_vector, embedder.embed([c.embed_text(with_context)])[0])) for c in chunks
    ]
    return sorted(scored, key=lambda pair: -pair[1])


# ---------------------------------------------------------------------------
# Ranking and scores
# ---------------------------------------------------------------------------

@needs_faiss
@pytest.mark.parametrize("with_context", [False, True])
def test_ranking_and_scores_equal_a_brute_force_ranking_by_cosine_similarity(with_context):
    chunks, embedder = make_chunks(), FakeEmbedder()
    index = FaissChunkIndex.build(chunks, embedder, with_context=with_context)

    for query in QUERIES:
        expected = brute_force(chunks, embedder, query, with_context)
        hits = index.search(query, k=len(chunks))

        assert [h.id for h in hits] == [cid for cid, _ in expected], query
        for hit, (_, score) in zip(hits, expected):
            assert hit.score == pytest.approx(score, abs=FLOAT32_TOLERANCE)


@needs_faiss
def test_top_k_is_the_first_k_of_the_full_ranking():
    chunks, embedder = make_chunks(), FakeEmbedder()
    index = FaissChunkIndex.build(chunks, embedder)

    top3 = index.search(QUERIES[0], k=3)
    everything = index.search(QUERIES[0], k=len(chunks))

    assert [h.id for h in top3] == [h.id for h in everything[:3]]
    assert [h.score for h in top3] == sorted((h.score for h in top3), reverse=True)


@needs_faiss
def test_a_hit_has_the_chunks_fields_and_a_cosine_distance():
    chunks, embedder = make_chunks(), FakeEmbedder()
    index = FaissChunkIndex.build(chunks, embedder)

    (hit,) = index.search("anything", k=1)
    chunk = next(c for c in chunks if c.id == hit.id)

    assert isinstance(hit, FaissHit)
    assert (hit.text, hit.section_id, hit.section, hit.source) == (chunk.text, chunk.section_id, chunk.section, chunk.source)
    assert hit.distance == pytest.approx(1.0 - hit.score)
    assert -1.0 - FLOAT32_TOLERANCE <= hit.score <= 1.0 + FLOAT32_TOLERANCE


@needs_faiss
def test_the_prefix_setting_decides_which_text_is_embedded():
    chunks = make_chunks(3)
    plain, prefixed = FakeEmbedder(), FakeEmbedder()

    FaissChunkIndex.build(chunks, plain, with_context=False)
    FaissChunkIndex.build(chunks, prefixed, with_context=True)

    assert plain.calls == [[c.text for c in chunks]]
    assert prefixed.calls == [[c.embed_text(True) for c in chunks]]
    assert prefixed.calls[0][0].startswith("Document: Document 0\nSection: ")


# ---------------------------------------------------------------------------
# Positions and chunk ids
# ---------------------------------------------------------------------------

@needs_faiss
def test_positions_map_back_to_the_right_chunk_ids():
    chunks, embedder = make_chunks(), FakeEmbedder()
    index = FaissChunkIndex.build(chunks, embedder)

    assert index.chunk_ids == [c.id for c in chunks]
    for chunk in chunks:
        # Searching with a chunk's own vector must return that chunk first, with similarity about 1.
        vector = embedder.embed([chunk.embed_text(False)])[0]
        (top,) = index.search_vector(vector, k=1)
        assert top.id == chunk.id
        assert top.score == pytest.approx(1.0, abs=FLOAT32_TOLERANCE)


@needs_faiss
def test_an_index_from_given_vectors_equals_one_that_embeds_them_itself():
    chunks, embedder = make_chunks(), FakeEmbedder()
    vectors = embedder.embed([c.embed_text(False) for c in chunks])

    given = FaissChunkIndex.from_vectors(chunks, vectors, embedder)
    built = FaissChunkIndex.build(chunks, embedder)

    for query in QUERIES:
        assert given.search(query, k=5) == built.search(query, k=5)


@needs_faiss
def test_from_vectors_needs_one_vector_per_chunk():
    chunks, embedder = make_chunks(3), FakeEmbedder()

    with pytest.raises(ValueError, match="got 2 vectors for 3 chunks"):
        FaissChunkIndex.from_vectors(chunks, embedder.embed(["a", "b"]), embedder)


@needs_faiss
def test_duplicate_chunk_ids_are_refused():
    chunks = make_chunks(3)
    chunks[2] = Chunk(**{**chunks[2].__dict__, "id": chunks[0].id})

    with pytest.raises(ValueError, match="duplicate chunk id"):
        FaissChunkIndex.build(chunks, FakeEmbedder())


# ---------------------------------------------------------------------------
# k larger than the index
# ---------------------------------------------------------------------------

@needs_faiss
def test_k_larger_than_the_number_of_chunks_returns_just_the_chunks():
    chunks, embedder = make_chunks(5), FakeEmbedder()
    index = FaissChunkIndex.build(chunks, embedder)

    hits = index.search("question", k=100)

    assert len(hits) == 5
    assert sorted(h.id for h in hits) == sorted(c.id for c in chunks)
    assert all(h.id in {c.id for c in chunks} for h in hits)  # no placeholder from a -1 position


@needs_faiss
def test_an_empty_index_returns_nothing():
    index = FaissChunkIndex.build([], FakeEmbedder())

    assert len(index) == 0
    assert index.search("question", k=3) == []


@needs_faiss
@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_an_error(k):
    index = FaissChunkIndex.build(make_chunks(3), FakeEmbedder())

    with pytest.raises(ValueError, match="k must be at least 1"):
        index.search("question", k=k)


@needs_faiss
def test_an_empty_query_is_an_error():
    index = FaissChunkIndex.build(make_chunks(3), FakeEmbedder())

    with pytest.raises(ValueError, match="query must not be empty"):
        index.search("   ")


# ---------------------------------------------------------------------------
# Vectors in
# ---------------------------------------------------------------------------

def test_vectors_become_a_float32_c_contiguous_copy():
    original = np.asfortranarray(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64))

    prepared = _matrix(original, 2, "test")

    assert prepared.dtype == np.float32 and prepared.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(prepared, original)


@needs_faiss
def test_searching_does_not_change_the_callers_query_vector():
    index = FaissChunkIndex.build(make_chunks(4), FakeEmbedder())
    vector = np.array(seeded_vector("q"), dtype=np.float32) * 7.0  # deliberately not length 1
    before = vector.copy()

    index.search_vector(vector, k=2)

    assert np.array_equal(vector, before)


@needs_faiss
def test_the_length_of_the_query_vector_does_not_change_the_ranking():
    chunks, embedder = make_chunks(), FakeEmbedder()
    index = FaissChunkIndex.build(chunks, embedder)
    vector = np.array(seeded_vector("q"))

    small = index.search_vector(vector * 0.01, k=5)
    large = index.search_vector(vector * 100.0, k=5)

    assert [h.id for h in small] == [h.id for h in large]
    assert [h.score for h in small] == pytest.approx([h.score for h in large], abs=FLOAT32_TOLERANCE)


class ZeroEmbedder(FakeEmbedder):
    def embed(self, texts):
        return [[0.0] * self.dimension for _ in texts]


class ShortEmbedder(FakeEmbedder):
    def embed(self, texts):
        return [seeded_vector(t)[:-1] for t in texts]


class MissingEmbedder(FakeEmbedder):
    def embed(self, texts):
        return [seeded_vector(t) for t in texts][:-1]


@needs_faiss
@pytest.mark.parametrize(
    "embedder, message",
    [
        pytest.param(ZeroEmbedder(), "zero vector", id="zero-vector"),
        pytest.param(ShortEmbedder(), "expected vectors of length", id="wrong-length"),
        pytest.param(MissingEmbedder(), "returned", id="wrong-count"),
    ],
)
def test_bad_embeddings_are_refused(embedder, message):
    with pytest.raises(ValueError, match=message):
        FaissChunkIndex.build(make_chunks(3), embedder)


def test_importing_the_module_does_not_import_faiss_and_using_it_says_how_to_install():
    code = (
        "import sys\n"
        "sys.modules['faiss'] = None\n"  # makes `import faiss` raise ImportError
        "from app.rag.faiss_index import FaissChunkIndex\n"
        "class E:\n"
        "    name = 'e'; dimension = 4\n"
        "    def embed(self, texts): return [[1.0, 0, 0, 0] for _ in texts]\n"
        "try:\n"
        "    FaissChunkIndex.build([], E())\n"
        "except ImportError as error:\n"
        "    print('IMPORT-ERROR', error)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "IMPORT-ERROR faiss is not installed" in result.stdout
    assert "faiss-cpu" in result.stdout


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------

def save_index(tmp_path, n=12, with_context=False, name="saved"):
    folder = tmp_path / name
    index = FaissChunkIndex.build(make_chunks(n), FakeEmbedder(), with_context=with_context)
    index.save(folder)
    return index, folder


@needs_faiss
@pytest.mark.parametrize("with_context", [False, True])
def test_a_save_and_load_round_trip_gives_identical_results(tmp_path, with_context):
    original, folder = save_index(tmp_path, with_context=with_context)

    loaded = FaissChunkIndex.load(folder, FakeEmbedder())

    assert loaded.info() == original.info() and loaded.chunk_ids == original.chunk_ids
    assert loaded.with_context is with_context
    for query in QUERIES:
        assert loaded.search(query, k=5) == original.search(query, k=5)  # same ids, text, scores and distances


@needs_faiss
def test_saving_writes_exactly_the_two_files(tmp_path):
    _, folder = save_index(tmp_path)

    assert sorted(p.name for p in folder.iterdir()) == sorted([INDEX_FILE, RECORDS_FILE])


@needs_faiss
def test_saving_again_replaces_the_files_and_loads_cleanly(tmp_path):
    _, folder = save_index(tmp_path, n=12)
    smaller = FaissChunkIndex.build(make_chunks(5), FakeEmbedder())

    smaller.save(folder)

    assert len(FaissChunkIndex.load(folder, FakeEmbedder())) == 5


@needs_faiss
def test_an_empty_index_survives_a_round_trip(tmp_path):
    folder = tmp_path / "empty"
    FaissChunkIndex.build([], FakeEmbedder()).save(folder)

    loaded = FaissChunkIndex.load(folder, FakeEmbedder())

    assert len(loaded) == 0 and loaded.search("question") == []


@needs_faiss
@pytest.mark.parametrize("missing", [INDEX_FILE, RECORDS_FILE])
def test_a_missing_file_is_reported_by_name(tmp_path, missing):
    _, folder = save_index(tmp_path)
    (folder / missing).unlink()

    with pytest.raises(IndexNotFoundError, match=missing.replace(".", r"\.")):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
def test_a_folder_with_no_index_is_not_found(tmp_path):
    with pytest.raises(IndexNotFoundError):
        FaissChunkIndex.load(tmp_path / "nothing", FakeEmbedder())


@needs_faiss
@pytest.mark.parametrize(
    "embedder",
    [FakeEmbedder(name="other-model"), FakeEmbedder(dimension=8)],
    ids=["other-name", "other-dimension"],
)
def test_loading_with_a_different_embedder_is_refused(tmp_path, embedder):
    _, folder = save_index(tmp_path)

    with pytest.raises(IndexEmbedderMismatchError, match="fake-v1"):
        FaissChunkIndex.load(folder, embedder)


# -- the files drifting apart ------------------------------------------------

def edit_records(folder, change):
    path = folder / RECORDS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


@needs_faiss
def test_records_in_a_different_order_are_detected_even_if_their_positions_are_renumbered(tmp_path):
    _, folder = save_index(tmp_path)

    def swap(payload):
        records = payload["records"]
        records[0], records[1] = records[1], records[0]
        for i, record in enumerate(records):  # tidy the position numbers, as a careless fix might
            record["position"] = i

    edit_records(folder, swap)

    with pytest.raises(IndexOutOfSyncError, match="record 0"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
def test_an_edited_record_text_is_detected(tmp_path):
    _, folder = save_index(tmp_path)
    edit_records(folder, lambda payload: payload["records"][3].update(text="changed text"))

    with pytest.raises(IndexOutOfSyncError, match="record 3"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
def test_a_missing_record_is_detected(tmp_path):
    _, folder = save_index(tmp_path)
    edit_records(folder, lambda payload: payload["records"].pop())

    with pytest.raises(IndexOutOfSyncError, match="12 vectors|holds 12"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
def test_an_index_file_from_another_build_is_detected_even_with_the_same_count(tmp_path):
    _, folder = save_index(tmp_path, n=12)
    other_chunks = [Chunk(**{**c.__dict__, "text": c.text + " (another build)"}) for c in make_chunks(12)]
    FaissChunkIndex.build(other_chunks, FakeEmbedder()).save(tmp_path / "other")

    shutil.copyfile(tmp_path / "other" / INDEX_FILE, folder / INDEX_FILE)

    with pytest.raises(IndexOutOfSyncError, match="SHA-256"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
def test_a_changed_index_file_is_detected(tmp_path):
    _, folder = save_index(tmp_path)
    data = bytearray((folder / INDEX_FILE).read_bytes())
    data[-1] ^= 0xFF
    (folder / INDEX_FILE).write_bytes(bytes(data))

    with pytest.raises(IndexOutOfSyncError, match="SHA-256"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
@pytest.mark.parametrize("field, value", [("n_parts", 7), ("doc_title", "Another Title")])
def test_fields_outside_the_embedded_text_are_covered_with_the_prefix_off(tmp_path, field, value):
    _, folder = save_index(tmp_path, with_context=False)
    edit_records(folder, lambda payload: payload["records"][2].update({field: value}))

    with pytest.raises(IndexOutOfSyncError, match="record 2"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
def test_build_records_stats_of_the_normalized_vectors():
    index = FaissChunkIndex.build(make_chunks(5), FakeEmbedder())

    stats = index.build_stats
    assert (stats["count"], stats["dtype"], stats["shape"]) == (5, "float32", (5, DIMENSION))
    assert stats["min_norm"] == pytest.approx(1.0, abs=FLOAT32_TOLERANCE)
    assert stats["max_norm"] == pytest.approx(1.0, abs=FLOAT32_TOLERANCE)


@needs_faiss
def test_a_changed_prefix_setting_in_the_records_is_detected(tmp_path):
    _, folder = save_index(tmp_path, with_context=False)
    edit_records(folder, lambda payload: payload.update(with_context=True))

    with pytest.raises(IndexOutOfSyncError, match="record 0"):
        FaissChunkIndex.load(folder, FakeEmbedder())


@needs_faiss
@pytest.mark.parametrize("content", ["not json at all", "{}", "[1, 2]"])
def test_an_unreadable_records_file_is_reported_as_out_of_sync(tmp_path, content):
    _, folder = save_index(tmp_path)
    (folder / RECORDS_FILE).write_text(content, encoding="utf-8")

    with pytest.raises(IndexOutOfSyncError, match="records.json"):
        FaissChunkIndex.load(folder, FakeEmbedder())