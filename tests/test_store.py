"""Tests for app.rag.store with a fake embedder, a temporary folder and no model download."""

import hashlib
import re
from types import SimpleNamespace

import chromadb
import pytest

from app.rag.similarity import cosine_similarity
from app.rag import store as store_module
from app.rag.store import (
    COLLECTION_NAME,
    EmbedderMismatchError,
    IngestCounts,
    StoreNotFoundError,
    _existing_collection,
    ingest_kb,
    retrieve,
    store_info,
)

DIMENSION = 256


def hashed_vector(text, dimension=DIMENSION):
    """A simple deterministic embedding: hashed word counts. Never a zero vector."""
    vector = [0.0] * dimension
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dimension] += 1.0
    vector[0] += 0.001
    return vector


class FakeEmbedder:
    def __init__(self, name="fake-v1", dimension=DIMENSION):
        self.name = name
        self.dimension = dimension
        self.calls = []  # the list of texts of every embed() call that had any

    def embed(self, texts):
        if texts:
            self.calls.append(list(texts))
        return [hashed_vector(t, self.dimension) for t in texts]

    @property
    def texts_embedded(self):
        return sum(len(call) for call in self.calls)


FRUIT = "## Fruit Rules\nBananas cherries and grapes must be washed. Apples are stored cold.\n"
METAL = "## Metal Rules\nCopper iron and zinc stay dry. Steel is oiled monthly.\n"
WEATHER = "## Weather Notes\nRain snow and fog close the shop early.\n"


def write(kb, name, text):
    (kb / name).write_text(text, encoding="utf-8")


def make_kb(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    write(kb, "alpha.md", f"> banner\n\n# Alpha Handbook\n\n{FRUIT}\n{METAL}")
    write(kb, "beta.md", f"> banner\n\n# Beta Guide\n\n{WEATHER}")
    write(kb, "README.md", "# README\n\n## Not Indexed\nNever stored.\n")
    return kb


@pytest.fixture
def setup(tmp_path):
    return make_kb(tmp_path), tmp_path / "db", FakeEmbedder()


def stored_collection(db):
    return chromadb.PersistentClient(path=str(db)).get_collection(COLLECTION_NAME, embedding_function=None)


# ---------------------------------------------------------------------------
# Re-running ingestion
# ---------------------------------------------------------------------------

def test_ingesting_twice_leaves_everything_unchanged_and_embeds_nothing_the_second_time(setup):
    kb, db, embedder = setup

    first = ingest_kb(kb, db, embedder)
    embedded_after_first = embedder.texts_embedded
    second = ingest_kb(kb, db, embedder)

    assert first == IngestCounts(added=3, updated=0, unchanged=0, deleted=0)
    assert second == IngestCounts(added=0, updated=0, unchanged=3, deleted=0)
    assert embedder.texts_embedded == embedded_after_first == 3
    assert len(retrieve(db, embedder, "bananas", k=10)) == 3  # no duplicates


def test_editing_one_section_updates_only_that_chunk(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)
    embedded_before = embedder.texts_embedded

    write(kb, "alpha.md", (kb / "alpha.md").read_text(encoding="utf-8").replace("Apples", "Pears"))
    counts = ingest_kb(kb, db, embedder)

    assert counts == IngestCounts(added=0, updated=1, unchanged=2, deleted=0)
    assert embedder.texts_embedded - embedded_before == 1
    assert "Pears" in retrieve(db, embedder, "pears stored cold", k=1)[0].text


def test_adding_a_section_adds_one_chunk(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    write(kb, "beta.md", (kb / "beta.md").read_text(encoding="utf-8") + "\n## Tide Tables\nTides rise at noon.\n")

    assert ingest_kb(kb, db, embedder) == IngestCounts(added=1, updated=0, unchanged=3, deleted=0)


def test_deleting_a_section_deletes_its_chunks(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    write(kb, "alpha.md", (kb / "alpha.md").read_text(encoding="utf-8").replace(METAL, ""))
    counts = ingest_kb(kb, db, embedder)

    assert counts == IngestCounts(added=0, updated=0, unchanged=2, deleted=1)
    assert "alpha.md#Metal-Rules" not in {r.section_id for r in retrieve(db, embedder, "copper", k=10)}


def test_deleting_a_document_deletes_its_chunks(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    (kb / "beta.md").unlink()

    assert ingest_kb(kb, db, embedder) == IngestCounts(added=0, updated=0, unchanged=2, deleted=1)


def test_changing_the_prefix_setting_embeds_everything_again(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder, with_context=True)
    embedded_before = embedder.texts_embedded

    without = ingest_kb(kb, db, embedder, with_context=False)

    assert without == IngestCounts(added=0, updated=3, unchanged=0, deleted=0)
    assert embedder.texts_embedded - embedded_before == 3
    assert stored_collection(db).metadata["with_context"] is False
    assert store_info(db)["with_context"] is False
    assert ingest_kb(kb, db, embedder, with_context=True) == IngestCounts(0, 3, 0, 0)


def test_stored_text_is_never_prefixed_but_the_embedded_text_is(setup):
    kb, db, embedder = setup

    ingest_kb(kb, db, embedder, with_context=True)

    stored_text = retrieve(db, embedder, "bananas", k=1)[0].text
    assert not stored_text.startswith("Document:")
    assert any(text.startswith("Document: Alpha Handbook\nSection: Fruit Rules") for call in embedder.calls for text in call)


def test_a_section_that_gets_fewer_parts_deletes_the_stale_chunks_and_a_bigger_split_adds_them(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    body = " ".join(f"Sentence {i} says something about rule {i}." for i in range(1, 9))
    write(kb, "long.md", f"# Long\n\n## Only Section\n{body}\n")
    db, embedder = tmp_path / "db", FakeEmbedder()

    whole = ingest_kb(kb, db, embedder, max_chars=800)
    split = ingest_kb(kb, db, embedder, max_chars=80)
    back = ingest_kb(kb, db, embedder, max_chars=800)

    assert whole == IngestCounts(1, 0, 0, 0)
    assert split.added >= 1 and split.updated == 1 and split.deleted == 0
    assert back.deleted == split.added and back.updated == 1
    assert [r.id for r in retrieve(db, embedder, "sentence rule", k=10)] == ["long.md#Only-Section::1"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_a_phrase_copied_from_a_section_returns_that_section_first(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    top = retrieve(db, embedder, "Copper iron and zinc stay dry", k=3)

    assert top[0].section_id == "alpha.md#Metal-Rules"
    assert top[0].id == "alpha.md#Metal-Rules::1"
    assert top[0].source == "alpha.md" and top[0].section == "Metal Rules"
    assert "Steel is oiled monthly" in top[0].text
    assert [r.score for r in top] == sorted((r.score for r in top), reverse=True)


def test_readme_is_not_in_the_store(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    assert store_info(db)["count"] == 3
    assert all(r.source != "README.md" for r in retrieve(db, embedder, "never stored", k=10))


def test_score_equals_our_own_cosine_with_the_stored_vector(setup):
    """The trap test: a collection left on Chroma's default distance (squared L2) fails this."""
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)
    query = "bananas cherries copper snow"
    query_vector = embedder.embed([query])[0]
    stored = stored_collection(db).get(include=["embeddings"])
    stored_by_id = dict(zip(stored["ids"], stored["embeddings"]))

    results = retrieve(db, embedder, query, k=3)

    assert len(results) == 3
    for result in results:
        expected = cosine_similarity(query_vector, [float(x) for x in stored_by_id[result.id]])
        assert result.score == pytest.approx(expected, abs=1e-5)
        assert result.distance == pytest.approx(1.0 - result.score, abs=1e-9)
    assert len({round(r.score, 4) for r in results}) == 3  # real, distinct similarities


def test_k_larger_than_the_store_returns_everything(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    assert len(retrieve(db, embedder, "anything", k=50)) == 3


@pytest.mark.parametrize("query, k", [("bananas", 0), ("bananas", -1), ("   ", 3), ("", 3)])
def test_bad_arguments_are_rejected(setup, query, k):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    with pytest.raises(ValueError):
        retrieve(db, embedder, query, k=k)


def test_retrieving_before_ingesting_says_so_and_does_not_create_the_store(tmp_path):
    db = tmp_path / "never_made"

    with pytest.raises(StoreNotFoundError, match="ingest_kb"):
        retrieve(db, FakeEmbedder(), "bananas")
    assert not db.exists()


# ---------------------------------------------------------------------------
# Mixing models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "other",
    [
        pytest.param(FakeEmbedder(name="other-model"), id="different-name"),
        pytest.param(FakeEmbedder(dimension=128), id="different-dimension"),
    ],
)
def test_a_mismatched_embedder_is_refused_by_retrieve(setup, other):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)

    with pytest.raises(EmbedderMismatchError, match="fake-v1"):
        retrieve(db, other, "bananas")


def test_a_mismatched_embedder_is_refused_by_ingest_unless_rebuilding(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)
    other = FakeEmbedder(name="other-model", dimension=128)

    with pytest.raises(EmbedderMismatchError, match="rebuild"):
        ingest_kb(kb, db, other)

    counts = ingest_kb(kb, db, other, rebuild=True)
    assert counts == IngestCounts(added=3, updated=0, unchanged=0, deleted=0)
    assert retrieve(db, other, "Copper iron zinc", k=1)[0].section_id == "alpha.md#Metal-Rules"
    with pytest.raises(EmbedderMismatchError):
        retrieve(db, embedder, "bananas")


def test_store_info_reports_what_the_store_was_built_with(setup):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder, with_context=False)

    assert store_info(db) == {
        "embedder_name": "fake-v1",
        "embedder_dimension": DIMENSION,
        "with_context": False,
        "count": 3,
    }


def test_an_embedder_that_returns_the_wrong_length_is_rejected(setup):
    kb, db, _ = setup

    class WrongLength(FakeEmbedder):
        def embed(self, texts):
            return [v[:10] for v in super().embed(texts)]

    with pytest.raises(ValueError, match="length 10"):
        ingest_kb(kb, db, WrongLength())


# ---------------------------------------------------------------------------
# Only a missing collection counts as "not found"
# ---------------------------------------------------------------------------

class FakeChromaClient:
    """Stands in for the Chroma client to make list_collections / get_collection fail on demand."""

    def __init__(self, listed=(), list_error=None, get_error=None):
        self.listed, self.list_error, self.get_error = list(listed), list_error, get_error
        self.created = False
        self.get_calls = 0

    def list_collections(self):
        if self.list_error:
            raise self.list_error
        return self.listed

    def get_collection(self, name, embedding_function=None):
        self.get_calls += 1
        if self.get_error:
            raise self.get_error
        return f"collection:{name}"

    def create_collection(self, **kwargs):
        self.created = True
        raise AssertionError("must not create a collection when looking one up failed")


def test_a_name_that_is_not_listed_means_no_collection_and_get_is_not_called():
    client = FakeChromaClient(listed=[SimpleNamespace(name="some_other_collection")])

    assert _existing_collection(client) is None
    assert client.get_calls == 0


@pytest.mark.parametrize(
    "listed",
    [
        pytest.param([SimpleNamespace(name=COLLECTION_NAME)], id="collection-objects-as-in-chroma-1.x"),
        pytest.param([COLLECTION_NAME], id="names-as-in-chroma-0.6"),
    ],
)
def test_a_listed_collection_is_fetched_whether_listing_gives_objects_or_names(listed):
    assert _existing_collection(FakeChromaClient(listed=listed)) == f"collection:{COLLECTION_NAME}"


def test_an_error_while_listing_is_not_mistaken_for_an_empty_store():
    client = FakeChromaClient(list_error=RuntimeError("database disk image is malformed"))

    with pytest.raises(RuntimeError, match="malformed"):
        _existing_collection(client)


def test_an_error_while_fetching_a_listed_collection_is_not_mistaken_for_a_missing_one():
    client = FakeChromaClient(listed=[COLLECTION_NAME], get_error=RuntimeError("segment is corrupt"))

    with pytest.raises(RuntimeError, match="corrupt"):
        _existing_collection(client)


def test_ingest_and_retrieve_stop_on_a_corrupt_store_instead_of_creating_a_new_one(setup, monkeypatch):
    kb, db, embedder = setup
    ingest_kb(kb, db, embedder)
    broken = FakeChromaClient(listed=[COLLECTION_NAME], get_error=RuntimeError("segment is corrupt"))
    monkeypatch.setattr(store_module, "_client", lambda path: broken)

    with pytest.raises(RuntimeError, match="corrupt"):
        ingest_kb(kb, db, embedder)
    with pytest.raises(RuntimeError, match="corrupt"):
        retrieve(db, embedder, "bananas")
    assert broken.created is False