"""Tests for app.rag.embedding: importing never needs torch, and the wrapper behaves."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", ["app.rag.embedding", "app.rag.store"])
def test_importing_does_not_load_sentence_transformers_or_torch(module):
    code = (
        f"import sys, {module}\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('sentence_transformers', 'torch')]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class _FakeVectors:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return [list(row) for row in self.rows]


class _FakeModel:
    def __init__(self, name, device=None):
        self.name, self.device, self.encoded = name, device, []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, show_progress_bar=False):
        self.encoded.append(list(texts))
        return _FakeVectors([[float(len(t)), 0.0, 1.0] for t in texts])


def install_fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_the_wrapper_reports_name_and_dimension_and_returns_plain_lists(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    from app.rag.embedding import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("some/model")

    assert embedder.name == "some/model"
    assert embedder.dimension == 3
    assert embedder.embed(["ab", "abc"]) == [[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]


def test_embedding_no_texts_does_not_call_the_model(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    from app.rag.embedding import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("some/model")

    assert embedder.embed([]) == []
    assert embedder._model.encoded == []