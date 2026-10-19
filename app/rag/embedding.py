"""Embedders: turn texts into vectors, behind one small interface.

`SentenceTransformerEmbedder` imports `sentence_transformers` only when one is
built, so importing this module, or the store, never needs PyTorch.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that turns a list of texts into a list of equal-length vectors."""

    name: str  # identifies the model; the store refuses to mix embedders with different names
    dimension: int  # length of every vector `embed` returns

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order. An empty list gives an empty list."""
        ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model run on the CPU. No prefix is added to any text."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # imported here on purpose

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        dimension = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dimension = int(dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, show_progress_bar=False).tolist()