"""Embedding providers behind one small interface.

The DB column has a fixed dimension (see migrations), so switching to a model with a different
dimension needs a migration + re-embed. `EMBEDDING_DIM` is the single source of truth.
"""

import asyncio
import hashlib
import math
from typing import Protocol

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5


class Embedder(Protocol):
    model_name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed: free, offline after the first model download."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.batch_size = batch_size
        self.dim = EMBEDDING_DIM

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=self.batch_size)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # CPU-bound: keep the event loop free.
        return await asyncio.to_thread(self._embed, texts)

    async def embed_query(self, text: str) -> list[float]:
        def run() -> list[float]:
            vector: list[float] = next(iter(self._model.query_embed(text))).tolist()
            return vector

        return await asyncio.to_thread(run)


class HashEmbedder:
    """Deterministic, dependency-free embeddings for tests: bag of hashed words, L2-normalised.
    Texts sharing words get similar vectors, which is enough to test the plumbing."""

    model_name = "hash-test"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in text.lower().split():
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "fastembed":
        return FastEmbedEmbedder(model_name)
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedding provider {provider!r}")
