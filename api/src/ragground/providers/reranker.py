"""Rerankers: score (query, passage) pairs jointly. Slower than vector similarity but much more
precise, so they're applied only to the top fused candidates."""

import asyncio
import re
from typing import Protocol


class Reranker(Protocol):
    model_name: str

    async def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage (higher = more relevant). Scale is model-specific."""
        ...


class FastEmbedReranker:
    """Local ONNX cross-encoder (default: ms-marco MiniLM-L-12, ~120 MB, CPU-friendly)."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-12-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        def run() -> list[float]:
            return [float(s) for s in self._model.rerank(query, passages)]

        return await asyncio.to_thread(run)


_WORD = re.compile(r"\w+")


class OverlapReranker:
    """Deterministic test double: fraction of query words present in the passage."""

    model_name = "overlap-test"

    async def score(self, query: str, passages: list[str]) -> list[float]:
        q = {w.lower() for w in _WORD.findall(query)}
        if not q:
            return [0.0] * len(passages)
        return [len(q & {w.lower() for w in _WORD.findall(p)}) / len(q) for p in passages]


def make_reranker(provider: str, model_name: str) -> Reranker | None:
    if provider == "none":
        return None
    if provider == "fastembed":
        return FastEmbedReranker(model_name)
    if provider == "overlap":
        return OverlapReranker()
    raise ValueError(f"unknown reranker provider {provider!r}")
