"""Hybrid retrieval: full-text + vector candidates → Reciprocal Rank Fusion → optional rerank.

Every candidate query joins chunks to documents and applies `readable_by(user)` in SQL, so
ranking (and therefore anything downstream, including the LLM) only ever sees chunks the user may
read. See ADR-003.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from opentelemetry import trace
from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragground.acl import readable_by
from ragground.models import Document, DocumentChunk, User
from ragground.providers.embeddings import Embedder
from ragground.providers.reranker import Reranker

SearchMode = Literal["fulltext", "vector", "hybrid", "hybrid_rerank"]
tracer = trace.get_tracer("ragground.retrieval")
RRF_K = 60  # standard constant from Cormack et al. (2009); damps the weight of top ranks

T = TypeVar("T")


@dataclass
class Hit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    source_uri: str
    heading_path: str
    text: str
    ord: int
    fulltext_rank: int | None = None  # 1-based rank within each retriever, None = not returned
    vector_rank: int | None = None
    fulltext_score: float | None = None
    vector_similarity: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass
class SearchResult:
    mode: SearchMode
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: int = 0


@dataclass
class SearchConfig:
    candidates_per_retriever: int = 30  # k1: how many each retriever contributes
    rerank_top_n: int = 20  # k2: how many fused candidates the reranker sees
    # Cross-encoder cost grows with input length; on CPU, truncating passages is the main
    # latency lever. None = full chunk.
    rerank_max_words: int | None = None
    hnsw_ef_search: int = 80
    # "and": every query word must match (websearch_to_tsquery default) - precise but misses most
    # natural-language questions. "or": any word may match; ts_rank_cd still rewards coverage.
    fulltext_operator: Literal["and", "or"] = "or"


def _base_query(user: User) -> Select[DocumentChunk, str, str]:
    return (
        select(DocumentChunk, Document.title, Document.source_uri)
        .join(Document, Document.id == DocumentChunk.document_id)
        # org_id on chunks lets the planner discard other tenants before the ACL subqueries.
        .where(DocumentChunk.org_id == user.org_id, readable_by(user))
    )


def _hit(chunk: DocumentChunk, title: str, uri: str) -> Hit:
    return Hit(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_title=title,
        source_uri=uri,
        heading_path=chunk.heading_path,
        text=chunk.text,
        ord=chunk.ord,
    )


_WORD = re.compile(r"\w+")


def fulltext_query_text(query: str, operator: Literal["and", "or"]) -> str:
    """Text handed to websearch_to_tsquery, which parses it safely (no tsquery injection).
    For "or", the query is reduced to its words joined by the websearch OR keyword; stop words
    are dropped by Postgres itself."""
    if operator == "and":
        return query
    return " or ".join(_WORD.findall(query))


async def fulltext_candidates(
    session: AsyncSession, user: User, query: str, k: int, operator: Literal["and", "or"] = "or"
) -> list[Hit]:
    tsq = func.websearch_to_tsquery("english", fulltext_query_text(query, operator))
    rank = func.ts_rank_cd(DocumentChunk.tsv, tsq, 32)  # 32: normalise rank into 0..1
    stmt = (
        _base_query(user)
        .add_columns(rank.label("rank"))
        .where(DocumentChunk.tsv.op("@@")(tsq))
        .order_by(rank.desc(), DocumentChunk.id)
        .limit(k)
    )
    hits = []
    for i, (chunk, title, uri, score) in enumerate((await session.execute(stmt)).all(), start=1):
        h = _hit(chunk, title, uri)
        h.fulltext_rank, h.fulltext_score = i, float(score)
        hits.append(h)
    return hits


async def vector_candidates(
    session: AsyncSession, user: User, query_vec: list[float], k: int, ef_search: int
) -> list[Hit]:
    # pgvector ≥ 0.8: with a WHERE filter, a plain HNSW scan can return fewer than k rows (it
    # filters *after* the index walk). Iterative scans keep walking until k rows pass the filter.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    distance = DocumentChunk.embedding.cosine_distance(query_vec)
    stmt = (
        _base_query(user)
        .add_columns(distance.label("distance"))
        .order_by(distance, DocumentChunk.id)
        .limit(k)
    )
    hits = []
    for i, (chunk, title, uri, dist) in enumerate((await session.execute(stmt)).all(), start=1):
        h = _hit(chunk, title, uri)
        h.vector_rank, h.vector_similarity = i, 1.0 - float(dist)
        hits.append(h)
    return hits


def reciprocal_rank_fusion(*ranked_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Merge ranked lists by Σ 1/(k + rank). Rank-based, so the retrievers' incomparable score
    scales (ts_rank vs cosine) never need calibrating against each other."""
    merged: dict[uuid.UUID, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(h.chunk_id, h)
            if h.fulltext_rank is not None:
                m.fulltext_rank, m.fulltext_score = h.fulltext_rank, h.fulltext_score
            if h.vector_rank is not None:
                m.vector_rank, m.vector_similarity = h.vector_rank, h.vector_similarity
            m.rrf_score = (m.rrf_score or 0.0) + 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda h: (-(h.rrf_score or 0.0), str(h.chunk_id)))


class Retriever:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        reranker: Reranker | None,
        config: SearchConfig | None = None,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or SearchConfig()

    async def search(self, user: User, query: str, mode: SearchMode, k: int = 5) -> SearchResult:
        timings: dict[str, float] = {}
        cfg = self.config

        async def timed(name: str, fn: Callable[[], Awaitable[T]]) -> T:
            start = time.perf_counter()
            with tracer.start_as_current_span(f"retrieval.{name}"):
                try:
                    return await fn()
                finally:
                    timings[name] = round((time.perf_counter() - start) * 1000, 1)

        async def run_fulltext() -> list[Hit]:
            async with self.sessionmaker() as s:
                return await fulltext_candidates(
                    s, user, query, cfg.candidates_per_retriever, cfg.fulltext_operator
                )

        async def run_vector() -> list[Hit]:
            vec = await timed("embed_query", lambda: self.embedder.embed_query(query))
            async with self.sessionmaker() as s, s.begin():  # SET LOCAL needs a transaction
                return await vector_candidates(
                    s, user, vec, cfg.candidates_per_retriever, cfg.hnsw_ef_search
                )

        total_start = time.perf_counter()
        if mode == "fulltext":
            fused = await timed("fulltext", run_fulltext)
        elif mode == "vector":
            fused = await timed("vector", run_vector)
        else:
            # The two retrievers are independent: run them concurrently on separate sessions.
            ft, vec = await asyncio.gather(
                timed("fulltext", run_fulltext), timed("vector", run_vector)
            )
            fused = reciprocal_rank_fusion(ft, vec)
        candidates = len(fused)

        if mode == "hybrid_rerank" and self.reranker is None:
            mode = "hybrid"  # no reranker configured: report what actually ran
        if mode == "hybrid_rerank" and self.reranker is not None and fused:
            pool = fused[: cfg.rerank_top_n]
            reranker = self.reranker
            passages = []
            for h in pool:
                body = h.text
                if cfg.rerank_max_words is not None:
                    body = " ".join(body.split()[: cfg.rerank_max_words])
                passages.append(f"{h.heading_path}\n\n{body}" if h.heading_path else body)
            scores = await timed("rerank", lambda: reranker.score(query, passages))
            for h, s in zip(pool, scores, strict=True):
                h.rerank_score = s
            fused = sorted(pool, key=lambda h: -(h.rerank_score or 0.0))

        timings["total"] = round((time.perf_counter() - total_start) * 1000, 1)
        return SearchResult(mode=mode, hits=fused[:k], timings_ms=timings, candidates=candidates)
