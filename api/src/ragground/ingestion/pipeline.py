"""The ingestion pipeline: blob → parse → chunk → embed → replace the document's chunks.

Runs in the worker (and inline in tests). Concurrency rule: each (re)ingest request bumps
`documents.ingest_version`, and a job only commits if the version it was enqueued with is still
current — so a slow job for an old upload can never overwrite chunks from a newer one.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragground.ingestion.chunkers import HEADING_SEP, Chunker
from ragground.ingestion.parsers import ParseError, parse
from ragground.ingestion.storage import BlobStorage
from ragground.models import Document, DocumentChunk, DocumentStatus
from ragground.providers.embeddings import Embedder

log = structlog.get_logger()

EMBED_BATCH = 64


@dataclass
class IngestDeps:
    sessionmaker: async_sessionmaker[AsyncSession]
    storage: BlobStorage
    embedder: Embedder
    chunker: Chunker


@dataclass
class IngestResult:
    status: str  # "ready" | "failed" | "stale" | "missing"
    chunks: int = 0
    error: str | None = None


async def _mark(
    deps: IngestDeps, doc_id: uuid.UUID, version: int, status: DocumentStatus, error: str | None
) -> bool:
    """Set status only if this job is still the current version. Returns whether it applied."""
    async with deps.sessionmaker() as session:
        result = await session.execute(
            update(Document)
            .where(Document.id == doc_id, Document.ingest_version == version)
            .values(status=status, error=error)
        )
        await session.commit()
        return bool(result.rowcount)  # type: ignore[attr-defined]


async def ingest_document(deps: IngestDeps, doc_id: uuid.UUID, version: int) -> IngestResult:
    bound = log.bind(document_id=str(doc_id), version=version)

    async with deps.sessionmaker() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return IngestResult("missing")  # deleted before the job ran
        if doc.ingest_version != version:
            return IngestResult("stale")
        storage_key, filename = doc.storage_key, doc.filename or doc.title

    if not await _mark(deps, doc_id, version, DocumentStatus.processing, None):
        return IngestResult("stale")
    if storage_key is None:
        await _mark(deps, doc_id, version, DocumentStatus.failed, "no stored content")
        return IngestResult("failed", error="no stored content")

    data = await deps.storage.get(storage_key)
    try:
        parsed = parse(data, filename)
    except ParseError as exc:  # permanent: record it, don't retry
        bound.info("ingest.parse_failed", error=str(exc))
        await _mark(deps, doc_id, version, DocumentStatus.failed, str(exc))
        return IngestResult("failed", error=str(exc))

    chunks = deps.chunker.chunk(parsed)
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vectors.extend(await deps.embedder.embed_documents([c.embedding_text for c in batch]))

    async with deps.sessionmaker() as session, session.begin():
        # Lock the row and re-check the version inside the write transaction.
        current = await session.scalar(
            select(Document.ingest_version).where(Document.id == doc_id).with_for_update()
        )
        if current != version:
            bound.info("ingest.stale_on_commit", current=current)
            return IngestResult("stale")
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
        org_id = await session.scalar(select(Document.org_id).where(Document.id == doc_id))
        session.add_all(
            DocumentChunk(
                document_id=doc_id,
                org_id=org_id,
                ord=c.ord,
                text=c.text,
                heading_path=HEADING_SEP.join(c.heading_path),
                word_count=c.word_count,
                embedding=v,
                meta={"embedding_model": deps.embedder.model_name},
            )
            for c, v in zip(chunks, vectors, strict=True)
        )
        await session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(
                status=DocumentStatus.ready,
                error=None,
                title=parsed.title[:500],
                ingested_at=datetime.now(UTC),
            )
        )
    bound.info("ingest.done", chunks=len(chunks))
    return IngestResult("ready", chunks=len(chunks))


async def mark_failed(deps: IngestDeps, doc_id: uuid.UUID, version: int, error: str) -> None:
    """Called by the worker when retries are exhausted."""
    await _mark(deps, doc_id, version, DocumentStatus.failed, error[:2000])
