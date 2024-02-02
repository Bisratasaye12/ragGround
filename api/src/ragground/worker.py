"""Background worker: `uv run arq ragground.worker.WorkerSettings`.

Retries transient failures (DB blips, storage errors) with backoff; parse errors are permanent
and recorded by the pipeline itself. After the last try the document is marked failed.
"""

import uuid
from typing import Any, ClassVar

import structlog
from arq import Retry
from arq.connections import RedisSettings

from ragground.config import get_settings
from ragground.db import make_engine, make_sessionmaker
from ragground.ingestion.chunkers import get_chunker
from ragground.ingestion.pipeline import IngestDeps, ingest_document, mark_failed
from ragground.ingestion.storage import LocalStorage
from ragground.log_config import configure_logging
from ragground.providers.embeddings import make_embedder

log = structlog.get_logger()
settings = get_settings()


async def startup(ctx: dict[str, Any]) -> None:
    configure_logging(settings.log_level, settings.log_json)
    engine = make_engine(settings.database_url)
    ctx["engine"] = engine
    ctx["deps"] = IngestDeps(
        sessionmaker=make_sessionmaker(engine),
        storage=LocalStorage(settings.storage_dir),
        embedder=make_embedder(settings.embedding_provider, settings.embedding_model),
        chunker=get_chunker(settings.chunker, settings.chunk_max_size, settings.chunk_size_unit),
    )
    log.info("worker.startup", embedder=settings.embedding_model, chunker=settings.chunker)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["engine"].dispose()


async def ingest_document_job(ctx: dict[str, Any], doc_id: str, version: int) -> str:
    deps: IngestDeps = ctx["deps"]
    try:
        result = await ingest_document(deps, uuid.UUID(doc_id), version)
    except Exception as exc:
        tries: int = ctx["job_try"]
        if tries >= settings.ingest_max_tries:
            log.error("ingest.gave_up", document_id=doc_id, error=repr(exc))
            await mark_failed(deps, uuid.UUID(doc_id), version, f"{type(exc).__name__}: {exc}")
            return "failed"
        log.warning("ingest.retry", document_id=doc_id, attempt=tries, error=repr(exc))
        raise Retry(defer=2**tries) from exc
    return result.status


class WorkerSettings:
    functions: ClassVar = [ingest_document_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_tries = settings.ingest_max_tries
    job_timeout = 600
