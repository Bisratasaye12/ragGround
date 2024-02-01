"""How the API hands ingestion jobs to the worker."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Protocol

from arq.connections import ArqRedis, RedisSettings, create_pool


class IngestQueue(Protocol):
    async def enqueue_ingest(self, doc_id: uuid.UUID, version: int) -> None: ...

    async def close(self) -> None: ...


class ArqQueue:
    def __init__(self, redis_url: str) -> None:
        self._settings = RedisSettings.from_dsn(redis_url)
        self._pool: ArqRedis | None = None

    async def enqueue_ingest(self, doc_id: uuid.UUID, version: int) -> None:
        if self._pool is None:
            self._pool = await create_pool(self._settings)
        # The job id makes duplicate enqueues of the same version a no-op.
        await self._pool.enqueue_job(
            "ingest_document_job", str(doc_id), version, _job_id=f"ingest:{doc_id}:{version}"
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()


class InlineQueue:
    """Runs the job immediately in-process. For tests and single-process demos."""

    def __init__(self, run: Callable[[uuid.UUID, int], Awaitable[object]]) -> None:
        self._run = run

    async def enqueue_ingest(self, doc_id: uuid.UUID, version: int) -> None:
        await self._run(doc_id, version)

    async def close(self) -> None:
        return None
