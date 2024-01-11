"""Liveness and readiness checks.

`/health` is liveness: the process is up. `/health/ready` is readiness: every dependency the API
needs (Postgres, Redis, the LLM provider) answers. Checks are stored on `app.state.health_checks`
so tests can swap them out without real services.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

import httpx
from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

HealthCheck = Callable[[], Awaitable[None]]

CHECK_TIMEOUT_S = 2.0

router = APIRouter(tags=["health"])


class CheckResult(BaseModel):
    ok: bool
    latency_ms: float
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, CheckResult]


def postgres_check(engine: AsyncEngine) -> HealthCheck:
    async def check() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    return check


def redis_check(client: Redis) -> HealthCheck:
    async def check() -> None:
        await client.ping()

    return check


def ollama_check(base_url: str, model: str) -> HealthCheck:
    async def check() -> None:
        async with httpx.AsyncClient(base_url=base_url, timeout=CHECK_TIMEOUT_S) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            models = {m["name"] for m in resp.json().get("models", [])}
            if model not in models:
                raise RuntimeError(f"model {model!r} not pulled (run: ollama pull {model})")

    return check


async def _run(check: HealthCheck) -> CheckResult:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT_S)
    except Exception as exc:  # any failure means "not ready"; report it, don't raise
        return CheckResult(
            ok=False,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
        )
    return CheckResult(ok=True, latency_ms=round((time.perf_counter() - start) * 1000, 1))


@router.get("/health")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    checks: dict[str, HealthCheck] = request.app.state.health_checks
    names = list(checks)
    results = await asyncio.gather(*(_run(checks[n]) for n in names))
    by_name = dict(zip(names, results, strict=True))
    ready = all(r.ok for r in results)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "degraded", checks=by_name)
