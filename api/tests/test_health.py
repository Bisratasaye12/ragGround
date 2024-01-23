import asyncio

import httpx
import pytest
from fastapi import FastAPI

from ragground.config import Settings
from ragground.main import create_app


async def _ok() -> None:
    return None


async def _fail() -> None:
    raise ConnectionError("connection refused")


async def _hang() -> None:
    await asyncio.sleep(60)


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def app() -> FastAPI:
    return create_app(Settings())


async def test_liveness(app: FastAPI) -> None:
    async with _client(app) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_ready_when_all_checks_pass(app: FastAPI) -> None:
    app.state.health_checks = {"postgres": _ok, "redis": _ok}
    async with _client(app) as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["postgres"]["ok"] is True


async def test_degraded_when_a_check_fails(app: FastAPI) -> None:
    app.state.health_checks = {"postgres": _ok, "redis": _fail}
    async with _client(app) as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] == {
        "ok": False,
        "latency_ms": body["checks"]["redis"]["latency_ms"],
        "error": "ConnectionError: connection refused",
    }


async def test_hanging_check_times_out(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragground.health.CHECK_TIMEOUT_S", 0.05)
    app.state.health_checks = {"llm": _hang}
    async with _client(app) as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"]["llm"]["error"] == "TimeoutError"
