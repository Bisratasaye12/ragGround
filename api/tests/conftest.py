"""Test fixtures.

Database tests run against a real Postgres (the compose service locally, a service container in
CI) using a dedicated `ragground_test` database. It is recreated and migrated with Alembic once
per test session (so migrations themselves are exercised), and every table is truncated after
each test.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.engine import make_url

from ragground.config import Settings
from ragground.db import Base
from ragground.main import create_app

API_DIR = Path(__file__).resolve().parents[1]
TEST_DB_URL = os.environ.get(
    "RAGGROUND_TEST_DATABASE_URL",
    "postgresql+asyncpg://ragground:ragground@localhost:5442/ragground_test",
)
PASSWORD = "correct-horse-battery-staple"


async def _recreate_database(url: str) -> None:
    parsed = make_url(url)
    conn = await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.host,
        port=parsed.port,
        database="postgres",
    )
    try:
        name = parsed.database
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def migrated_db() -> str:
    try:
        asyncio.run(_recreate_database(TEST_DB_URL))
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.fail(
            f"Test Postgres unavailable at {TEST_DB_URL} ({exc}). "
            "Start it with: docker compose up -d postgres"
        )
    cfg = Config(str(API_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(API_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.upgrade(cfg, "head")
    return TEST_DB_URL


@pytest.fixture
def settings(migrated_db: str, tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        database_url=migrated_db,
        llm_provider="none",
        ingest_inline=True,  # run ingestion synchronously inside the request
        embedding_provider="hash",  # deterministic, no model download
        reranker_provider="overlap",
        storage_dir=str(tmp_path / "blobs"),
    )


@pytest.fixture
async def db_app(settings: Settings) -> AsyncIterator[FastAPI]:
    app = create_app(settings)
    yield app
    tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
    async with app.state.engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    await app.state.engine.dispose()
    await app.state.redis.aclose()


@pytest.fixture
async def client(db_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=db_app), base_url="http://test"
    ) as c:
        yield c


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def register(client: httpx.AsyncClient, org: str = "Acme") -> dict[str, str]:
    """Signs up a new org; returns the admin's token pair plus email."""
    email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post(
        "/auth/register", json={"org_name": org, "email": email, "password": PASSWORD}
    )
    assert resp.status_code == 201, resp.text
    return {**resp.json(), "email": email}


async def add_user(client: httpx.AsyncClient, admin_token: str, role: str = "member") -> str:
    """Admin creates a user; returns the new user's access token."""
    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post(
        "/users",
        json={"email": email, "password": PASSWORD, "role": role},
        headers=auth(admin_token),
    )
    assert resp.status_code == 201, resp.text
    login = await client.post("/auth/login", json={"email": email, "password": PASSWORD})
    token: str = login.json()["access_token"]
    return token
