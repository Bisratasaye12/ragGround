import httpx
import pytest

from ragground import seed as seed_module
from ragground.config import get_settings


async def test_seeded_users_can_log_in(
    client: httpx.AsyncClient, migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAGGROUND_DATABASE_URL", migrated_db)
    monkeypatch.setenv("RAGGROUND_ENV", "test")
    get_settings.cache_clear()
    try:
        await seed_module.seed()
        await seed_module.seed()  # idempotent
    finally:
        get_settings.cache_clear()

    for email in ("admin@acme.example.com", "Alice@Acme.example.com"):
        resp = await client.post(
            "/auth/login", json={"email": email, "password": seed_module.DEMO_PASSWORD}
        )
        assert resp.status_code == 200, (email, resp.text)
