import time

import httpx
import jwt
import pytest

from ragground.config import INSECURE_DEV_SECRET, Settings
from ragground.security import (
    JWT_ALGORITHM,
    InvalidTokenError,
    create_access_token,
    decode_access_token,
)

from .conftest import PASSWORD, add_user, auth, register


async def test_register_returns_tokens_and_creates_admin(client: httpx.AsyncClient) -> None:
    tokens = await register(client, org="Acme")
    me = await client.get("/auth/me", headers=auth(tokens["access_token"]))
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == tokens["email"]
    assert body["role"] == "admin"


async def test_register_duplicate_email_conflicts(client: httpx.AsyncClient) -> None:
    tokens = await register(client)
    resp = await client.post(
        "/auth/register",
        json={"org_name": "Other", "email": tokens["email"].upper(), "password": PASSWORD},
    )
    assert resp.status_code == 409  # emails are case-insensitive


async def test_register_rejects_short_password(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/auth/register", json={"org_name": "A", "email": "a@example.com", "password": "short"}
    )
    assert resp.status_code == 422


async def test_login_success_and_failures(client: httpx.AsyncClient) -> None:
    tokens = await register(client)
    ok = await client.post("/auth/login", json={"email": tokens["email"], "password": PASSWORD})
    assert ok.status_code == 200
    wrong_pw = await client.post("/auth/login", json={"email": tokens["email"], "password": "x"})
    unknown = await client.post("/auth/login", json={"email": "no@example.com", "password": "x"})
    # Same status and message either way: no user enumeration.
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json() == unknown.json()


async def test_me_requires_valid_token(client: httpx.AsyncClient, settings: Settings) -> None:
    assert (await client.get("/auth/me")).status_code == 401
    assert (await client.get("/auth/me", headers=auth("garbage"))).status_code == 401
    forged = jwt.encode(
        {"sub": "00000000-0000-0000-0000-000000000000", "typ": "access", "exp": time.time() + 60},
        "a-different-secret-of-at-least-32-bytes",
        algorithm=JWT_ALGORITHM,
    )
    assert (await client.get("/auth/me", headers=auth(forged))).status_code == 401


def test_expired_access_token_is_rejected() -> None:
    import uuid

    token = create_access_token(uuid.uuid4(), uuid.uuid4(), "s" * 32, ttl_s=-1)
    with pytest.raises(InvalidTokenError, match="expired"):
        decode_access_token(token, "s" * 32)


async def test_refresh_rotates_and_detects_reuse(client: httpx.AsyncClient) -> None:
    tokens = await register(client)
    first = tokens["refresh_token"]

    rotated = await client.post("/auth/refresh", json={"refresh_token": first})
    assert rotated.status_code == 200
    second = rotated.json()["refresh_token"]
    assert second != first

    # Reusing the old token is treated as theft: rejected AND the whole family is revoked.
    reuse = await client.post("/auth/refresh", json={"refresh_token": first})
    assert reuse.status_code == 401
    after_reuse = await client.post("/auth/refresh", json={"refresh_token": second})
    assert after_reuse.status_code == 401


async def test_logout_revokes_session(client: httpx.AsyncClient) -> None:
    tokens = await register(client)
    resp = await client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 204
    again = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert again.status_code == 401


async def test_deactivated_user_is_locked_out_immediately(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    member_token = await add_user(client, admin["access_token"])
    me = await client.get("/auth/me", headers=auth(member_token))
    member_id = me.json()["id"]

    resp = await client.post(f"/users/{member_id}/deactivate", headers=auth(admin["access_token"]))
    assert resp.status_code == 200
    # The still-unexpired access token stops working at once.
    assert (await client.get("/auth/me", headers=auth(member_token))).status_code == 401


def test_production_refuses_insecure_secret() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(env="production", jwt_secret=INSECURE_DEV_SECRET)
    Settings(env="production", jwt_secret="x" * 40)  # a real secret is accepted
