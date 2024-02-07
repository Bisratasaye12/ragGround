"""Upload → ingestion → chunks, end to end against Postgres (inline queue, hash embedder)."""

import uuid
from typing import Any

import httpx
from fastapi import FastAPI
from sqlalchemy import func, select

from ragground.ingestion.chunkers import HeadingChunker
from ragground.ingestion.pipeline import IngestDeps, ingest_document
from ragground.models import Document, DocumentChunk
from ragground.providers.embeddings import HashEmbedder

from .conftest import add_user, auth, register

GUIDE = b"""# Handbook

## Vacation

Everyone gets 25 days of paid vacation per year.

## Expenses

Submit receipts within 30 days.
"""


async def _upload(
    client: httpx.AsyncClient,
    token: str,
    name: str = "handbook.md",
    data: bytes = GUIDE,
    groups: list[str] | None = None,
) -> httpx.Response:
    form: dict[str, Any] = {"group_ids": groups} if groups else {}
    return await client.post(
        "/documents", files={"file": (name, data)}, data=form, headers=auth(token)
    )


async def _chunks(app: FastAPI, doc_id: str) -> list[DocumentChunk]:
    async with app.state.sessionmaker() as s:
        rows = await s.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == uuid.UUID(doc_id))
            .order_by(DocumentChunk.ord)
        )
        return list(rows)


async def test_upload_ingests_into_chunks(client: httpx.AsyncClient, db_app: FastAPI) -> None:
    admin = await register(client)
    resp = await _upload(client, admin["access_token"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["outcome"] == "created"
    doc = body["document"]
    assert doc["status"] == "ready"
    assert doc["title"] == "Handbook"
    assert doc["chunk_count"] >= 1

    chunks = await _chunks(db_app, doc["id"])
    assert any("25 days" in c.text for c in chunks)
    assert any(c.heading_path == "Handbook > Vacation" for c in chunks)
    assert all(len(c.embedding) == 384 for c in chunks)

    # Postgres maintains the full-text vector itself.
    async with db_app.state.sessionmaker() as s:
        hits = await s.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.tsv.op("@@")(func.plainto_tsquery("english", "receipts")))
        )
    assert hits == 1


async def test_reupload_same_bytes_is_noop(client: httpx.AsyncClient, db_app: FastAPI) -> None:
    admin = await register(client)
    first = (await _upload(client, admin["access_token"])).json()["document"]
    before = {c.id for c in await _chunks(db_app, first["id"])}

    again = await _upload(client, admin["access_token"])
    assert again.status_code == 200
    assert again.json()["outcome"] == "unchanged"
    assert {c.id for c in await _chunks(db_app, first["id"])} == before  # chunks untouched


async def test_reupload_changed_content_replaces_chunks(
    client: httpx.AsyncClient, db_app: FastAPI
) -> None:
    admin = await register(client)
    first = (await _upload(client, admin["access_token"])).json()["document"]
    changed = GUIDE.replace(b"25 days", b"30 days")
    resp = await _upload(client, admin["access_token"], data=changed)
    assert resp.status_code == 202
    assert resp.json()["outcome"] == "updated"
    assert resp.json()["document"]["id"] == first["id"]
    texts = " ".join(c.text for c in await _chunks(db_app, first["id"]))
    assert "30 days" in texts and "25 days" not in texts


async def test_unparseable_file_is_marked_failed(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    resp = await _upload(client, admin["access_token"], name="broken.pdf", data=b"garbage")
    doc = resp.json()["document"]
    assert doc["status"] == "failed"
    assert "PDF" in doc["error"]


async def test_upload_validation(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    t = admin["access_token"]
    assert (await _upload(client, t, name="x.exe")).status_code == 415
    assert (await _upload(client, t, data=b"")).status_code == 422
    # path components are stripped from the filename
    resp = await _upload(client, t, name="../../etc/passwd.txt", data=b"root stuff")
    assert resp.json()["document"]["filename"] == "passwd.txt"


async def test_document_visibility_follows_acl(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    h = auth(admin["access_token"])
    hr = (await client.post("/groups", json={"name": "hr"}, headers=h)).json()
    member = await add_user(client, admin["access_token"])

    secret = await _upload(
        client,
        admin["access_token"],
        name="salaries.md",
        data=b"# Salaries\n\nCEO earns a lot.",
        groups=[hr["id"]],
    )
    public = await _upload(client, admin["access_token"])
    secret_id = secret.json()["document"]["id"]

    listed = {d["title"] for d in (await client.get("/documents", headers=auth(member))).json()}
    assert listed == {"Handbook"}
    assert public.status_code == 201
    # Not visible → 404 (not 403), and can't be overwritten by uploading the same filename.
    assert (await client.get(f"/documents/{secret_id}", headers=auth(member))).status_code == 404
    overwrite = await _upload(client, member, name="salaries.md", data=b"# Salaries\n\nzero")
    assert overwrite.status_code == 409


async def test_member_cannot_restrict_to_foreign_group(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    hr = (
        await client.post("/groups", json={"name": "hr"}, headers=auth(admin["access_token"]))
    ).json()
    member = await add_user(client, admin["access_token"])
    resp = await _upload(client, member, groups=[hr["id"]])
    assert resp.status_code == 403


async def test_delete_requires_uploader_or_admin(
    client: httpx.AsyncClient, db_app: FastAPI
) -> None:
    admin = await register(client)
    member = await add_user(client, admin["access_token"])
    doc_id = (await _upload(client, admin["access_token"])).json()["document"]["id"]
    assert (await client.delete(f"/documents/{doc_id}", headers=auth(member))).status_code == 403
    h = auth(admin["access_token"])
    assert (await client.delete(f"/documents/{doc_id}", headers=h)).status_code == 204
    assert await _chunks(db_app, doc_id) == []  # chunks cascade


async def test_stale_job_never_overwrites_newer_version(
    client: httpx.AsyncClient, db_app: FastAPI
) -> None:
    """A slow job for version N must not clobber chunks written for version N+1."""
    admin = await register(client)
    doc_id = (await _upload(client, admin["access_token"])).json()["document"]["id"]
    await _upload(client, admin["access_token"], data=GUIDE.replace(b"25", b"40"))  # version 2

    deps = IngestDeps(
        sessionmaker=db_app.state.sessionmaker,
        storage=db_app.state.storage,
        embedder=HashEmbedder(),
        chunker=HeadingChunker(),
    )
    result = await ingest_document(deps, uuid.UUID(doc_id), version=1)
    assert result.status == "stale"
    async with db_app.state.sessionmaker() as s:
        doc = await s.get(Document, uuid.UUID(doc_id))
        assert doc is not None and doc.ingest_version == 2
    assert "40 days" in " ".join(c.text for c in await _chunks(db_app, doc_id))
