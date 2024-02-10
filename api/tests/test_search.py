"""Hybrid retrieval against Postgres (hash embedder + word-overlap reranker test doubles)."""

import uuid
from typing import Any

import httpx
import pytest

from ragground.retrieval.search import Hit, reciprocal_rank_fusion

from .conftest import add_user, auth, register

ALL_MODES = ["fulltext", "vector", "hybrid", "hybrid_rerank"]

HANDBOOK = b"""# Handbook

## Vacation

Everyone gets 25 days of paid vacation per year. Unused vacation days roll over.

## Expenses

Submit expense receipts within 30 days through the finance portal.
"""

SALARIES = b"""# Compensation

## Salary bands

Senior engineers are paid between 150k and 190k. Salary bands are reviewed yearly.
"""


async def _upload(
    client: httpx.AsyncClient, token: str, name: str, data: bytes, groups: list[str] | None = None
) -> dict[str, Any]:
    form: dict[str, Any] = {"group_ids": groups} if groups else {}
    resp = await client.post(
        "/documents", files={"file": (name, data)}, data=form, headers=auth(token)
    )
    assert resp.status_code == 201, resp.text
    doc: dict[str, Any] = resp.json()["document"]
    assert doc["status"] == "ready"
    return doc


async def _search(
    client: httpx.AsyncClient, token: str, query: str, mode: str, k: int = 5
) -> dict[str, Any]:
    resp = await client.post(
        "/search", json={"query": query, "mode": mode, "k": k}, headers=auth(token)
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


@pytest.mark.parametrize("mode", ALL_MODES)
async def test_each_mode_finds_the_relevant_chunk(client: httpx.AsyncClient, mode: str) -> None:
    admin = await register(client)
    await _upload(client, admin["access_token"], "handbook.md", HANDBOOK)
    body = await _search(client, admin["access_token"], "expense receipts", mode)
    assert body["mode"] == mode
    top = body["hits"][0]
    assert "receipts" in top["text"]
    assert top["heading_path"] == "Handbook > Expenses"
    assert top["document_title"] == "Handbook"
    assert "total" in body["timings_ms"]


async def test_hybrid_reports_both_retrievers_and_rerank(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    await _upload(client, admin["access_token"], "handbook.md", HANDBOOK)
    body = await _search(client, admin["access_token"], "vacation days", "hybrid_rerank")
    top = body["hits"][0]
    assert top["fulltext_rank"] is not None and top["vector_rank"] is not None
    assert top["rrf_score"] > 0 and top["rerank_score"] is not None
    assert {"fulltext", "vector", "embed_query", "rerank"} <= set(body["timings_ms"])


@pytest.mark.parametrize("mode", ALL_MODES)
async def test_acl_forbidden_chunks_never_returned(client: httpx.AsyncClient, mode: str) -> None:
    """The salary doc is the *best* match for the query, but the member can't read it."""
    admin = await register(client)
    h = auth(admin["access_token"])
    hr = (await client.post("/groups", json={"name": "hr"}, headers=h)).json()
    await _upload(client, admin["access_token"], "handbook.md", HANDBOOK)
    await _upload(client, admin["access_token"], "salaries.md", SALARIES, groups=[hr["id"]])
    member = await add_user(client, admin["access_token"])

    query = "senior engineers salary bands"
    member_hits = (await _search(client, member, query, mode, k=50))["hits"]
    assert all(hit["document_title"] != "Compensation" for hit in member_hits)
    assert all("150k" not in hit["text"] for hit in member_hits)

    admin_hits = (await _search(client, admin["access_token"], query, mode))["hits"]
    assert admin_hits[0]["document_title"] == "Compensation"  # admins can read it


@pytest.mark.parametrize("mode", ALL_MODES)
async def test_other_orgs_documents_are_invisible(client: httpx.AsyncClient, mode: str) -> None:
    acme = await register(client, org="Acme")
    globex = await register(client, org="Globex")
    await _upload(client, acme["access_token"], "handbook.md", HANDBOOK)
    hits = (await _search(client, globex["access_token"], "vacation", mode, k=50))["hits"]
    assert hits == []


async def test_search_validation_and_auth(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    t = admin["access_token"]
    assert (await client.post("/search", json={"query": "x"})).status_code == 401
    bad_mode = await client.post("/search", json={"query": "x", "mode": "magic"}, headers=auth(t))
    assert bad_mode.status_code == 422
    empty = await client.post("/search", json={"query": ""}, headers=auth(t))
    assert empty.status_code == 422
    # Query text is passed as data, never as SQL / tsquery syntax.
    weird = await _search(client, t, "'); DROP TABLE chunks; -- & | ! :*", "hybrid")
    assert weird["hits"] == []


def _h(n: int) -> Hit:
    return Hit(uuid.UUID(int=n), uuid.UUID(int=0), "t", "u", "", f"chunk {n}", n)


def test_rrf_rewards_agreement_between_retrievers() -> None:
    a, b, c = _h(1), _h(2), _h(3)
    for i, h in enumerate([a, b, c], start=1):
        h.fulltext_rank = i
    vec = [_h(3), _h(4), _h(1)]
    for i, h in enumerate(vec, start=1):
        h.vector_rank = i
    fused = reciprocal_rank_fusion([a, b, c], vec)
    order = [h.chunk_id.int for h in fused]
    # 1 and 3 appear in both lists, so they beat 2 and 4 which each appear once.
    assert set(order[:2]) == {1, 3}
    assert set(order[2:]) == {2, 4}
    top = fused[0]
    # The merged hit carries both retrievers' ranks.
    assert top.fulltext_rank is not None and top.vector_rank is not None


def test_fulltext_or_query_text() -> None:
    from ragground.retrieval.search import fulltext_query_text

    assert fulltext_query_text("How do I disable the cache?", "or") == (
        "How or do or I or disable or the or cache"
    )
    assert fulltext_query_text("exact words", "and") == "exact words"
    # tsquery operators in user input are just dropped, never interpreted.
    assert fulltext_query_text("a & b | !c:*", "or") == "a or b or c"
