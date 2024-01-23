import httpx

from .conftest import PASSWORD, add_user, auth, register


async def test_member_cannot_use_admin_endpoints(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    member = await add_user(client, admin["access_token"])
    assert (await client.get("/users", headers=auth(member))).status_code == 403
    resp = await client.post("/groups", json={"name": "eng"}, headers=auth(member))
    assert resp.status_code == 403


async def test_admin_manages_groups_and_membership(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    h = auth(admin["access_token"])
    member = await add_user(client, admin["access_token"])
    member_id = (await client.get("/auth/me", headers=auth(member))).json()["id"]

    group = (await client.post("/groups", json={"name": "engineering"}, headers=h)).json()
    dup = await client.post("/groups", json={"name": "engineering"}, headers=h)
    assert dup.status_code == 409

    url = f"/groups/{group['id']}/members/{member_id}"
    assert (await client.put(url, headers=h)).status_code == 204
    assert (await client.put(url, headers=h)).status_code == 204  # idempotent
    me = (await client.get("/auth/me", headers=auth(member))).json()
    assert [g["name"] for g in me["groups"]] == ["engineering"]

    assert (await client.delete(url, headers=h)).status_code == 204
    me = (await client.get("/auth/me", headers=auth(member))).json()
    assert me["groups"] == []


async def test_orgs_are_isolated(client: httpx.AsyncClient) -> None:
    acme = await register(client, org="Acme")
    globex = await register(client, org="Globex")
    acme_h, globex_h = auth(acme["access_token"]), auth(globex["access_token"])

    acme_group = (await client.post("/groups", json={"name": "eng"}, headers=acme_h)).json()
    globex_admin_id = (await client.get("/auth/me", headers=globex_h)).json()["id"]

    # Globex can't see Acme's users or groups...
    globex_users = (await client.get("/users", headers=globex_h)).json()
    assert {u["email"] for u in globex_users} == {globex["email"]}
    assert (await client.get("/groups", headers=globex_h)).json() == []

    # ...or touch them (404, not 403, so their existence isn't revealed).
    url = f"/groups/{acme_group['id']}/members/{globex_admin_id}"
    assert (await client.put(url, headers=globex_h)).status_code == 404
    # Acme can't add a Globex user to its own group either.
    assert (await client.put(url, headers=acme_h)).status_code == 404


async def test_admin_cannot_deactivate_self(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    h = auth(admin["access_token"])
    me = (await client.get("/auth/me", headers=h)).json()
    assert (await client.post(f"/users/{me['id']}/deactivate", headers=h)).status_code == 400


async def test_created_user_can_log_in(client: httpx.AsyncClient) -> None:
    admin = await register(client)
    resp = await client.post(
        "/users",
        json={"email": "New.Person@Example.com", "password": PASSWORD},
        headers=auth(admin["access_token"]),
    )
    assert resp.status_code == 201
    assert resp.json()["email"] == "new.person@example.com"
    login = await client.post(
        "/auth/login", json={"email": "new.person@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200
