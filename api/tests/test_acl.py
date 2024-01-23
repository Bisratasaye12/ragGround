"""The document ACL rule, tested directly against Postgres.

Scenario (two orgs):
  Acme: admin, alice (engineering), bob (hr), carol (no groups), dave (engineering, inactive)
  Docs: handbook (no groups → org-wide), design (engineering), salaries (hr),
        roadmap (engineering + hr)
  Globex: one org-wide doc that no Acme user may ever see.
"""

from dataclasses import dataclass

from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragground.acl import readable_by
from ragground.models import Document, Group, Org, Role, User


@dataclass
class World:
    users: dict[str, User]


async def _build_world(session: AsyncSession) -> World:
    acme, globex = Org(name="Acme"), Org(name="Globex")
    session.add_all([acme, globex])
    await session.flush()

    eng, hr = Group(org_id=acme.id, name="engineering"), Group(org_id=acme.id, name="hr")

    def user(
        name: str,
        org: Org,
        role: Role = Role.member,
        groups: list[Group] | None = None,
        active: bool = True,
    ) -> User:
        return User(
            org_id=org.id,
            email=f"{name}@{org.name.lower()}.test",
            password_hash="x",
            role=role,
            groups=groups or [],
            is_active=active,
        )

    users = {
        "admin": user("admin", acme, Role.admin),
        "alice": user("alice", acme, groups=[eng]),
        "bob": user("bob", acme, groups=[hr]),
        "carol": user("carol", acme),
        "dave": user("dave", acme, groups=[eng], active=False),
        "globex_admin": user("admin", globex, Role.admin),
    }

    def doc(title: str, org: Org, groups: list[Group]) -> Document:
        return Document(
            org_id=org.id,
            source_type="upload",
            source_uri=f"mem://{title}",
            title=title,
            groups=groups,
        )

    session.add_all(
        [
            eng,
            hr,
            *users.values(),
            doc("handbook", acme, []),
            doc("design", acme, [eng]),
            doc("salaries", acme, [hr]),
            doc("roadmap", acme, [eng, hr]),
            doc("globex-wiki", globex, []),
        ]
    )
    await session.commit()
    return World(users=users)


async def _readable(session: AsyncSession, user: User) -> set[str]:
    rows = await session.scalars(select(Document.title).where(readable_by(user)))
    return set(rows)


async def test_acl_rules(db_app: FastAPI) -> None:
    async with db_app.state.sessionmaker() as session:
        world = await _build_world(session)
        u = world.users
        expected = {
            "admin": {"handbook", "design", "salaries", "roadmap"},  # whole org, nothing else
            "alice": {"handbook", "design", "roadmap"},
            "bob": {"handbook", "salaries", "roadmap"},
            "carol": {"handbook"},  # no groups → only org-wide docs
            "dave": set(),  # inactive users see nothing
            "globex_admin": {"globex-wiki"},  # admin power stops at the org boundary
        }
        for name, titles in expected.items():
            assert await _readable(session, u[name]) == titles, name


async def test_acl_follows_group_membership_changes(db_app: FastAPI) -> None:
    async with db_app.state.sessionmaker() as session:
        world = await _build_world(session)
        carol = world.users["carol"]
        hr = await session.scalar(select(Group).where(Group.name == "hr"))
        assert hr is not None
        await session.refresh(carol, ["groups"])
        carol.groups.append(hr)
        await session.commit()
        assert await _readable(session, carol) == {"handbook", "salaries", "roadmap"}
