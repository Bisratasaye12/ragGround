"""Seed a demo org for local development: `uv run python -m ragground.seed`.

Creates org "Acme Demo" with groups engineering + hr and four users (password for all of them is
DEMO_PASSWORD below). Idempotent: does nothing if the demo admin already exists.
"""

import asyncio

from sqlalchemy import select

from ragground.config import get_settings
from ragground.db import make_engine, make_sessionmaker
from ragground.models import Group, Org, Role, User
from ragground.security import hash_password

DEMO_PASSWORD = "ragground-demo-password"  # noqa: S105 - local demo data only
ADMIN_EMAIL = "admin@acme.example.com"


async def seed() -> None:
    settings = get_settings()
    if settings.env not in {"development", "test"}:
        raise SystemExit("Refusing to seed demo users outside development")
    engine = make_engine(settings.database_url)
    async with make_sessionmaker(engine)() as session:
        if await session.scalar(select(User).where(User.email == ADMIN_EMAIL)):
            print("Demo data already present; nothing to do.")
            await engine.dispose()
            return
        org = Org(name="Acme Demo")
        session.add(org)
        await session.flush()
        eng = Group(org_id=org.id, name="engineering")
        hr = Group(org_id=org.id, name="hr")
        pw = hash_password(DEMO_PASSWORD)
        people = [
            ("admin@acme.example.com", Role.admin, []),
            ("alice@acme.example.com", Role.member, [eng]),
            ("bob@acme.example.com", Role.member, [hr]),
            ("carol@acme.example.com", Role.member, []),
        ]
        session.add_all([eng, hr])
        for email, role, groups in people:
            session.add(
                User(org_id=org.id, email=email, password_hash=pw, role=role, groups=groups)
            )
        await session.commit()
    await engine.dispose()
    print(f"Seeded 'Acme Demo': {', '.join(p[0] for p in people)} (password: {DEMO_PASSWORD})")


if __name__ == "__main__":
    asyncio.run(seed())
