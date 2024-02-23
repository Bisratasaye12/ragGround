"""Build (or reuse) an isolated eval org whose documents are the eval corpus.

One org per chunking configuration, so the experiment matrix can compare chunkers without
re-ingesting every run. Documents get `source_uri = eval://<corpus-relative path>` (several
files are called index.md), and everything under the restricted prefix is visible only to the
`security-team` group.
"""

import secrets
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ragground.evals.golden import EvalUser
from ragground.ingestion.pipeline import IngestDeps, ingest_document
from ragground.models import Document, DocumentChunk, DocumentStatus, Group, Org, Role, User
from ragground.security import hash_password

RESTRICTED_PREFIX = "tutorial/security/"
URI_PREFIX = "eval://"


@dataclass
class CorpusFile:
    relpath: str
    data: bytes


@dataclass
class EvalOrg:
    org_id: str
    name: str
    users: dict[EvalUser, User]
    documents: int
    chunks: int
    reused: bool


def load_corpus(corpus_root: Path) -> list[CorpusFile]:
    """FastAPI docs live under corpus_root/fastapi/tutorial/…, synthetic docs under
    corpus_root/injected/…; both are addressed by the path the golden set uses."""
    files: list[CorpusFile] = []
    fastapi = corpus_root / "fastapi"
    if not (fastapi / "tutorial").is_dir():
        raise FileNotFoundError(
            f"{fastapi}/tutorial missing — run: python evals/corpus/fetch_fastapi_docs.py"
        )
    for p in sorted((fastapi / "tutorial").rglob("*.md")):
        files.append(CorpusFile(str(p.relative_to(fastapi)), p.read_bytes()))
    for p in sorted((corpus_root / "injected").glob("*.md")):
        files.append(CorpusFile(str(p.relative_to(corpus_root)), p.read_bytes()))
    return files


def relpath_of(source_uri: str) -> str:
    return source_uri.removeprefix(URI_PREFIX)


async def prepare_eval_org(
    deps: IngestDeps, corpus: list[CorpusFile], org_name: str, reingest: bool = False
) -> EvalOrg:
    sm = deps.sessionmaker
    async with sm() as session:
        org = await session.scalar(select(Org).where(Org.name == org_name))
        if org is not None and not reingest:
            n_docs = await session.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.org_id == org.id, Document.status == DocumentStatus.ready)
            )
            if n_docs == len(corpus):
                users = await _load_users(session, org)
                n_chunks = await session.scalar(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.org_id == org.id)
                )
                return EvalOrg(str(org.id), org_name, users, n_docs, n_chunks or 0, True)
        if org is not None:
            await session.delete(org)  # cascades to users, groups, documents, chunks
            await session.commit()

        org = Org(name=org_name)
        session.add(org)
        await session.flush()
        security = Group(org_id=org.id, name="security-team")
        tag = secrets.token_hex(4)  # emails are globally unique; one set per eval org
        pw = hash_password(secrets.token_urlsafe(24))  # never used to log in
        new_users: dict[EvalUser, User] = {
            "admin": User(
                org_id=org.id,
                email=f"eval-admin-{tag}@example.com",
                password_hash=pw,
                role=Role.admin,
                groups=[],
            ),
            "member": User(
                org_id=org.id,
                email=f"eval-member-{tag}@example.com",
                password_hash=pw,
                role=Role.member,
                groups=[security],
            ),
            "outsider": User(
                org_id=org.id,
                email=f"eval-outsider-{tag}@example.com",
                password_hash=pw,
                role=Role.member,
                groups=[],
            ),
        }
        session.add_all([security, *new_users.values()])
        docs = []
        for f in corpus:
            doc = Document(
                org_id=org.id,
                source_type="eval",
                source_uri=URI_PREFIX + f.relpath,
                title=Path(f.relpath).stem,
                filename=Path(f.relpath).name,
                size_bytes=len(f.data),
                groups=[security] if f.relpath.startswith(RESTRICTED_PREFIX) else [],
                ingest_version=1,
            )
            session.add(doc)
            docs.append((doc, f))
        await session.flush()
        for doc, f in docs:
            doc.storage_key = f"eval/{org.id}/{doc.id}"
            await deps.storage.put(doc.storage_key, f.data)
        await session.commit()
        org_id = org.id
        doc_ids = [d.id for d, _ in docs]

    failed = []
    for doc_id in doc_ids:
        result = await ingest_document(deps, doc_id, version=1)
        if result.status != "ready":
            failed.append((doc_id, result.error))
    if failed:
        raise RuntimeError(f"{len(failed)} eval documents failed to ingest: {failed[:3]}")

    async with sm() as session:
        org = await session.get(Org, org_id)
        assert org is not None
        users = await _load_users(session, org)
        n_chunks = await session.scalar(
            select(func.count()).select_from(DocumentChunk).where(DocumentChunk.org_id == org_id)
        )
    return EvalOrg(str(org_id), org_name, users, len(doc_ids), n_chunks or 0, False)


async def _load_users(session: AsyncSession, org: Org) -> dict[EvalUser, User]:
    """Eval users are recognised by their email pattern: eval-<kind>-<tag>@example.com."""
    rows = list(
        await session.scalars(
            select(User).where(User.org_id == org.id).options(selectinload(User.groups))
        )
    )
    by_kind: dict[EvalUser, User] = {}
    for u in rows:
        kind = u.email.split("-")[1]
        if kind in ("admin", "member", "outsider"):
            by_kind[kind] = u  # type: ignore[index]
    if set(by_kind) != {"admin", "member", "outsider"}:
        raise RuntimeError(f"eval org {org.name!r} is missing users; rerun with --reingest")
    return by_kind
