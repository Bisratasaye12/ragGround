"""Document upload and management.

Permissions:
- Anyone in the org can upload. Members may only restrict a document to groups they belong to;
  admins to any group in the org. No groups means org-wide.
- Listing and reading go through the ACL predicate (`acl.readable_by`).
- Replacing (same filename), deleting and reindexing need admin or the original uploader.
"""

import hashlib
import uuid
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ragground.acl import readable_by
from ragground.deps import CurrentUser, SessionDep, SettingsDep
from ragground.ingestion.parsers import SUPPORTED_EXTENSIONS
from ragground.ingestion.queue import IngestQueue
from ragground.ingestion.storage import BlobStorage
from ragground.models import Document, DocumentChunk, DocumentStatus, Group, Role, User
from ragground.schemas import DocumentOut, UploadResult

router = APIRouter(prefix="/documents", tags=["documents"])

_NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, "Not found")


def _storage(request: Request) -> BlobStorage:
    storage: BlobStorage = request.app.state.storage
    return storage


def _queue(request: Request) -> IngestQueue:
    queue: IngestQueue = request.app.state.queue
    return queue


def _safe_filename(raw: str | None) -> str:
    name = PurePosixPath((raw or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Missing filename")
    return name[:255]


def _can_modify(user: User, doc: Document) -> bool:
    return user.role == Role.admin or doc.created_by == user.id


async def _chunk_counts(session: AsyncSession, ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    if not ids:
        return {}
    rows = await session.execute(
        select(DocumentChunk.document_id, func.count())
        .where(DocumentChunk.document_id.in_(ids))
        .group_by(DocumentChunk.document_id)
    )
    return {doc_id: n for doc_id, n in rows.all()}


async def _out(session: AsyncSession, doc: Document) -> DocumentOut:
    counts = await _chunk_counts(session, [doc.id])
    out = DocumentOut.model_validate(doc)
    out.chunk_count = counts.get(doc.id, 0)
    return out


async def _resolve_groups(
    session: AsyncSession, user: User, group_ids: list[uuid.UUID]
) -> list[Group]:
    if not group_ids:
        return []
    groups = list(
        await session.scalars(
            select(Group).where(Group.id.in_(group_ids), Group.org_id == user.org_id)
        )
    )
    if len(groups) != len(set(group_ids)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown group")
    if user.role != Role.admin:
        mine = {g.id for g in user.groups}
        if any(g.id not in mine for g in groups):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "You can only restrict documents to your own groups"
            )
    return groups


async def _readable_doc(session: AsyncSession, user: User, doc_id: uuid.UUID) -> Document:
    doc = await session.scalar(
        select(Document)
        .where(Document.id == doc_id, readable_by(user))
        .options(selectinload(Document.groups))
    )
    if doc is None:
        raise _NOT_FOUND
    return doc


@router.post("", response_model=UploadResult)
async def upload_document(
    request: Request,
    response: Response,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    group_ids: Annotated[list[uuid.UUID] | None, Form()] = None,
) -> UploadResult:
    filename = _safe_filename(file.filename)
    if PurePosixPath(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Max upload is {settings.max_upload_mb} MB"
        )
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Empty file")

    groups = await _resolve_groups(session, user, group_ids or [])
    content_hash = hashlib.sha256(data).hexdigest()
    source_uri = f"upload://{filename}"

    doc = await session.scalar(
        select(Document)
        .where(Document.org_id == user.org_id, Document.source_uri == source_uri)
        .options(selectinload(Document.groups))
        .with_for_update()
    )
    if doc is not None:
        if not _can_modify(user, doc):
            # Don't reveal whether the existing document is readable to this user.
            raise HTTPException(
                status.HTTP_409_CONFLICT, "A document with this filename already exists"
            )
        doc.groups = groups  # visibility always follows the latest upload
        if doc.content_hash == content_hash and doc.status != DocumentStatus.failed:
            # Same bytes: no re-ingest (idempotent re-upload). A failed doc is retried instead.
            await session.commit()
            response.status_code = status.HTTP_200_OK
            return UploadResult(document=await _out(session, doc), outcome="unchanged")
        outcome = "updated"
    else:
        doc = Document(
            org_id=user.org_id,
            source_type="upload",
            source_uri=source_uri,
            title=PurePosixPath(filename).stem,
            created_by=user.id,
            groups=groups,
        )
        session.add(doc)
        await session.flush()
        outcome = "created"

    old_key = doc.storage_key
    doc.filename = filename
    doc.size_bytes = len(data)
    doc.content_hash = content_hash
    doc.storage_key = f"{user.org_id}/{doc.id}/{content_hash}"
    doc.status = DocumentStatus.pending
    doc.error = None
    doc.ingest_version += 1
    version = doc.ingest_version

    storage = _storage(request)
    await storage.put(doc.storage_key, data)
    await session.commit()
    if old_key and old_key != doc.storage_key:
        await storage.delete(old_key)

    await _queue(request).enqueue_ingest(doc.id, version)
    # The job may have run (inline mode) and changed status; reload columns and groups.
    await session.refresh(doc)
    await session.refresh(doc, ["groups"])
    response.status_code = (
        status.HTTP_201_CREATED if outcome == "created" else status.HTTP_202_ACCEPTED
    )
    return UploadResult(document=await _out(session, doc), outcome=outcome)


@router.get("", response_model=list[DocumentOut])
async def list_documents(user: CurrentUser, session: SessionDep) -> list[DocumentOut]:
    docs = list(
        await session.scalars(
            select(Document)
            .where(readable_by(user))
            .options(selectinload(Document.groups))
            .order_by(Document.created_at.desc())
        )
    )
    counts = await _chunk_counts(session, [d.id for d in docs])
    out = []
    for d in docs:
        item = DocumentOut.model_validate(d)
        item.chunk_count = counts.get(d.id, 0)
        out.append(item)
    return out


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> DocumentOut:
    return await _out(session, await _readable_doc(session, user, doc_id))


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    request: Request, doc_id: uuid.UUID, user: CurrentUser, session: SessionDep
) -> None:
    doc = await _readable_doc(session, user, doc_id)
    if not _can_modify(user, doc):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the uploader or an admin can delete")
    key = doc.storage_key
    await session.delete(doc)  # chunks cascade
    await session.commit()
    if key:
        await _storage(request).delete(key)


@router.post("/{doc_id}/reindex", response_model=DocumentOut, status_code=202)
async def reindex_document(
    request: Request, doc_id: uuid.UUID, user: CurrentUser, session: SessionDep
) -> DocumentOut:
    """Re-run ingestion, e.g. after changing the chunker or embedding model."""
    doc = await _readable_doc(session, user, doc_id)
    if not _can_modify(user, doc):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the uploader or an admin can reindex")
    doc.ingest_version += 1
    doc.status = DocumentStatus.pending
    doc.error = None
    version = doc.ingest_version
    await session.commit()
    await _queue(request).enqueue_ingest(doc.id, version)
    # The job may have run (inline mode) and changed status; reload columns and groups.
    await session.refresh(doc)
    await session.refresh(doc, ["groups"])
    return await _out(session, doc)
