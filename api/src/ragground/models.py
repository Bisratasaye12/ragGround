"""ORM models: tenancy (orgs, users, groups), auth (refresh tokens) and document ACLs.

Access rule (see `acl.py`): a user can read a document when it belongs to the user's org AND
either the document has no groups (org-wide) or the user is in at least one of its groups.
Org admins can read every document in their org.
"""

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ragground.db import Base
from ragground.providers.embeddings import EMBEDDING_DIM


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Role(enum.StrEnum):
    admin = "admin"
    member = "member"


class DocumentStatus(enum.StrEnum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


user_groups = Table(
    "user_groups",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    # PK leads with the other column; ACL lookups by group need their own index.
    Column("group_id", ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True, index=True),
)

document_groups = Table(
    "document_groups",
    Base.metadata,
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    # PK leads with the other column; ACL lookups by group need their own index.
    Column("group_id", ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True, index=True),
)


class Org(Base):
    __tablename__ = "orgs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = _created_at()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    # Stored lower-cased; uniqueness is global (one user belongs to one org).
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="user_role", values_callable=lambda e: [m.value for m in e]),
        default=Role.member,
    )
    is_active: Mapped[bool] = mapped_column(default=True, server_default=sql_text("true"))
    created_at: Mapped[datetime] = _created_at()

    groups: Mapped[list["Group"]] = relationship(secondary=user_groups, back_populates="members")


class Group(Base):
    __tablename__ = "groups"
    __table_args__ = (UniqueConstraint("org_id", "name"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = _created_at()

    members: Mapped[list[User]] = relationship(secondary=user_groups, back_populates="groups")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("org_id", "source_uri"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(32))  # upload | github | url
    source_uri: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    storage_key: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    # Bumped on every (re)ingest request; a worker only commits if its job is still current.
    ingest_version: Mapped[int] = mapped_column(Integer, default=0, server_default=sql_text("0"))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=DocumentStatus.pending,
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = _created_at()
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    groups: Mapped[list[Group]] = relationship(secondary=document_groups)


class DocumentChunk(Base):
    """A retrievable unit of a document. `org_id` is denormalised from the document so retrieval
    can pre-filter by tenant without a join; the ACL predicate still joins to documents."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_ord", "document_id", "ord", unique=True),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text, server_default=sql_text("''"))
    word_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Full-text index over headings + body, maintained by Postgres itself.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', heading_path || ' ' || text)", persisted=True),
    )
    meta: Mapped[dict[str, object]] = mapped_column(
        "metadata", JSONB, server_default=sql_text("'{}'::jsonb")
    )


class RefreshToken(Base):
    """Opaque refresh tokens, stored hashed. Rotated on every use; tokens in one login session
    share a `family_id` so that reuse of a rotated token revokes the whole family."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()


class Query(Base):
    """One answered question: what was asked, what was retrieved and cited, cost and latency.
    Feeds query history, cost dashboards and (later) new golden-set cases from feedback."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column()
    abstain_reason: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list[dict[str, object]]] = mapped_column(JSONB)
    cited: Mapped[list[int]] = mapped_column(JSONB)
    invalid_citations: Mapped[list[int]] = mapped_column(JSONB)
    model: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(32))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column()
    latency_ms: Mapped[float] = mapped_column()
    timings_ms: Mapped[dict[str, float]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()
