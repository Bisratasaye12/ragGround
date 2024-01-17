"""Request/response models for the public API."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from ragground.models import DocumentStatus, Role


def _normalise_email(value: str) -> str:
    return value.strip().lower()


class _EmailIn(BaseModel):
    email: EmailStr

    @field_validator("email", mode="after")
    @classmethod
    def _lower(cls, v: str) -> str:
        return _normalise_email(v)


class RegisterIn(_EmailIn):
    """Sign-up creates a new org with the caller as its first admin."""

    org_name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=12, max_length=256)


class LoginIn(_EmailIn):
    password: str = Field(min_length=1, max_length=256)


class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=512)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in: int


class GroupOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    email: str
    role: Role
    is_active: bool
    created_at: datetime
    groups: list[GroupOut] = []


class UserCreateIn(_EmailIn):
    password: str = Field(min_length=12, max_length=256)
    role: Role = Role.member


class GroupCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source_type: str
    source_uri: str
    filename: str | None
    size_bytes: int | None
    content_hash: str | None
    status: DocumentStatus
    error: str | None
    created_by: uuid.UUID | None
    created_at: datetime
    ingested_at: datetime | None
    groups: list[GroupOut] = []
    chunk_count: int = 0


class UploadResult(BaseModel):
    document: DocumentOut
    # "created" | "updated" (content changed → re-ingesting) | "unchanged" (same bytes)
    outcome: str


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["fulltext", "vector", "hybrid", "hybrid_rerank"] | None = None
    k: int = Field(default=5, ge=1, le=50)


class SearchHitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    source_uri: str
    heading_path: str
    text: str
    ord: int
    fulltext_rank: int | None
    vector_rank: int | None
    fulltext_score: float | None
    vector_similarity: float | None
    rrf_score: float | None
    rerank_score: float | None


class SearchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mode: str
    candidates: int
    timings_ms: dict[str, float]
    hits: list[SearchHitOut]


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["fulltext", "vector", "hybrid", "hybrid_rerank"] | None = None
    k: int | None = Field(default=None, ge=1, le=20)
    stream: bool = True


class QueryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID | None
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None
    error: str | None
    cited: list[int]
    invalid_citations: list[int]
    sources: list[dict[str, object]]
    model: str
    mode: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timings_ms: dict[str, float]
    created_at: datetime
