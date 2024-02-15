"""Grounded chat. `POST /chat` streams Server-Sent Events by default:

    event: sources  data: [{n, chunk_id, document_title, heading_path, text, rerank_score}, …]
    event: delta    data: {"text": "…"}          (repeated)
    event: done     data: {answer, cited, abstained, timings_ms, query_id, …}

The final `answer` in `done` is authoritative: invalid citation markers streamed earlier are
removed from it. With `"stream": false` the same result comes back as one JSON object.
"""

import dataclasses
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragground.config import Settings
from ragground.deps import CurrentUser, SessionDep, SettingsDep
from ragground.generation.answer import (
    AnswerResult,
    ChatConfig,
    DeltaEvent,
    DoneEvent,
    SourcesEvent,
    answer_question,
)
from ragground.models import Query, Role, User
from ragground.providers.llm import LLM
from ragground.retrieval.search import Retriever
from ragground.schemas import ChatIn, QueryOut

router = APIRouter(tags=["chat"])


def chat_config(settings: Settings, body: ChatIn) -> ChatConfig:
    return ChatConfig(
        mode=body.mode or settings.search_default_mode,
        k=body.k or settings.chat_k,
        abstain_below_rerank_score=settings.abstain_below_rerank_score,
        max_context_words=settings.chat_max_context_words,
        max_tokens=settings.chat_max_tokens,
    )


async def _save(
    sessionmaker: async_sessionmaker[AsyncSession], user: User, r: AnswerResult
) -> uuid.UUID:
    row = Query(
        org_id=user.org_id,
        user_id=user.id,
        question=r.question,
        answer=r.answer,
        abstained=r.abstained,
        abstain_reason=r.abstain_reason,
        error=r.error,
        sources=[dataclasses.asdict(s) for s in r.sources],
        cited=r.cited,
        invalid_citations=r.invalid_citations,
        model=r.model,
        mode=r.mode,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
        latency_ms=r.timings_ms.get("total", 0.0),
        timings_ms=r.timings_ms,
    )
    async with sessionmaker() as session:
        session.add(row)
        await session.commit()
        return row.id


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat", response_model=None)
async def chat(
    body: ChatIn, request: Request, user: CurrentUser, settings: SettingsDep
) -> StreamingResponse | dict[str, Any]:
    llm: LLM | None = request.app.state.llm
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No LLM provider configured")
    retriever: Retriever = request.app.state.retriever
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    cfg = chat_config(settings, body)
    events = answer_question(user, body.question, retriever, llm, cfg)

    if not body.stream:
        result: AnswerResult | None = None
        async for event in events:
            if isinstance(event, DoneEvent):
                result = event.result
        assert result is not None
        query_id = await _save(sessionmaker, user, result)
        return {**dataclasses.asdict(result), "query_id": str(query_id)}

    async def stream() -> AsyncIterator[str]:
        # Uses its own DB session: request-scoped dependencies may be torn down mid-stream.
        async for event in events:
            if isinstance(event, SourcesEvent):
                yield _sse("sources", [dataclasses.asdict(s) for s in event.sources])
            elif isinstance(event, DeltaEvent):
                yield _sse("delta", {"text": event.text})
            elif isinstance(event, DoneEvent):
                query_id = await _save(sessionmaker, user, event.result)
                yield _sse("done", {**dataclasses.asdict(event.result), "query_id": query_id})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/queries", response_model=list[QueryOut])
async def list_queries(user: CurrentUser, session: SessionDep, limit: int = 50) -> list[Query]:
    """Your own history; admins see the whole org's."""
    stmt = select(Query).where(Query.org_id == user.org_id)
    if user.role != Role.admin:
        stmt = stmt.where(Query.user_id == user.id)
    rows = await session.scalars(stmt.order_by(Query.created_at.desc()).limit(min(limit, 200)))
    return list(rows)
