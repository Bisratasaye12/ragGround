"""Retrieval debug endpoint: see exactly what each retriever returned and how it was ranked.
The chat endpoint (M4) uses the same `Retriever`."""

from fastapi import APIRouter, Request

from ragground.deps import CurrentUser, SettingsDep
from ragground.retrieval.search import Retriever
from ragground.schemas import SearchIn, SearchOut

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchOut)
async def search(
    body: SearchIn, request: Request, user: CurrentUser, settings: SettingsDep
) -> SearchOut:
    retriever: Retriever = request.app.state.retriever
    result = await retriever.search(
        user, body.query, body.mode or settings.search_default_mode, k=body.k
    )
    return SearchOut.model_validate(result)
