"""Grounded question answering: retrieve → decide whether to abstain → stream → validate.

Yields events so the HTTP layer can stream them (SSE) and the eval harness can consume the same
code path non-streamingly. Abstaining *before* calling the LLM when retrieval is weak saves the
most expensive step and removes the main hallucination opportunity.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from opentelemetry import trace

from ragground.generation.citations import CitationCheck, check_citations, is_abstention
from ragground.generation.prompts import (
    ABSTAIN_ANSWER,
    SYSTEM_PROMPT,
    PromptSource,
    build_user_message,
)
from ragground.models import User
from ragground.providers.llm import LLM, LLMError, Message, TextDelta, Usage, cost_usd
from ragground.retrieval.search import Hit, Retriever, SearchMode

tracer = trace.get_tracer("ragground.generation")


@dataclass
class ChatConfig:
    mode: SearchMode = "hybrid_rerank"
    k: int = 5
    # Cross-encoder logit below which the best chunk is treated as "not relevant enough".
    # ms-marco MiniLM: relevant ≈ 0…+10, unrelated ≈ -11. Calibrated in M5.
    abstain_below_rerank_score: float = -2.0
    max_context_words: int = 1500  # prompt-size budget; CPU prompt processing is the bottleneck
    max_tokens: int = 400


@dataclass
class Source:
    n: int
    chunk_id: str
    document_id: str
    document_title: str
    source_uri: str
    heading_path: str
    text: str
    rerank_score: float | None


@dataclass
class AnswerResult:
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None  # no_results | low_retrieval_confidence | model_declined
    sources: list[Source]
    cited: list[int]
    invalid_citations: list[int]
    uncited: bool
    model: str
    mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class SourcesEvent:
    sources: list[Source]


@dataclass
class DeltaEvent:
    text: str


@dataclass
class DoneEvent:
    result: AnswerResult


ChatEvent = SourcesEvent | DeltaEvent | DoneEvent


def _select_sources(hits: list[Hit], max_words: int, min_rerank_score: float) -> list[Source]:
    """Keep hits in rank order until the context budget is spent. After the first, drop hits the
    reranker scored as irrelevant: on CPU every prompt token costs time, and unrelated context
    invites the model to pad its answer with it."""
    out: list[Source] = []
    used = 0
    for h in hits:
        words = len(h.text.split())
        if out and used + words > max_words:
            break
        if out and h.rerank_score is not None and h.rerank_score < min_rerank_score:
            continue
        used += words
        out.append(
            Source(
                n=len(out) + 1,
                chunk_id=str(h.chunk_id),
                document_id=str(h.document_id),
                document_title=h.document_title,
                source_uri=h.source_uri,
                heading_path=h.heading_path,
                text=h.text,
                rerank_score=h.rerank_score,
            )
        )
    return out


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def answer_question(
    user: User, question: str, retriever: Retriever, llm: LLM, cfg: ChatConfig
) -> AsyncIterator[ChatEvent]:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    with tracer.start_as_current_span("rag.retrieve") as span:
        span.set_attribute("rag.mode", cfg.mode)
        result = await retriever.search(user, question, cfg.mode, k=cfg.k)
        span.set_attribute("rag.hits", len(result.hits))
    timings["retrieval"] = _ms(t0)
    for stage, ms in result.timings_ms.items():
        timings[f"retrieval.{stage}"] = ms

    sources = _select_sources(result.hits, cfg.max_context_words, cfg.abstain_below_rerank_score)
    yield SourcesEvent(sources)

    def finish(
        answer: str,
        *,
        abstain_reason: str | None = None,
        check: CitationCheck | None = None,
        usage: Usage | None = None,
        error: str | None = None,
    ) -> DoneEvent:
        timings["total"] = _ms(t0)
        usage = usage or Usage(0, 0)
        return DoneEvent(
            AnswerResult(
                question=question,
                answer=answer,
                abstained=abstain_reason is not None,
                abstain_reason=abstain_reason,
                sources=sources,
                cited=check.cited if check else [],
                invalid_citations=check.invalid if check else [],
                uncited=check.has_uncited_text if check else False,
                model=f"{llm.provider}:{llm.model}",
                mode=result.mode,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost_usd(llm.provider, llm.model, usage),
                timings_ms=timings,
                error=error,
            )
        )

    top = sources[0].rerank_score if sources else None
    reason = None
    if not sources:
        reason = "no_results"
    elif top is not None and top < cfg.abstain_below_rerank_score:
        reason = "low_retrieval_confidence"
    if reason is not None:
        yield DeltaEvent(ABSTAIN_ANSWER)
        yield finish(ABSTAIN_ANSWER, abstain_reason=reason)
        return

    prompt_sources = [PromptSource(s.n, s.document_title, s.heading_path, s.text) for s in sources]
    messages = [Message("user", build_user_message(question, prompt_sources))]
    parts: list[str] = []
    usage = Usage(0, 0)
    t_gen = time.perf_counter()
    with tracer.start_as_current_span("rag.generate") as span:
        span.set_attribute("llm.model", llm.model)
        span.set_attribute("rag.sources", len(sources))
        try:
            async for event in llm.stream(
                SYSTEM_PROMPT, messages, max_tokens=cfg.max_tokens, temperature=0.0
            ):
                if isinstance(event, TextDelta):
                    if not parts:
                        timings["time_to_first_token"] = _ms(t_gen)
                    parts.append(event.text)
                    yield DeltaEvent(event.text)
                else:
                    usage = event
        except LLMError as exc:
            span.record_exception(exc)
            timings["generation"] = _ms(t_gen)
            yield finish("", error=str(exc))
            return
        span.set_attribute("llm.input_tokens", usage.input_tokens)
        span.set_attribute("llm.output_tokens", usage.output_tokens)
    timings["generation"] = _ms(t_gen)

    check = check_citations("".join(parts), len(sources))
    if is_abstention(check.answer):
        yield finish(ABSTAIN_ANSWER, abstain_reason="model_declined", usage=usage)
    else:
        yield finish(check.answer, check=check, usage=usage)
