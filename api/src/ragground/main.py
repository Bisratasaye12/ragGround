"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from ragground import __version__
from ragground.config import Settings, get_settings
from ragground.db import make_engine, make_sessionmaker
from ragground.health import HealthCheck, ollama_check, postgres_check, redis_check
from ragground.health import router as health_router
from ragground.ingestion.chunkers import get_chunker
from ragground.ingestion.pipeline import IngestDeps, ingest_document
from ragground.ingestion.queue import ArqQueue, IngestQueue, InlineQueue
from ragground.ingestion.storage import LocalStorage
from ragground.log_config import configure_logging
from ragground.observability import setup_tracing
from ragground.providers.embeddings import make_embedder
from ragground.providers.llm import make_llm
from ragground.providers.reranker import make_reranker
from ragground.retrieval.search import Retriever, SearchConfig
from ragground.routers import auth, chat, documents, org, search

log = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    # Created eagerly (connections are lazy) so the app works with or without a lifespan run,
    # e.g. under httpx's ASGITransport in tests.
    engine = make_engine(settings.database_url)
    redis = Redis.from_url(settings.redis_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("startup", env=settings.env, llm_provider=settings.llm_provider)
        yield
        await queue.close()
        await redis.aclose()
        await engine.dispose()
        log.info("shutdown")

    app = FastAPI(title="ragGround API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessionmaker = make_sessionmaker(engine)
    app.state.redis = redis
    storage = LocalStorage(settings.storage_dir)
    app.state.storage = storage
    # Models load once per process (~1 s each on CPU) and are shared by search and inline ingest.
    embedder = make_embedder(settings.embedding_provider, settings.embedding_model)
    app.state.retriever = Retriever(
        app.state.sessionmaker,
        embedder,
        make_reranker(settings.reranker_provider, settings.reranker_model),
        SearchConfig(
            rerank_top_n=settings.rerank_top_n,
            rerank_max_words=settings.rerank_max_words,
            fulltext_operator=settings.fulltext_operator,
        ),
    )
    app.state.llm = make_llm(
        settings.llm_provider, settings.llm_model, settings.ollama_base_url, settings.llm_num_ctx
    )
    queue: IngestQueue
    if settings.ingest_inline:
        deps = IngestDeps(
            sessionmaker=app.state.sessionmaker,
            storage=storage,
            embedder=embedder,
            chunker=get_chunker(
                settings.chunker, settings.chunk_max_size, settings.chunk_size_unit
            ),
        )
        queue = InlineQueue(lambda doc_id, version: ingest_document(deps, doc_id, version))
    else:
        queue = ArqQueue(settings.redis_url)
    app.state.queue = queue
    checks: dict[str, HealthCheck] = {
        "postgres": postgres_check(engine),
        "redis": redis_check(redis),
    }
    if settings.llm_provider == "ollama":
        checks["llm"] = ollama_check(settings.ollama_base_url, settings.llm_model)
    app.state.health_checks = checks

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(auth.router)
    app.include_router(org.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    setup_tracing(app, settings)
    return app


app = create_app()
