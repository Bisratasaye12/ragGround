"""Application settings, loaded from environment variables (and `.env` in development)."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_DEV_SECRET = "dev-insecure-secret-change-me-before-deploying"  # noqa: S105 - deliberate dev-only default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAGGROUND_", extra="ignore")

    env: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    database_url: str = "postgresql+asyncpg://ragground:ragground@localhost:5442/ragground"
    redis_url: str = "redis://localhost:6389/0"

    # Auth
    jwt_secret: str = INSECURE_DEV_SECRET
    access_token_ttl_s: int = 15 * 60
    refresh_token_ttl_s: int = 7 * 24 * 3600

    # Ingestion
    storage_dir: str = "./data/blobs"
    max_upload_mb: int = 25
    chunker: Literal["heading", "fixed"] = "heading"
    # Chunk size limit, measured in `chunk_size_unit`. "tokens" uses the embedding tokenizer.
    chunk_max_size: int = 350
    chunk_size_unit: Literal["words", "tokens"] = "tokens"  # ADR-005
    embedding_provider: str = "fastembed"  # "fastembed" (local) | "hash" (tests only)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    ingest_max_tries: int = 3

    # Retrieval
    reranker_provider: Literal["fastembed", "overlap", "none"] = "fastembed"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    search_default_mode: Literal["fulltext", "vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    rerank_top_n: int = 10  # ADR-005: same quality as 20 at about half the latency
    rerank_max_words: int | None = None
    fulltext_operator: Literal["and", "or"] = "or"
    # Run ingestion inside the API process instead of the arq worker (tests, quick demos).
    ingest_inline: bool = False

    # LLM provider: "ollama" (default, local) | "none". Hosted adapters plug into providers/llm.py.
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"

    # Chat. Defaults are sized for CPU-only Ollama, where prompt processing dominates latency.
    chat_k: int = 5
    chat_max_context_words: int = 1200
    chat_max_tokens: int = 400
    abstain_below_rerank_score: float = -2.0  # calibrated in M5
    llm_num_ctx: int = 8192

    # Observability: OTLP/HTTP traces (Jaeger in docker-compose listens on 4318).
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "ragground-api"

    cors_origins: list[str] = ["http://localhost:3000"]

    @model_validator(mode="after")
    def _require_real_secret_outside_dev(self) -> Self:
        if self.env not in {"development", "test"} and (
            self.jwt_secret == INSECURE_DEV_SECRET or len(self.jwt_secret) < 32
        ):
            raise ValueError("RAGGROUND_JWT_SECRET must be set to a random value of 32+ chars")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
