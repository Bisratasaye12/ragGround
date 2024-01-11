# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app /app/data
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    RAGGROUND_STORAGE_DIR=/app/data/blobs \
    FASTEMBED_CACHE_PATH=/app/data/models
EXPOSE 8000
# Migrations run on start for this single-instance setup; with several replicas, run them as a
# separate release step instead.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn ragground.main:app --host 0.0.0.0 --port 8000"]
