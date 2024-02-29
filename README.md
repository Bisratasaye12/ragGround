# ragGround

Grounded answers over your documents: hybrid retrieval, document-level access control, cited
answers, and an evaluation harness that gates every change in CI.

> 🚧 Work in progress. See [PLAN.md](PLAN.md) for the roadmap and progress log.

## Measured, not assumed

Retrieval on a 120-case golden set over the FastAPI docs (97 answerable cases scored; ACL leak
rate **0** in every configuration). Methodology in [evals/README.md](evals/README.md);
decisions in [ADR-005](docs/decisions/ADR-005-chunking-and-rerank-from-evals.md).

| Retrieval mode (heading chunks ≤350 tokens) | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|
| Full-text only (Postgres `tsvector`, OR query) | 0.717 | 0.551 | 0.620 |
| Vector only (`bge-small`, pgvector HNSW) | 0.840 | 0.753 | 0.785 |
| Hybrid (RRF) | 0.887 | 0.739 | 0.789 |
| **Hybrid + cross-encoder rerank (default)** | **0.938** | **0.871** | **0.881** |

What the eval changed: measuring chunks with the model tokenizer showed 11% of word-limited chunks
silently overflowed the 512-token model limit. Switching to token-measured chunks raised
Recall@5 0.912 → 0.938 and MRR 0.844 → 0.871 while halving rerank latency.
Heading-aware chunks beat fixed windows (MRR 0.871 vs 0.755).

CI blocks any PR that lowers these numbers or leaks a restricted document
([evals.yml](.github/workflows/evals.yml)).

## Quickstart (local development)

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Node.js ≥ 20.19 (22 LTS recommended),
and [Ollama](https://ollama.com) with a model pulled (`ollama pull llama3.1:8b`).

```bash
cp api/.env.example api/.env

# Everything in Docker (API on :8000, web on :3000, Jaeger UI on :16686)
docker compose up --build

# Or: infrastructure in Docker, API and web on the host
docker compose up -d postgres redis
cd api && uv sync && uv run uvicorn ragground.main:app --reload   # http://localhost:8000
cd api && uv run arq ragground.worker.WorkerSettings               # ingestion worker
cd web && npm install && npm run dev                               # http://localhost:3000
```

Check readiness: `curl localhost:8000/health/ready` reports Postgres, Redis and the LLM provider.

Database migrations and demo data (when running the API on the host):

```bash
cd api && uv run alembic upgrade head && uv run python -m ragground.seed
# demo users: admin@ / alice@ / bob@ / carol@acme.example.com, password: ragground-demo-password
```

API docs: http://localhost:8000/docs

Upload a document (as any seeded user) and watch it get ingested:

```bash
TOKEN=$(curl -s localhost:8000/auth/login -H 'content-type: application/json' \
  -d '{"email":"alice@acme.example.com","password":"ragground-demo-password"}' | jq -r .access_token)
curl -s localhost:8000/documents -H "Authorization: Bearer $TOKEN" -F file=@some-doc.md | jq
curl -s localhost:8000/documents -H "Authorization: Bearer $TOKEN" | jq '.[] | {title, status, chunk_count}'
```

Search (debug view of every retriever's ranks and scores) and ask (streamed, cited answer):

```bash
curl -s localhost:8000/search -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"query":"what do I do after a security breach","mode":"hybrid_rerank"}' | jq '.hits[0]'
curl -N localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"question":"How much can I spend on training per year?"}'
```

`/chat` streams Server-Sent Events: `sources`, then `delta` tokens, then `done` with the validated
answer, citations, abstention reason, token counts and per-stage timings. Add `"stream": false`
for a single JSON response. History: `GET /queries`.

Tracing: set `RAGGROUND_OTEL_ENABLED=true`, run `docker compose up -d jaeger`, open
http://localhost:16686 (service `ragground-api`).

Supported: `.md` `.txt` `.pdf` `.html` `.docx` (max 25 MB). The first ingestion downloads the
local embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB).

Host ports: Postgres `5442`, Redis `6389` (non-default to avoid clashing with other local stacks).

## Development

```bash
cd api && uv run ruff check . && uv run mypy src && uv run pytest
cd web && npm run lint && npm run typecheck && npm run build
```

## Layout

| Path | What |
|---|---|
| `api/` | FastAPI service (Python 3.13, uv) |
| `web/` | Next.js 15 frontend |
| `docker-compose.yml` | Postgres + pgvector, Redis, Jaeger, API, ingestion worker, web |
| `.github/workflows/` | CI: lint, typecheck, migration check, tests, image builds |
| `docs/decisions/` | Architecture decision records |
