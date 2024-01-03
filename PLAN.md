# Project 1 — Production RAG Platform with Eval Harness

> Working name: **ragGround** (rename freely). "Chat with your company's docs" — but built and
> measured like a production system.
> Part of the 6-project portfolio — see [../PORTFOLIO.md](../PORTFOLIO.md).

## 1. Why this project

Every AI-portfolio guide lists a RAG app, but almost nobody **measures** it. Hiring managers look
for: retrieval quality, groundedness/faithfulness, a golden test set, LLM-as-judge, and a CI
regression gate. This repo's headline is its **eval table**, not its chat UI.

## 2. Scope

### Must have (MVP → portfolio-ready)
- **Ingestion:** file upload (PDF, Markdown, HTML, DOCX, TXT) + GitHub repo connector (Markdown/docs
  folders) + URL crawler. Pluggable `Connector` interface.
- **Async pipeline:** parse → clean → chunk → embed → index, run by a background worker with
  retries, status tracking, and idempotent re-ingestion (content hashing).
- **Hybrid retrieval:** Postgres full-text (BM25-style `ts_rank`) + pgvector cosine, fused with
  Reciprocal Rank Fusion, then optional **reranker** (cross-encoder).
- **Grounded answering:** streaming answers with **inline citations** that link to source chunks;
  says "I don't know" when retrieval confidence is low.
- **Multi-tenant + ACL:** orgs, users, roles; per-document access groups. Retrieval filters by
  ACL *before* ranking (users can never retrieve chunks they can't read).
- **Eval harness:** golden set, retrieval + generation metrics, LLM judge, CLI, CI gate.
- **Experiments:** compare chunking strategies × retrieval modes; publish the table in README.
- **Observability:** OpenTelemetry traces per request (retrieve, rerank, generate spans),
  token + cost accounting per request, latency p50/p95.
- **Guardrails:** retrieved content treated strictly as data (prompt-injection-in-docs defense +
  test cases), optional PII redaction at ingestion.
- **Deploy:** `docker compose up` locally; public demo deployment on a safe public corpus.

### Nice to have
- Notion / Confluence connectors.
- Conversation memory + query rewriting (multi-turn).
- Feedback buttons (👍/👎) feeding back into the golden set.
- Ragas comparison column next to the in-house metrics.
- Export traces to Project #5 later.

### Out of scope
- Fine-tuning (that's #6). Agentic multi-step tool use (that's #2).

## 3. Architecture

```
            ┌──────────────┐  SSE stream   ┌─────────────────────────────┐
  Browser ─▶│  Next.js web │──────────────▶│        FastAPI API           │
            │ (chat, docs, │◀──────────────│ auth · orgs · ACL · /chat    │
            │  eval views) │               │ /documents · /evals          │
            └──────────────┘               └──────┬──────────┬────────────┘
                                                  │enqueue   │ query
                                          ┌───────▼──┐   ┌───▼────────────────────┐
                                          │  Redis   │   │ Postgres 18 + pgvector 0.8  │
                                          │ (queue)  │   │ docs · chunks · tsvector│
                                          └───────┬──┘   │ embeddings · users · ACL│
                                                  │      │ eval runs · traces/cost │
                                          ┌───────▼──────┴──┐                      │
                                          │ Ingestion worker │────────────────────▶┘
                                          │ parse/chunk/embed│
                                          └──────────────────┘
   Providers (pluggable): LLM (default Claude) · Embeddings (local fastembed default, Voyage/OpenAI optional)
   · Reranker (local cross-encoder default, Cohere optional)
   OTel → Jaeger (local) / any OTLP backend
```

### Retrieval pipeline
1. Query → (optional rewrite) → embed.
2. ACL filter (org + allowed groups) applied in SQL.
3. Top-k₁ from vector search + top-k₁ from full-text search.
4. RRF fusion → top-k₂ → reranker → top-k₃ context chunks.
5. Confidence check (reranker score threshold) → answer or abstain.
6. Prompt with numbered sources; model must cite `[n]`; server validates citations exist.

## 4. Tech stack (decided defaults — change before M1 if you disagree)

| Layer | Choice | Why |
|---|---|---|
| Backend | Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy 2 (async), Alembic | AI-ecosystem language; typed |
| Package mgmt | `uv` | fast, lockfile |
| DB | Postgres 18 + pgvector 0.8 (HNSW index) + built-in full-text | one store for hybrid search + ACL joins |
| Queue/worker | Redis + `arq` | simple async jobs, retries |
| Parsing | `pymupdf` (PDF), `markdown-it`, `trafilatura` (HTML), `python-docx` | |
| Embeddings | `fastembed` (bge-small, local, free) default; Voyage/OpenAI adapters | zero-cost demo, swappable |
| Reranker | local cross-encoder `Xenova/ms-marco-MiniLM-L-12-v2` via fastembed (~120 MB; `bge-reranker-base` is 1.04 GB — too big/slow here, kept as an M6 option); Cohere adapter later | ADR-004 |
| LLM | **Ollama (local) is the current default** — `llama3.1:8b` (also installed: `mistral-nemo:12b`), via Ollama's HTTP API on `localhost:11434` (`host.docker.internal` from containers). Provider interface keeps Anthropic (Claude) and OpenAI-compatible adapters as drop-in options for stronger published eval numbers later. Judge model configurable separately. | free, no API key; user decision 2026-11-06 |
| Frontend | Next.js 15 (App Router), TypeScript, Tailwind, shadcn/ui, pnpm | |
| Auth | Backend-issued JWT (access + refresh), argon2 password hashing | owns the ACL logic |
| Observability | OpenTelemetry SDK, Jaeger in compose, structlog JSON logs | |
| Tests | pytest + pytest-asyncio + testcontainers (real Postgres); Playwright for web e2e | |
| CI | GitHub Actions: lint (ruff), typecheck (mypy/pyright, tsc), tests, **eval gate** | |
| Deploy | Docker images; Fly.io or Railway (decide at M8); optional Terraform | |

## 5. Data model (initial)

- `orgs(id, name)`
- `users(id, org_id, email, password_hash, role[admin|member])`
- `groups(id, org_id, name)`, `user_groups(user_id, group_id)`
- `documents(id, org_id, source_type, source_uri, title, content_hash, status, error, created_at)`
- `document_groups(document_id, group_id)` — empty = visible to whole org
- `chunks(id, document_id, org_id, ord, text, heading_path, token_count, tsv tsvector, embedding vector(dim), metadata jsonb)`
- `queries(id, org_id, user_id, question, answer, citations jsonb, retrieved jsonb, latency_ms, tokens_in, tokens_out, cost_usd, created_at)`
- `feedback(query_id, rating, comment)`
- `eval_runs(id, git_sha, config jsonb, metrics jsonb, created_at)`, `eval_results(run_id, case_id, scores jsonb, output jsonb)`

## 6. Evaluation design (the headline)

### Demo corpus
A public, license-friendly docs set (candidate: **PostgreSQL docs subset** or **FastAPI docs**;
decide at M5). Safe to deploy publicly.

### Golden set (`evals/golden/*.jsonl`, target ≥ 100 cases)
Each case: `id, question, expected_answer, relevant_chunk_ids|source_refs, category, acl_user`.
Categories:
- factual single-hop, multi-hop (needs 2+ chunks), numeric/config lookup
- **unanswerable** (must abstain)
- **ACL** (answer exists but user lacks access → must not leak)
- **prompt-injection** (a doc contains "ignore previous instructions…") → must not comply
Creation: LLM-drafted from corpus, then human-reviewed (document the process in README).

### Metrics
| Retrieval | Generation | System |
|---|---|---|
| Recall@k, MRR, nDCG@10 | Faithfulness (claims supported by context, LLM judge) | p50/p95 latency |
| | Answer correctness vs expected (LLM judge, 1–5) | tokens + $ per query |
| | Citation accuracy (cited chunk supports claim) | |
| | Abstention accuracy (unanswerable), ACL leak rate (must be 0), injection resistance | |

Judge calibration: hand-label ~30 cases and report judge–human agreement in README.

### Experiments matrix (results table in README)
- Chunking: fixed-512 tokens w/ overlap · recursive · heading-aware (Markdown structure) · semantic
- Retrieval: BM25 only · vector only · hybrid (RRF) · hybrid + rerank
- Report each cell: Recall@5, faithfulness, latency, cost.

### CI gate
`ragground-eval run --suite smoke` (≈20 cases, cheap) on every PR; full suite nightly/manual.
Fails if any metric drops > configured tolerance vs `evals/baseline.json`, or ACL leak rate > 0.
Posts a markdown metrics table as a PR comment.

## 7. Repo layout (planned)

```
01-rag-platform/
  PLAN.md
  README.md
  docker-compose.yml          # postgres(pgvector), redis, api, worker, web, jaeger
  .env.example
  api/                        # FastAPI app (uv project)
    pyproject.toml
    src/ragground/
      main.py  config.py  db.py  models.py  auth/  routers/
      ingestion/ (connectors/, parsers/, chunkers/, pipeline.py)
      retrieval/ (fulltext.py, vector.py, fusion.py, rerank.py)
      generation/ (prompts.py, answer.py, citations.py)
      providers/ (llm.py, embeddings.py, reranker.py)
      observability.py  worker.py
    migrations/  tests/
  evals/                      # eval harness package + golden sets + baseline
    golden/  baseline.json  src/  reports/
  web/                        # Next.js app
  .github/workflows/ (ci.yml, evals.yml)
  docs/ (architecture.md, decisions/ADR-*.md)
```

## 8. Milestones

Each milestone ends with: tests green, short demo, Progress Log updated.

- [x] **M0 — Skeleton:** repo layout, compose (pgvector, redis, jaeger), FastAPI health endpoint,
  Next.js shell, ruff/mypy/tsc, CI running lint+tests. *Accept:* `docker compose up` → health OK; CI green.
- [x] **M1 — Data + auth:** models, Alembic migrations, orgs/users/groups, JWT auth, seed script.
  *Accept:* register/login tests; ACL model unit tests.
- [x] **M2 — Ingestion pipeline:** upload API, parsers, heading-aware + fixed chunkers, local
  embeddings, worker with retries, content-hash dedupe, status endpoint.
  *Accept:* ingest a folder of Markdown/PDF; chunks + embeddings in DB; re-ingest is a no-op.
- [x] **M3 — Hybrid retrieval:** full-text + vector + RRF + reranker, ACL filter in SQL,
  `/search` debug endpoint. *Accept:* tests prove ACL filtering; retrieval returns sensible hits.
- [x] **M4 — Grounded answering:** `/chat` SSE streaming, citation prompt + validation,
  abstain logic, token/cost accounting, OTel spans. (Ollama local; no key)
  *Accept:* cited streamed answers; traces visible in Jaeger.
- [ ] **M5 — Eval harness:** golden set (≥100), metrics, LLM judge, CLI, markdown/JSON reports,
  judge calibration. (Ollama judge; optionally a stronger API judge)
- [ ] **M6 — Experiments + CI gate:** run chunking × retrieval matrix, commit results, baseline,
  GitHub Action eval gate with PR comment. ⚠️ *Open decision:* GitHub's CPU runners are too slow
  for an 8B Ollama model. Options: (a) run a tiny model (e.g. 1–3B) in CI for the smoke suite,
  (b) a cheap hosted API key as a GitHub secret, (c) a self-hosted runner on this machine. Decide at M6.
- [ ] **M7 — Web UI:** login, doc upload + ingestion status, chat with clickable citations
  (side panel shows source chunk), eval-runs dashboard, query history with cost/latency.
  Playwright e2e.
- [ ] **M8 — Connectors + guardrails + deploy:** GitHub repo + URL connectors, injection test
  suite, optional PII redaction, deploy demo. 🔑 *Needs deploy account.*
- [ ] **M9 — Polish:** README (diagram, eval table, GIF), ADRs, security review, load test
  (k6 or locust) with p95 numbers.

## 9. What Claude needs from the user (asked at the milestone that needs it)

| When | Item |
|---|---|
| M4 | Nothing required: Ollama runs locally. Ollama must be running (`ollama serve`). Optional later: Anthropic/OpenAI key for stronger eval numbers |
| M2/M5 | (optional) Your own sample docs to test with; otherwise the public demo corpus |
| M2 (optional) | Embeddings API key (Voyage/OpenAI) if you want hosted embeddings vs local |
| M6 | Create GitHub repo + add API key as Actions secret; confirm budget for CI eval runs |
| M8 | Deploy target account (Fly.io/Railway/VPS) + domain (optional) |
| M8 | GitHub token for the repo connector demo (read-only, public repos fine) |

## 10. Interview talking points to own
- Why hybrid + RRF instead of vector-only; what the experiments showed.
- Why ACL filtering must happen before ranking (and how you tested leak rate = 0).
- How you validated the LLM judge against human labels.
- Chunking tradeoffs (heading-aware vs fixed vs semantic) with your numbers.
- Cost/latency budget per query and where time goes (trace screenshot).
- Prompt-injection-in-documents threat model.

## 11. Decisions log (ADRs)
Full text in [docs/decisions/](docs/decisions/).
- ADR-001: Postgres+pgvector as single store (vs dedicated vector DB) — keeps ACL joins and
  hybrid search in one query; revisit beyond ~10M chunks.
- ADR-002: 15-min JWT access tokens (user reloaded per request) + rotating opaque refresh tokens
  with family-based reuse detection; argon2id; no user enumeration.
- ADR-003: ACL = SQL predicate `readable_by(user)` applied before ranking; admins read all org docs;
  inactive users read nothing; cross-org returns 404.
- ADR-004: hybrid retrieval — full-text + vector concurrently (30 each), RRF (k=60), rerank top 20
  with MiniLM-L-12 cross-encoder; `hnsw.iterative_scan=relaxed_order` so filtered HNSW scans
  don't return short result sets.

## 12. Progress Log
- 2026-11-06 — Plan written. Renamed to **ragGround**. LLM default switched to **local Ollama**
  (`llama3.1:8b`) by user decision.
- 2026-11-06 — **M0 done.** FastAPI skeleton (`/health`, `/health/ready` checking Postgres, Redis,
  Ollama with per-check timeouts; 4 tests), ruff + mypy strict, Dockerfile; Next.js 15 shell
  showing live system status; docker-compose (pgvector pg18, redis, jaeger, api, web); GitHub
  Actions CI (api lint/type/test, web lint/type/build, image builds). Git repo initialised on
  `main`, **nothing committed yet**.
  Environment notes:
  - Host ports **5442** (Postgres) and **6389** (Redis) because 5432/6379 are used by other
    local projects' containers.
  - Local Node is **20.18**, too old for the Tailwind v4 build (needs ≥ 20.19; `.nvmrc` = 22).
    Web lint/typecheck work locally; **build runs in Docker (node:24-alpine)** until Node is upgraded.
  - Global `pnpm` is broken on Node 20.18 → web uses **npm**.
  - Ollama (systemd service) listens on **127.0.0.1 only**, so the `api` *container* can't reach
    it (readiness `llm` = ConnectError). The API run on the host works. Fix pending user choice.
  - Verified: `docker compose up` → web :3000 renders live status; API :8000 ready except `llm`.
  - npm registry connection was very slow; use `--maxsockets=3 --fetch-timeout=300000` if installs time out.
  Ollama decision: option (a) — run the API on the host during development (no system change).
- 2026-11-06 — **M1 done.** Models + first Alembic migration (orgs, users, groups, user_groups,
  documents, document_groups, refresh_tokens; up/down/up verified; `alembic check` clean, also in CI).
  Auth: `/auth/register` (creates org + admin), `/auth/login`, `/auth/refresh` (rotation + reuse
  detection), `/auth/logout`, `/auth/me`. Org admin: `/users` (list/create/deactivate),
  `/groups` (list/create), `PUT|DELETE /groups/{id}/members/{user_id}`. ACL predicate in `acl.py`.
  Seed: `uv run python -m ragground.seed` → Acme Demo: admin/alice(engineering)/bob(hr)/carol @acme.example.com,
  password `ragground-demo-password`. **22 tests pass** against real Postgres (`ragground_test` DB,
  recreated + migrated per session, truncated per test). ADRs 001–003 written.
  Gotcha found: `email-validator` rejects reserved TLDs like `.test` — use example.com.
  Still uncommitted (waiting on user). Next: M2 (ingestion pipeline).
- 2026-11-06 — **M2 done.** `chunks` table (pgvector `vector(384)` + HNSW cosine index, generated
  `tsvector` + GIN index, heading_path, word_count). Parsers: Markdown (heading paths via
  markdown-it), PDF (pymupdf, page sections), HTML (trafilatura → Markdown), DOCX (styles →
  Markdown), TXT. Chunkers: `fixed` (sliding window) and `heading` (section-bounded, paragraph
  packing, sentence split for giant paragraphs, tiny-chunk merge; embeds `headings + text`).
  Embeddings: local fastembed `BAAI/bge-small-en-v1.5` (cached at /tmp/fastembed_cache on host);
  `HashEmbedder` for tests. Worker: arq (`uv run arq ragground.worker.WorkerSettings`), retries with
  exponential backoff, parse errors permanent. Correctness: content-hash dedupe (same bytes →
  `unchanged`), `ingest_version` guard so stale jobs never overwrite newer chunks (tested), upload
  ACLs (members restrict only to own groups; replace/delete/reindex = uploader or admin;
  overwriting an unreadable doc → 409). API: `POST/GET /documents`, `GET/DELETE /documents/{id}`,
  `POST /documents/{id}/reindex`. Local blob storage behind a `BlobStorage` protocol (S3 later).
  **42 tests pass.** Verified live: upload → worker → `ready` with 3 chunks in 0.16 s; re-upload →
  `unchanged`. Compose now has a `worker` service + shared `appdata` volume.
  Gotchas: column named `text` shadows `sqlalchemy.text()` in the model body (import as `sql_text`);
  migrations need `CREATE EXTENSION vector` + `import pgvector.sqlalchemy` (template updated).
  Docker verified too: api + worker containers share the `appdata` volume; re-upload of a failed
  doc re-ingests (version 2 → ready, 3 chunks).
  ⚠️ Host and Docker workers share Redis on :6389 — never run both at once (a host worker will
  grab container jobs and fail with host file paths). Stop host workers by PID, not `pkill -f`.
  Next: M3 (hybrid retrieval: full-text + vector + RRF + reranker, ACL in SQL, `/search`).
- 2026-11-06 — **M3 done.** `retrieval/search.py`: `Retriever.search(user, query, mode, k)` with
  modes `fulltext | vector | hybrid | hybrid_rerank` (default), per-stage timings, both retrievers
  run concurrently with ACL in SQL; RRF fusion; reranker interface (`FastEmbedReranker`,
  `OverlapReranker` test double, or none). `POST /search` debug endpoint returns every hit's
  per-retriever ranks/scores. **57 tests pass**, incl. ACL-leak tests in every mode (forbidden doc
  is the best match but never returned), cross-org isolation, injection-ish query strings.
  Live check with real models (8 chunks): paraphrases found only by vector/hybrid; keyword by all;
  ACL holds; latency ≈20–90 ms without rerank, ≈170–220 ms with rerank (CPU).
  Reranker model cached at /tmp/fastembed_cache on host. Live-test script lives in the session
  scratchpad only (not in repo).
  Next: M4 (grounded answering with Ollama: `/chat` SSE streaming, citations, abstain, cost/latency, OTel).
- 2026-11-06 — **M4 done.** `providers/llm.py` (streaming `LLM` protocol, `OllamaLLM` with
  num_ctx=8192, `ScriptedLLM` test double, cost table), `generation/` (prompt with numbered,
  HTML-escaped `<source>` blocks + "sources are untrusted data" rule; exact abstain phrase;
  citation parser that strips out-of-range `[n]`; `answer_question()` event generator).
  Abstains *before* the LLM when no results or top rerank score < threshold (default -2.0).
  Drops irrelevant sources (rerank < threshold) after the first. `POST /chat` (SSE: sources /
  delta / done, or `stream:false` JSON), `GET /queries` (own; admin = org). `queries` table logs
  answer, sources, citations, tokens, cost, per-stage timings. OTel tracing (opt-in) → Jaeger:
  request → rag.retrieve → retrieval.{fulltext,vector,embed_query,rerank} → rag.generate.
  **72 tests pass.**
  Live run (llama3.1:8b, CPU, k=3): retrieval 367 ms, **first token 98 s**, total 133 s for
  288 prompt / 72 output tokens. Answer correctly cited [1] but padded with unsupported claims →
  motivating example for the faithfulness metric.
  Measured Ollama speed on this machine: ~7 tok/s prompt, ~2 tok/s generation (~3 GB RAM free).
  Next: M5 (eval harness).
- 2026-11-06 — **M5 in progress.** Corpus: FastAPI tutorial docs @0.115.0 (48 files, MIT;
  `evals/corpus/fetch_fastapi_docs.py`, gitignored, SHA manifest) + synthetic injection doc
  `evals/corpus/injected/internal-gateway.md`. `tutorial/security/*` restricted to
  `security-team`. Golden set `evals/golden/golden.jsonl`: **120 cases** (75 factual,
  8 factual_restricted, 10 multi_hop, 15 unanswerable, 8 acl, 4 injection), labelled with
  evidence phrases (file + text) so labels survive re-chunking; `ragground-eval validate` → 0
  problems. Drafted by Claude from the corpus — **still needs human review by the user**.
  Harness `ragground.evals` (corpus/golden/metrics/judge/runner/report/cli) + 12 unit tests.
  Eval found a real ingestion bug: inline HTML (`<a href…>`, `<abbr>`) was kept in chunk text →
  parser now strips inline HTML outside code fences.
  **First retrieval baseline** (heading-300, 489 chunks, 97 answerable cases, ACL leak 0 in all
  modes): fulltext R@5 0.361 / MRR 0.334 · vector 0.835 / 0.748 · hybrid 0.871 / 0.766 ·
  hybrid_rerank 0.912 / 0.844 but **p50 7.7 s** (20 × 300-word passages through MiniLM-L12 on
  CPU; latency is linear in input tokens). Made rerank depth/truncation/model and full-text
  AND/OR configurable; experiment matrix running.
- 2026-11-07 — Session was interrupted (machine/Docker restart: start with
  `docker compose up -d postgres redis`; eval data persists in volumes). Round-1 results
  (heading-300 words): full-text OR vs AND: fulltext R@5 0.361→0.696 but **hybrid MRR
  0.766→0.694** (noisier candidates crowd fusion); rerank top-10 = same quality as top-20
  (R@5 0.912, MRR 0.847) at 4.1 s vs 7.7 s; truncating rerank input to 120 *words* didn't help.
  Root cause found by measuring with the real tokenizer: **2.44 tokens/word** in these docs
  (code-heavy), median chunk 188 tokens, **11% of chunks > 512 tokens → silently truncated by
  the embedding model and reranker.** Chunkers now take a pluggable length function;
  `chunk_size_unit = words | tokens` (tokens = bge tokenizer). Default still words until the
  eval decides. 86 tests pass. Latency numbers are noisy: the desktop shares CPU with Chrome/VS
  Code (load ~6 on 8 cores) — compare quality, not small latency deltas. Round 2 running:
  heading-350tok, L6 reranker, fixed chunker.
