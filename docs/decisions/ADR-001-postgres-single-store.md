# ADR-001: Postgres + pgvector as the single data store

**Status:** accepted (2026-11-06)

## Context
ragGround needs vector search, keyword (full-text) search, relational data (orgs, users, groups)
and per-document access control. A dedicated vector database (Qdrant, Weaviate, Pinecone) is the
common choice for the vector part.

## Decision
Use one Postgres instance with the pgvector extension (HNSW index) and built-in full-text search.

## Consequences
- Hybrid retrieval and the ACL filter run in **one SQL query**: the ACL predicate (ADR-003) joins
  chunks → documents → groups inside the same WHERE clause as the similarity search. With a
  separate vector DB, ACLs must be copied into the vector store's metadata and kept in sync, which
  is a common source of leaks.
- One system to back up, migrate and operate; transactions cover documents and their chunks.
- Trade-off: a dedicated vector DB scales further and has richer ANN tuning. Revisit beyond
  roughly 10M chunks or if p95 retrieval latency exceeds budget; the retrieval layer sits behind an
  interface so the store can be swapped.
