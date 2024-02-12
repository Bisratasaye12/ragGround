# ADR-004: Hybrid retrieval with RRF fusion and a small local cross-encoder

**Status:** accepted (2026-11-06)

## Context
Keyword search misses paraphrases ("we got hacked" vs "breach"); vector search misses exact
tokens (error codes, product names, numbers). Their scores (ts_rank vs cosine similarity) are on
incomparable scales.

## Decision
1. Run **full-text** (`websearch_to_tsquery` + `ts_rank_cd`) and **vector** (pgvector HNSW,
   cosine) retrieval concurrently, each returning up to 30 candidates, both with the ACL
   predicate in SQL (ADR-003).
2. Fuse with **Reciprocal Rank Fusion**, `score = Σ 1/(60 + rank)`. Rank-based, so no score
   calibration between retrievers is needed.
3. **Rerank** the top 20 fused candidates with a cross-encoder, default
   `Xenova/ms-marco-MiniLM-L-12-v2` (≈120 MB ONNX, CPU). `BAAI/bge-reranker-base` was the original
   plan but is 1.04 GB and much slower on CPU; it stays an option behind the `Reranker` interface
   for the M6 experiments.
4. Set `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8): with a WHERE filter, a plain HNSW
   scan filters *after* the index walk and can silently return fewer than k rows for users who can
   read only a small fraction of the corpus.

## Consequences
- First live check (8 chunks, real models): paraphrased queries return nothing from full-text
  alone but are found by vector/hybrid; keyword queries are found by every mode.
- Rerank adds ≈150–200 ms on CPU. Modes are selectable per request so the M6 matrix can measure
  quality vs latency (`fulltext`, `vector`, `hybrid`, `hybrid_rerank`).
- Open question for M6: whether the reranker helps enough on this corpus to justify its latency.
