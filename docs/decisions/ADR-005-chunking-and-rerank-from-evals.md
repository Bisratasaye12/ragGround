# ADR-005: Token-measured heading chunks and rerank top-10, chosen by the eval

**Status:** accepted (2026-11-07). Supersedes the size/rerank defaults in ADR-004.

## Context
The first retrieval baseline (heading chunks of ≤300 *words*, rerank top-20) reached
Recall@5 0.912 with a p50 of 7.7 s, almost all of it in the reranker. Measuring the chunks with
the model tokenizer showed the word limit was the wrong unit: the FastAPI docs average
**2.44 tokens per word** (code is token-dense), median chunk 188 tokens, and **11% of chunks
exceeded the 512-token input limit** of `bge-small` and the MiniLM reranker. Both truncate
silently, so the end of those chunks was invisible to vector search and reranking.

## Decision
- Measure chunk size in **model tokens** (bge tokenizer), default **≤350 tokens** (headroom for
  the heading prefix and special tokens).
- Rerank the **top 10** fused candidates (not 20) with MiniLM-L-12.
- Keep the heading-aware chunker; keep MiniLM-L-12 over L-6.

## Evidence (120-case golden set, 97 answerable cases scored, `evals/reports/exp-*.json`)

| Config | Recall@5 | MRR | nDCG@10 | p50 |
|---|---|---|---|---|
| heading ≤300 words, rerank top-20 (old default) | 0.912 | 0.844 | 0.854 | 7.7 s |
| heading ≤300 words, rerank top-10 | 0.912 | 0.847 | 0.856 | 4.1 s |
| heading ≤300 words, rerank top-20, input cut to 120 words | 0.892 | 0.839 | 0.853 | 7.4 s |
| heading ≤300 words, **L-6** reranker top-10 | 0.907 | 0.839 | 0.854 | 2.8 s |
| fixed 300-word windows, rerank top-10 | 0.876 | 0.755 | 0.798 | 5.4 s |
| **heading ≤350 tokens, rerank top-10 (new default)** | **0.938** | **0.871** | **0.881** | **3.6 s** |

ACL leak rate was 0 in every configuration. Latencies come from a shared desktop CPU
(browser and IDE running), so read them as relative, not absolute.

## Consequences
- Best quality on every metric, at half the old latency.
- More, smaller chunks (636 vs 489), so storage is slightly larger.
- Re-ingest is needed for existing documents (`POST /documents/{id}/reindex`).
- Open: full-text OR vs AND. OR raises full-text-only recall (0.36 → 0.70) but lowered hybrid
  MRR without reranking (0.766 → 0.694). With reranking on, OR is the default; revisit with
  weighted RRF.
