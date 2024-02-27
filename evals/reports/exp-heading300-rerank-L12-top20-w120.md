## Retrieval eval

| `chunker`=heading | `max_words`=300 | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=20 | `rerank_max_words`=120 | `fulltext_operator`=or | `chunks`=489 |
|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| hybrid_rerank | 0.892 | 0.933 | 0.839 | 0.853 | 0.773 | 0.000 | 7442.3 | 8186.2 |

Answerable cases scored: 97.
