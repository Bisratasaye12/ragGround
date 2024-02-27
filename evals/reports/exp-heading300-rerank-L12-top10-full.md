## Retrieval eval

| `chunker`=heading | `max_words`=300 | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=10 | `rerank_max_words`=None | `fulltext_operator`=or | `chunks`=489 |
|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| hybrid_rerank | 0.912 | 0.923 | 0.847 | 0.856 | 0.783 | 0.000 | 4102.3 | 4340.6 |

Answerable cases scored: 97.
