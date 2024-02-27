## Retrieval eval

| `chunker`=heading | `max_size`=300 | `size_unit`=words | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-6-v2 | `rerank_top_n`=10 | `rerank_max_words`=None | `fulltext_operator`=or | `chunks`=489 |
|---|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| hybrid_rerank | 0.907 | 0.923 | 0.839 | 0.854 | 0.763 | 0.000 | 2785.3 | 3056.6 |

Answerable cases scored: 97.
