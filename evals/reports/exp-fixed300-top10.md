## Retrieval eval

| `chunker`=fixed | `max_size`=300 | `size_unit`=words | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=10 | `rerank_max_words`=None | `fulltext_operator`=or | `chunks`=207 |
|---|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| vector | 0.814 | 0.912 | 0.690 | 0.741 | 0.567 | 0.000 | 51.6 | 78.7 |
| hybrid | 0.871 | 0.928 | 0.716 | 0.766 | 0.608 | 0.000 | 72.6 | 104.2 |
| hybrid_rerank | 0.876 | 0.928 | 0.755 | 0.798 | 0.660 | 0.000 | 5387.5 | 5773.2 |

Answerable cases scored: 97.
