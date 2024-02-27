## Retrieval eval

| `chunker`=heading | `max_words`=300 | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=20 | `rerank_max_words`=None | `fulltext_operator`=or | `chunks`=489 |
|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fulltext | 0.696 | 0.794 | 0.519 | 0.582 | 0.381 | 0.000 | 33.2 | 80.3 |
| hybrid | 0.866 | 0.923 | 0.694 | 0.748 | 0.577 | 0.000 | 48.5 | 78.4 |

Answerable cases scored: 97.
