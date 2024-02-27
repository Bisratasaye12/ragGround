## Retrieval eval

| `chunker`=heading | `max_words`=300 | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=20 | `rerank_max_words`=None | `fulltext_operator`=and | `chunks`=489 |
|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fulltext | 0.361 | 0.371 | 0.334 | 0.339 | 0.309 | 0.000 | 6.0 | 10.2 |
| hybrid | 0.871 | 0.907 | 0.766 | 0.795 | 0.691 | 0.000 | 36.8 | 53.6 |

Answerable cases scored: 97.
