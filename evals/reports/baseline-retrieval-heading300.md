## Retrieval eval

| `chunker`=heading | `max_words`=300 | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `chunks`=489 |
|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fulltext | 0.361 | 0.371 | 0.334 | 0.339 | 0.309 | 0.000 | 6.8 | 10.3 |
| vector | 0.835 | 0.902 | 0.748 | 0.779 | 0.670 | 0.000 | 38.7 | 50.4 |
| hybrid | 0.871 | 0.907 | 0.766 | 0.795 | 0.691 | 0.000 | 41.7 | 56.8 |
| hybrid_rerank | 0.912 | 0.923 | 0.844 | 0.854 | 0.773 | 0.000 | 7671.9 | 8326.7 |

Answerable cases scored: 97.
