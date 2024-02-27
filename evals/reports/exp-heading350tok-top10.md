## Retrieval eval

| `chunker`=heading | `max_size`=350 | `size_unit`=tokens | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=10 | `rerank_max_words`=None | `fulltext_operator`=or | `chunks`=636 |
|---|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| vector | 0.840 | 0.907 | 0.753 | 0.785 | 0.670 | 0.000 | 50.9 | 71.0 |
| hybrid | 0.887 | 0.949 | 0.739 | 0.789 | 0.619 | 0.000 | 70.5 | 106.2 |
| hybrid_rerank | 0.938 | 0.949 | 0.871 | 0.881 | 0.804 | 0.000 | 3630.8 | 4058.6 |

Answerable cases scored: 97.
