## Retrieval eval

| `chunker`=heading | `max_size`=350 | `size_unit`=tokens | `embedding`=BAAI/bge-small-en-v1.5 | `reranker`=Xenova/ms-marco-MiniLM-L-12-v2 | `rerank_top_n`=10 | `rerank_max_words`=None | `fulltext_operator`=or | `chunks`=636 |
|---|---|---|---|---|---|---|---|---|

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fulltext | 0.717 | 0.856 | 0.551 | 0.620 | 0.402 | 0.000 | 52.3 | 109.5 |

Answerable cases scored: 97.
