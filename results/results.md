| Method | NDCG@10 | Precision@10 | Avg latency (ms/query) |
|---|---|---|---|
| BM25 (keyword) | 0.654 | 0.752 | 11.9 |
| Dense (embeddings) | 0.658 | 0.781 | 13.5 |
| Hybrid (RRF) | 0.707 | 0.815 | 27.3 |

480 queries, 42994 products. Unjudged products count as irrelevant.
