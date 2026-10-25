"""Compare BM25, dense and hybrid (RRF) search on WANDS human relevance labels.

Usage:
  python evaluate.py --data data/WANDS/dataset
  python evaluate.py --data data/WANDS/dataset --max-queries 100   # quick run
"""
import argparse
import time
from pathlib import Path

import numpy as np

from src.data import build_documents, load_wands, relevance_by_query
from src.metrics import ndcg_at_k, precision_at_k
from src.retrievers import BM25, Dense, rrf, top_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/WANDS/dataset")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="embedding model name, or 'hashing' for an offline smoke test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all queries")
    ap.add_argument("--out", default="results/results.md")
    args = ap.parse_args()

    products, queries, labels = load_wands(args.data)
    rel_by_q = relevance_by_query(labels)
    docs = build_documents(products)
    pids = products["product_id"].to_numpy()

    if args.max_queries:
        queries = queries.head(args.max_queries)
    queries = queries[queries["query_id"].isin(rel_by_q)]
    print(f"{len(products)} products | {len(queries)} labeled queries | {len(labels)} judgments")

    bm25 = BM25(docs)
    dense = Dense(docs, args.model)

    methods = {
        "BM25 (keyword)": lambda q: bm25.scores(q),
        "Dense (embeddings)": lambda q: dense.scores(q),
        "Hybrid (RRF)": lambda q: rrf([bm25.scores(q), dense.scores(q)]),
    }

    rows = []
    for name, fn in methods.items():
        ndcgs, precs, t0 = [], [], time.perf_counter()
        for qid, qtext in zip(queries["query_id"], queries["query"]):
            ranked = [int(pids[i]) for i in top_k(fn(str(qtext)), args.k)]
            ndcgs.append(ndcg_at_k(ranked, rel_by_q[int(qid)], args.k))
            precs.append(precision_at_k(ranked, rel_by_q[int(qid)], args.k))
        ms = (time.perf_counter() - t0) / len(queries) * 1000
        rows.append((name, np.mean(ndcgs), np.mean(precs), ms))
        print(f"{name:20s} NDCG@{args.k}={rows[-1][1]:.4f}  P@{args.k}={rows[-1][2]:.4f}  {ms:.1f} ms/query")

    md = [f"| Method | NDCG@{args.k} | Precision@{args.k} | Avg latency (ms/query) |", "|---|---|---|---|"]
    md += [f"| {n} | {a:.3f} | {p:.3f} | {m:.1f} |" for n, a, p, m in rows]
    md.append(f"\n{len(queries)} queries, {len(products)} products. Unjudged products count as irrelevant.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
