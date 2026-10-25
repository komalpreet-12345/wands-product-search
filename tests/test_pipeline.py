"""End-to-end smoke test on a tiny synthetic dataset (no downloads, no model)."""
import pandas as pd

from src.data import build_documents, load_wands, relevance_by_query
from src.metrics import ndcg_at_k
from src.retrievers import BM25, Dense, rrf, top_k


def make_dataset(tmp_path):
    pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "product_name": ["oak dining table", "velvet sofa", "glass coffee table", "floor lamp"],
        "product_class": ["Tables", "Sofas", "Tables", "Lamps"],
        "product_description": ["solid oak wood table", "soft velvet couch", "tempered glass top", "tall standing light"],
        "product_features": ["", "", "", ""],
    }).to_csv(tmp_path / "product.csv", sep="\t", index=False)
    pd.DataFrame({"query_id": [10], "query": ["wood dining table"], "query_class": ["Tables"]}
                 ).to_csv(tmp_path / "query.csv", sep="\t", index=False)
    pd.DataFrame({"id": [0, 1, 2], "query_id": [10, 10, 10], "product_id": [1, 3, 2],
                  "label": ["Exact", "Partial", "Irrelevant"]}
                 ).to_csv(tmp_path / "label.csv", sep="\t", index=False)


def test_bm25_ranks_best_match_first(tmp_path):
    make_dataset(tmp_path)
    products, queries, labels = load_wands(str(tmp_path))
    docs = build_documents(products)
    bm25 = BM25(docs)
    ranked = [int(products["product_id"].iloc[i]) for i in top_k(bm25.scores("wood dining table"), 3)]
    assert ranked[0] == 1
    rel = relevance_by_query(labels)[10]
    assert ndcg_at_k(ranked, rel) > 0.9


def test_hybrid_runs(tmp_path):
    make_dataset(tmp_path)
    products, _, _ = load_wands(str(tmp_path))
    docs = build_documents(products)
    bm25, dense = BM25(docs), Dense(docs, "hashing", cache_dir=str(tmp_path / "cache"))
    fused = rrf([bm25.scores("glass table"), dense.scores("glass table")], depth=4)
    assert int(products["product_id"].iloc[top_k(fused, 1)[0]]) == 3
