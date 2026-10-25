"""Minimal search API.  Run: uvicorn api:app --reload"""
import os

from fastapi import FastAPI, HTTPException, Query

from src.data import build_documents, load_wands
from src.retrievers import BM25, Dense, rrf, top_k

DATA = os.getenv("WANDS_PATH", "data/WANDS/dataset")
MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="WANDS product search")
products, _, _ = load_wands(DATA)
docs = build_documents(products)
bm25, dense = BM25(docs), Dense(docs, MODEL)


@app.get("/search")
def search(q: str = Query(..., min_length=1), mode: str = "hybrid", k: int = Query(10, ge=1, le=50)):
    if mode == "bm25":
        s = bm25.scores(q)
    elif mode == "dense":
        s = dense.scores(q)
    elif mode == "hybrid":
        s = rrf([bm25.scores(q), dense.scores(q)])
    else:
        raise HTTPException(400, "mode must be bm25, dense or hybrid")
    idx = top_k(s, k)
    cols = [c for c in ("product_id", "product_name", "product_class") if c in products]
    return {"query": q, "mode": mode, "results": products.iloc[idx][cols].to_dict("records")}
