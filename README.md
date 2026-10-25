# WANDS Product Search

Product search engine for home goods, built and evaluated on **WANDS** (Wayfair ANnotation Dataset),
Wayfair's public search-relevance dataset with human relevance labels (Exact / Partial / Irrelevant).

It compares three ranking approaches and measures them with NDCG@10:

1. **BM25**: keyword baseline (own implementation, sparse matrices)
2. **Dense**: sentence embeddings (all-MiniLM-L6-v2, 384-dim) with cosine similarity
3. **Hybrid**: Reciprocal Rank Fusion of BM25 and dense rankings

## Results

<!-- Paste the table from results/results.md here after you run evaluate.py. Do not edit the numbers. -->

| Method | NDCG@10 | Precision@10 | Avg latency (ms/query) |
|---|---|---|---|
| BM25 (keyword) | TBD | TBD | TBD |
| Dense (embeddings) | TBD | TBD | TBD |
| Hybrid (RRF) | TBD | TBD | TBD |

**What I found:** TBD (one or two sentences: which method won, on which kinds of queries, and why you think so).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/wayfair/WANDS data/WANDS
```

## Run

```bash
python evaluate.py --data data/WANDS/dataset                   # full evaluation (first run embeds ~43k products)
python evaluate.py --data data/WANDS/dataset --max-queries 50  # quick check
uvicorn api:app --reload                                       # then open /docs
# example: http://localhost:8000/search?q=mid%20century%20sofa&mode=hybrid
python -m pytest                                               # unit tests
```

## Method notes

- Each product is one document: name + class + description (first 600 chars) + features.
- Gains: Exact = 2, Partial = 1, Irrelevant = 0. NDCG uses gain 2^g - 1.
- Products with no label for a query count as irrelevant, which can under-score systems that
  retrieve relevant but unjudged items. All three methods share this limitation.
- Precision@10 counts Exact and Partial as relevant.

## Limitations

- English only; single embedding model; no query understanding (spelling, synonyms, filters).
- RRF uses fixed parameters (k=60, depth=200); not tuned.
- No click or purchase signals, which production search relies on heavily.
