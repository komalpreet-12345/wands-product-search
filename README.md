# WANDS Product Search

Product search engine for home goods, built and evaluated on **WANDS** (Wayfair ANnotation Dataset),
Wayfair's public search-relevance dataset with human relevance labels (Exact / Partial / Irrelevant).

It compares three ranking approaches and measures them with NDCG@10:

1. **BM25**: keyword baseline (own implementation, sparse matrices)
2. **Dense**: sentence embeddings (all-MiniLM-L6-v2, 384-dim) with cosine similarity
3. **Hybrid**: Reciprocal Rank Fusion of BM25 and dense rankings

## Results

Evaluated on all 480 labeled queries (42,994 products, 233,448 human judgments), `all-MiniLM-L6-v2` embeddings, CPU only.

| Method | NDCG@10 | Precision@10 | Avg latency (ms/query) |
|---|---|---|---|
| BM25 (keyword) | 0.654 | 0.752 | 11.9 |
| Dense (embeddings) | 0.658 | 0.781 | 13.5 |
| Hybrid (RRF) | 0.707 | 0.815 | 27.3 |

**What I found:** Hybrid Reciprocal Rank Fusion was the best method, improving NDCG@10 by 0.053 over BM25 (about 8% relative) and by 0.049 over dense retrieval alone, and Precision@10 from 0.752 to 0.815. Dense retrieval on its own was close to BM25 on NDCG (0.658 vs 0.654), with a higher Precision@10, so neither single method dominates. Fusing them helps, which suggests they retrieve different relevant products. The cost is latency: hybrid runs both retrievers, so it takes about twice as long per query (27 ms vs 12 to 14 ms). I have not yet broken results down by query type or tested significance, so I do not claim why each method wins on specific queries.

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
