import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


class BM25:
    """Okapi BM25 over a sparse term matrix. Scores all products for a query."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.vec = CountVectorizer(stop_words="english", lowercase=True)
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n_docs else 1.0
        # Precompute the document-side BM25 weight for every nonzero (doc, term).
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * dl[rows] / avgdl)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.indices]
        self.W = sparse.csr_matrix((w, tf.indices, tf.indptr), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        if q.nnz == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        q.data[:] = 1.0  # binary query term weights
        return (self.W @ q.T).toarray().ravel()


class HashingEmbedder:
    """Tiny offline stand-in for smoke tests. Not semantic; do NOT report its numbers."""

    def __init__(self, dim: int = 384):
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def encode(self, texts, **_):
        return self.vec.transform(texts).toarray().astype(np.float32)


def load_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


class Dense:
    """Cosine-similarity search over normalized embeddings (cached on disk)."""

    def __init__(self, docs: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "cache"):
        self.model = load_embedder(model_name)
        key = hashlib.md5((model_name + str(len(docs))).encode()).hexdigest()[:10]
        cache = Path(cache_dir) / f"emb_{key}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = np.asarray(self.model.encode(docs, batch_size=128, show_progress_bar=True,
                                                    normalize_embeddings=True), dtype=np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def scores(self, query: str) -> np.ndarray:
        q = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        return self.emb @ q


def rrf(score_lists: list[np.ndarray], k: int = 60, depth: int = 200) -> np.ndarray:
    """Reciprocal Rank Fusion: combine rankings without tuning score scales."""
    fused = np.zeros_like(score_lists[0], dtype=np.float64)
    for s in score_lists:
        d = min(depth, len(s))
        top = np.argpartition(-s, d - 1)[:d]
        top = top[np.argsort(-s[top])]
        fused[top] += 1.0 / (k + np.arange(1, len(top) + 1))
    return fused


def top_k(scores: np.ndarray, k: int = 10) -> np.ndarray:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]
