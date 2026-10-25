import math


def dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[int], rel: dict[int, int], k: int = 10) -> float:
    """NDCG@k with graded gains. Ideal ranking uses all judged products for the query."""
    gains = [rel.get(pid, 0) for pid in ranked_ids[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def precision_at_k(ranked_ids: list[int], rel: dict[int, int], k: int = 10) -> float:
    """Fraction of the top-k that is Exact or Partial (gain >= 1)."""
    top = ranked_ids[:k]
    return sum(1 for pid in top if rel.get(pid, 0) >= 1) / k if k else 0.0
