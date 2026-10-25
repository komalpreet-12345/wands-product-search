import math

from src.metrics import ndcg_at_k, precision_at_k


def test_perfect_ranking_is_one():
    rel = {1: 2, 2: 1, 3: 0}
    assert math.isclose(ndcg_at_k([1, 2, 3], rel, 10), 1.0)


def test_worse_ranking_scores_lower():
    rel = {1: 2, 2: 1}
    assert ndcg_at_k([2, 1], rel) < ndcg_at_k([1, 2], rel)


def test_no_relevant_results_is_zero():
    assert ndcg_at_k([9, 8], {1: 2}) == 0.0


def test_precision_counts_exact_and_partial():
    assert precision_at_k([1, 2, 3, 4], {1: 2, 2: 1, 3: 0}, 4) == 0.5
