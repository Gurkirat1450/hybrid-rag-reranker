"""Tests for app/metrics.py. Expected to FAIL until both functions are implemented."""

import pytest

from app.metrics import hit_at_k, reciprocal_rank

# ---------------------------------------------------------------- hit_at_k


def test_hit_dedupes_before_cutoff():
    # ["a","a","a","b"] -> unique ["a","b"]; top-2 = ["a","b"] contains b -> 1.0
    # (cutting first would give ["a","a"] -> 0.0, which is wrong)
    assert hit_at_k(["a", "a", "a", "b"], {"b"}, 2) == 1.0


def test_hit_k1_misses_second_unique_doc():
    # unique ["a","b"]; top-1 = ["a"]; gold {"b"} -> 0.0
    assert hit_at_k(["a", "a", "a", "b"], {"b"}, 1) == 0.0


def test_hit_at_1():
    assert hit_at_k(["b", "a"], {"b"}, 1) == 1.0


def test_hit_no_gold_in_ranking():
    assert hit_at_k(["a", "b", "c"], {"z"}, 5) == 0.0


def test_hit_empty_gold_or_ranking():
    assert hit_at_k(["a"], set(), 3) == 0.0
    assert hit_at_k([], {"a"}, 3) == 0.0


def test_hit_k_larger_than_unique_docs():
    # unique ["a","c"], k=10 uses both -> c found -> 1.0
    assert hit_at_k(["a", "c", "a"], {"c"}, 10) == 1.0


def test_hit_k_zero():
    assert hit_at_k(["a"], {"a"}, 0) == 0.0


def test_hit_multiple_gold_docs():
    # unique ["x","y","b"]; top-3 contains b (one of the gold docs) -> 1.0
    assert hit_at_k(["x", "y", "b", "a"], {"a", "b"}, 3) == 1.0


# ---------------------------------------------------------------- reciprocal_rank


def test_rr_dedupes_before_cutoff():
    # ["a","a","a","b"] -> unique ["a","b"]; b at rank 2 -> 1/2 = 0.5
    assert reciprocal_rank(["a", "a", "a", "b"], {"b"}, k=2) == pytest.approx(0.5)


def test_rr_cutoff_excludes_gold():
    # unique ["a","b"], k=1 -> only ["a"] -> 0.0
    assert reciprocal_rank(["a", "a", "a", "b"], {"b"}, k=1) == 0.0


def test_rr_rank_three_after_dedupe():
    # ["a","a","b","c"] -> unique ["a","b","c"]; c at rank 3 -> 1/3
    assert reciprocal_rank(["a", "a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_rr_first_gold_wins():
    # unique ["a","b"]; both gold; first gold at rank 1 -> 1.0
    assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0


def test_rr_no_gold():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_rr_default_k_is_10():
    docs = [f"d{i}" for i in range(11)]  # d0..d10, all unique; d9 is rank 10, d10 is rank 11
    assert reciprocal_rank(docs, {"d9"}) == pytest.approx(1 / 10)  # rank 10 -> 0.1
    assert reciprocal_rank(docs, {"d10"}) == 0.0  # rank 11 > 10


def test_rr_k_zero():
    assert reciprocal_rank(["a"], {"a"}, k=0) == 0.0
