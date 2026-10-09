"""Tests for app/fusion.py. Expected to FAIL until reciprocal_rank_fusion is implemented."""

import pytest

from app.fusion import reciprocal_rank_fusion


def ids(result):
    return [cid for cid, _ in result]


def test_single_list_k1():
    # k=1, ranking [5, 3, 9]
    # 5: rank 1 -> 1/(1+1) = 0.5
    # 3: rank 2 -> 1/(1+2) = 0.3333...
    # 9: rank 3 -> 1/(1+3) = 0.25
    result = reciprocal_rank_fusion([[5, 3, 9]], k=1)
    assert ids(result) == [5, 3, 9]
    assert [s for _, s in result] == pytest.approx([0.5, 1 / 3, 0.25])


def test_two_lists_with_missing_ids_k1():
    # k=1, A = [1, 2, 3], B = [3, 4]
    # 1: A rank 1                  -> 1/2           = 0.5
    # 2: A rank 2                  -> 1/3           = 0.3333...
    # 3: A rank 3 + B rank 1       -> 1/4 + 1/2     = 0.75
    # 4: B rank 2 (absent from A)  -> 1/3           = 0.3333...
    # order: 3 (0.75), 1 (0.5), then 2 and 4 tie at 1/3 -> smaller id first: 2, 4
    result = reciprocal_rank_fusion([[1, 2, 3], [3, 4]], k=1)
    assert ids(result) == [3, 1, 2, 4]
    assert [s for _, s in result] == pytest.approx([0.75, 0.5, 1 / 3, 1 / 3])


def test_tie_broken_by_smaller_chunk_id():
    # k=60, A = [7, 2], B = [2, 7]
    # 7: 1/61 + 1/62 ; 2: 1/62 + 1/61 -> identical scores -> 2 before 7
    result = reciprocal_rank_fusion([[7, 2], [2, 7]], k=60)
    assert ids(result) == [2, 7]
    expected = 1 / 61 + 1 / 62  # = 0.016393... + 0.016129... = 0.032522...
    assert [s for _, s in result] == pytest.approx([expected, expected])


def test_tie_between_different_lists_top_items():
    # k=60, A = [9], B = [4]: both get 1/(60+1) = 1/61 -> 4 before 9
    result = reciprocal_rank_fusion([[9], [4]], k=60)
    assert ids(result) == [4, 9]
    assert [s for _, s in result] == pytest.approx([1 / 61, 1 / 61])


def test_default_k_is_60():
    # [[10]] with default k=60 -> 1/(60+1) = 1/61 = 0.016393...
    result = reciprocal_rank_fusion([[10]])
    assert ids(result) == [10]
    assert result[0][1] == pytest.approx(1 / 61)


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_list_equals_single_list():
    # k=1, A = [], B = [8, 6]: 8 -> 1/2 = 0.5, 6 -> 1/3
    result = reciprocal_rank_fusion([[], [8, 6]], k=1)
    assert ids(result) == [8, 6]
    assert [s for _, s in result] == pytest.approx([0.5, 1 / 3])


def test_agreement_beats_single_top_rank():
    # k=60, A = [1, 2], B = [3, 2]
    # 1: 1/61                  = 0.016393
    # 3: 1/61                  = 0.016393
    # 2: 1/62 + 1/62 = 2/62    = 0.032258  (in both lists -> wins)
    # 1 and 3 tie -> 1 before 3
    result = reciprocal_rank_fusion([[1, 2], [3, 2]], k=60)
    assert ids(result) == [2, 1, 3]
    assert [s for _, s in result] == pytest.approx([2 / 62, 1 / 61, 1 / 61])
