"""Tests for app/rerank.py. Expected to FAIL until rerank is implemented."""

import pytest

from app.rerank import rerank
from app.types import Candidate


def make(texts):
    # chunk_id = position; initial score is a fake RRF score; ranks are arbitrary markers
    return [
        Candidate(chunk_id=i, doc_id=f"{t}.md", heading=f"H {t}", text=t, score=0.0, vector_rank=i + 1, bm25_rank=None)
        for i, t in enumerate(texts)
    ]


class FakePredict:
    """Scores each pair by looking up the passage text; records every call."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, pairs):
        self.calls.append(list(pairs))
        return [self.table[text] for _, text in pairs]


def test_builds_query_text_pairs_in_input_order():
    cands = make(["a", "b", "c"])
    fake = FakePredict({"a": 0.1, "b": 0.2, "c": 0.3})
    rerank("my query", cands, fake, top_n=3)
    all_pairs = [p for call in fake.calls for p in call]
    assert all_pairs == [("my query", "a"), ("my query", "b"), ("my query", "c")]


def test_sorts_by_score_and_truncates_to_top_n():
    # scores: a=0.1, b=0.9, c=0.5, d=0.7 -> desc: b, d, c, a -> top 2: b, d
    cands = make(["a", "b", "c", "d"])
    out = rerank("q", cands, FakePredict({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}), top_n=2)
    assert [c.text for c in out] == ["b", "d"]
    assert [c.score for c in out] == pytest.approx([0.9, 0.7])


def test_top_n_larger_than_candidates_returns_all_sorted():
    # same scores, top_n=10 > 4 -> b(0.9), d(0.7), c(0.5), a(0.1)
    cands = make(["a", "b", "c", "d"])
    out = rerank("q", cands, FakePredict({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}), top_n=10)
    assert [c.text for c in out] == ["b", "d", "c", "a"]


def test_ties_keep_input_order():
    # a=0.5, b=0.9, c=0.5, d=0.5 -> b first, then the 0.5 tie in input order: a, c, d
    cands = make(["a", "b", "c", "d"])
    out = rerank("q", cands, FakePredict({"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.5}), top_n=4)
    assert [c.text for c in out] == ["b", "a", "c", "d"]
    # top_n=3 cuts the last tied item: b, a, c
    cands = make(["a", "b", "c", "d"])
    out = rerank("q", cands, FakePredict({"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.5}), top_n=3)
    assert [c.text for c in out] == ["b", "a", "c"]


def test_sets_score_and_keeps_other_fields():
    cands = make(["a", "b"])
    out = rerank("q", cands, FakePredict({"a": 0.25, "b": 0.75}), top_n=2)
    b = out[0]
    assert b.text == "b"
    assert b.score == pytest.approx(0.75)
    assert isinstance(b.score, float)
    # unchanged fields from make(): chunk_id=1, doc_id="b.md", heading="H b", vector_rank=2, bm25_rank=None
    assert (b.chunk_id, b.doc_id, b.heading, b.vector_rank, b.bm25_rank) == (1, "b.md", "H b", 2, None)
    # the score is set on the candidate objects themselves
    assert cands[0].score == pytest.approx(0.25)
    assert cands[1].score == pytest.approx(0.75)


def test_accepts_tuple_scores():
    # predict_fn may return any Sequence[float]
    cands = make(["x", "y"])
    out = rerank("q", cands, lambda pairs: (0.0, 1.0), top_n=1)
    assert [c.text for c in out] == ["y"]


def test_empty_candidates():
    assert rerank("q", [], lambda pairs: [], top_n=5) == []
