"""Cross-encoder re-ranking (HAND-WRITTEN CORE: implement this yourself)."""

from __future__ import annotations

from typing import Callable, Sequence

from app.types import Candidate


def rerank(
    query: str,
    candidates: list[Candidate],
    predict_fn: Callable[[list[tuple[str, str]]], Sequence[float]],
    top_n: int,
) -> list[Candidate]:
    """Re-score candidates with a cross-encoder and return the best ``top_n``.

    Contract:
        * Build one (query, candidate.text) pair per candidate, in input order.
        * Call ``predict_fn(pairs)``; it returns one relevance score per pair, in the
          same order (a list, tuple or numpy array; higher = more relevant).
        * Set ``candidate.score`` to its new score (as a float) on the candidate
          objects themselves. All other fields (chunk_id, doc_id, heading, text,
          vector_rank, bm25_rank) are left unchanged.
        * Return at most ``top_n`` candidates sorted by the new score descending.
          The sort is stable: candidates with equal scores keep their input order
          (so the earlier fused rank wins a tie).
        * If ``top_n`` exceeds ``len(candidates)``, return all of them, sorted.
        * ``candidates == []`` returns ``[]``.

    Example:
        texts  = ["a", "b", "c", "d"], predicted = [0.1, 0.9, 0.5, 0.7], top_n = 2
        result = [b (0.9), d (0.7)]

    Args:
        query: the user query.
        candidates: fused candidates, best-first.
        predict_fn: scores a batch of (query, passage) pairs, e.g. CrossEncoder.predict.
        top_n: how many candidates to return.

    Returns:
        list[Candidate]: re-ranked candidates, best first, with updated scores.
    """
    raise NotImplementedError("TODO: implement")
