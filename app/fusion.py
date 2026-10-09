"""Reciprocal rank fusion (HAND-WRITTEN CORE: implement this yourself)."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several best-first rankings of chunk_ids with Reciprocal Rank Fusion.

    Contract:
        * ``rankings`` is a list of rankings; each ranking is a list of chunk_ids
          ordered best-first. Rankings may have different lengths and may be empty.
        * Ranks are 1-based: the first element of a ranking has rank 1.
        * For every chunk_id that appears in at least one ranking:
              score(id) = sum over rankings containing id of 1 / (k + rank_in_that_ranking)
          A ranking that does not contain the id contributes nothing (not zero-rank).
        * Return a list of (chunk_id, score) tuples covering every id seen, sorted by
          score descending; ties are broken by the smaller chunk_id first.
        * ``rankings == []`` or all-empty rankings return ``[]``.

    Example (k=1):
        rankings = [[1, 2, 3], [3, 4]]
        1 -> 1/(1+1)               = 0.5
        2 -> 1/(1+2)               = 0.333...
        3 -> 1/(1+3) + 1/(1+1)     = 0.75
        4 -> 1/(1+2)               = 0.333...
        result = [(3, 0.75), (1, 0.5), (2, 0.333...), (4, 0.333...)]

    Args:
        rankings: best-first lists of chunk_ids (e.g. [vector_ids, bm25_ids]).
        k: RRF damping constant (default 60, as in Cormack et al. 2009).

    Returns:
        list[tuple[int, float]]: (chunk_id, fused_score), best first.
    """
    raise NotImplementedError("TODO: implement")
