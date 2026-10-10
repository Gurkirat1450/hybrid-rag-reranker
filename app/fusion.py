"""Reciprocal rank fusion (HAND-WRITTEN CORE: implement this yourself)."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        already_scored: set[int] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in already_scored:
                continue
            already_scored.add(chunk_id)
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
