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
    if not candidates or top_n <= 0:
        return []
    pairs = [(query, candidate.text) for candidate in candidates]
    scores = predict_fn(pairs)
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    return ranked[:top_n]
