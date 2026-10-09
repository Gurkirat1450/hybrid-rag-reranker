"""Shared data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Candidate:
    chunk_id: int
    doc_id: str
    heading: str
    text: str
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
