from __future__ import annotations


def _top_unique_docs(ranked_chunk_doc_ids: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    unique_docs = list(dict.fromkeys(ranked_chunk_doc_ids))
    return unique_docs[:k]


def hit_at_k(ranked_chunk_doc_ids: list[str], gold_docs: set[str], k: int) -> float:
    if not gold_docs:
        return 0.0
    for doc_id in _top_unique_docs(ranked_chunk_doc_ids, k):
        if doc_id in gold_docs:
            return 1.0
    return 0.0


def reciprocal_rank(ranked_chunk_doc_ids: list[str], gold_docs: set[str], k: int = 10) -> float:
    if not gold_docs:
        return 0.0
    for rank, doc_id in enumerate(_top_unique_docs(ranked_chunk_doc_ids, k), start=1):
        if doc_id in gold_docs:
            return 1.0 / rank
    return 0.0
