"""Doc-level retrieval metrics (HAND-WRITTEN CORE: implement this yourself).

Both functions receive the doc_ids of the ranked *chunks* (so the same doc_id can
appear several times). They first de-duplicate to unique doc_ids, keeping the
first occurrence and its order, and only THEN apply the cutoff ``k``.

Example of why the order matters:
    ranked_chunk_doc_ids = ["a.md", "a.md", "a.md", "b.md"], gold = {"b.md"}, k = 2
    dedupe first -> ["a.md", "b.md"] -> top-2 contains b.md -> hit = 1.0, RR = 1/2
    (cutting first would give ["a.md", "a.md"] and miss it - that is NOT the contract)
"""

from __future__ import annotations


def hit_at_k(ranked_chunk_doc_ids: list[str], gold_docs: set[str], k: int) -> float:
    """Doc-level hit@k.

    Contract:
        * Dedupe ``ranked_chunk_doc_ids`` preserving first occurrence, then keep the
          first ``k`` unique doc_ids.
        * Return 1.0 if any of them is in ``gold_docs``, else 0.0.
        * Empty ranking, empty ``gold_docs`` or ``k <= 0`` return 0.0.
        * ``k`` larger than the number of unique docs simply uses all of them.

    Args:
        ranked_chunk_doc_ids: doc_id of each retrieved chunk, best first.
        gold_docs: doc_ids that answer the question.
        k: cutoff on unique documents.

    Returns:
        float: 1.0 or 0.0.
    """
    raise NotImplementedError("TODO: implement")


def reciprocal_rank(ranked_chunk_doc_ids: list[str], gold_docs: set[str], k: int = 10) -> float:
    """Doc-level reciprocal rank with cutoff k (average over questions = MRR@k).

    Contract:
        * Dedupe ``ranked_chunk_doc_ids`` preserving first occurrence, then keep the
          first ``k`` unique doc_ids.
        * Return 1 / rank (1-based) of the first doc_id that is in ``gold_docs``.
        * Return 0.0 if no gold doc is within the cutoff, if the ranking or
          ``gold_docs`` is empty, or if ``k <= 0``.

    Args:
        ranked_chunk_doc_ids: doc_id of each retrieved chunk, best first.
        gold_docs: doc_ids that answer the question.
        k: cutoff on unique documents (default 10).

    Returns:
        float: in [0.0, 1.0].
    """
    raise NotImplementedError("TODO: implement")
