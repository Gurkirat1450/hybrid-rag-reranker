"""Retriever: loads the index and both models once, then serves four retrieval modes.

Modes:
    vector         top_k from FAISS (bge-small, inner product on normalized vectors)
    bm25           top_k from BM25Okapi over heading + text
    hybrid         VECTOR_POOL + BM25_POOL fused with reciprocal rank fusion
    hybrid_rerank  hybrid, then the top RERANK_POOL re-scored by a cross-encoder

"vector" and "bm25" never touch the hand-written modules (fusion.py, rerank.py),
so they work while those are still stubs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from app.chunking import Chunk, embedding_text
from app.config import QUERY_PREFIX, Settings, settings as default_settings
from app.fusion import reciprocal_rank_fusion
from app.rerank import rerank
from app.types import Candidate

logger = logging.getLogger(__name__)

MODES: tuple[str, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

T = TypeVar("T")


class IndexUnavailableError(RuntimeError):
    """Index files are missing, corrupt or built with a different embedding model (HTTP 503)."""


class HandWrittenNotImplementedError(RuntimeError):
    """A hand-written core function still raises NotImplementedError (HTTP 501)."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"{filename} is not implemented yet (it still raises NotImplementedError).")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def call_hand_written(filename: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        raise HandWrittenNotImplementedError(filename) from exc


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def load_index_files(index_dir: Path, expected_model: str) -> tuple[dict, list[Chunk]]:
    """Read and validate meta.json + chunks.jsonl. Raises IndexUnavailableError."""
    meta_path = index_dir / "meta.json"
    chunks_path = index_dir / "chunks.jsonl"
    faiss_path = index_dir / "faiss.index"
    missing = [p.name for p in (meta_path, chunks_path, faiss_path) if not p.exists()]
    if missing:
        raise IndexUnavailableError(
            f"Index files missing in '{index_dir}': {', '.join(missing)}. "
            "Run `python scripts/build_index.py` first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("embed_model") != expected_model:
        raise IndexUnavailableError(
            f"Index was built with embed model '{meta.get('embed_model')}' but EMBED_MODEL is "
            f"'{expected_model}'. Rebuild with `python scripts/build_index.py` or fix EMBED_MODEL."
        )

    chunks: list[Chunk] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                chunks.append(Chunk(row["chunk_id"], row["doc_id"], row["heading"], row["text"]))
    for i, c in enumerate(chunks):
        if c.chunk_id != i:
            raise IndexUnavailableError(f"chunks.jsonl is corrupt: line {i} has chunk_id {c.chunk_id}.")
    if meta.get("chunk_count") != len(chunks):
        raise IndexUnavailableError(
            f"meta.json says {meta.get('chunk_count')} chunks but chunks.jsonl has {len(chunks)}. Rebuild the index."
        )
    return meta, chunks


class Retriever:
    def __init__(self, cfg: Settings = default_settings) -> None:
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.cfg = cfg
        self.meta, self.chunks = load_index_files(cfg.index_dir, cfg.embed_model)

        self.index = faiss.read_index(str(cfg.index_dir / "faiss.index"))
        if self.index.ntotal != len(self.chunks) or self.index.d != self.meta.get("dim"):
            raise IndexUnavailableError(
                f"faiss.index has {self.index.ntotal} vectors of dim {self.index.d}, expected "
                f"{len(self.chunks)} of dim {self.meta.get('dim')}. Rebuild the index."
            )

        self.bm25 = BM25Okapi([tokenize(embedding_text(c)) for c in self.chunks])

        logger.info("Loading embed model %s", cfg.embed_model)
        self.embedder = SentenceTransformer(cfg.embed_model, device="cpu")
        dim = self.embedder.get_sentence_embedding_dimension()
        if dim != self.index.d:
            raise IndexUnavailableError(
                f"Embed model '{cfg.embed_model}' produces {dim}-d vectors but the index is {self.index.d}-d."
            )
        logger.info("Loading re-rank model %s", cfg.rerank_model)
        self.cross_encoder = CrossEncoder(cfg.rerank_model, device="cpu", max_length=512)

    # ------------------------------------------------------------------ stages

    def _vector(self, query: str, n: int, timings: dict[str, float]) -> list[tuple[int, float]]:
        t = time.perf_counter()
        q = self.embedder.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        timings["embed_ms"] = _ms(t)

        t = time.perf_counter()
        n = min(n, self.index.ntotal)
        scores, ids = self.index.search(q, n)
        timings["vector_search_ms"] = _ms(t)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _bm25(self, query: str, n: int, timings: dict[str, float]) -> list[tuple[int, float]]:
        t = time.perf_counter()
        tokens = tokenize(query)
        hits: list[tuple[int, float]] = []
        if tokens:
            scores = self.bm25.get_scores(tokens)
            order = np.argsort(-scores, kind="stable")[:n]
            # Chunks with no matching term (score <= 0) are not real BM25 hits.
            hits = [(int(i), float(scores[i])) for i in order if scores[i] > 0]
        timings["bm25_ms"] = _ms(t)
        return hits

    def _candidate(self, chunk_id: int, score: float, vector_rank: int | None, bm25_rank: int | None) -> Candidate:
        c = self.chunks[chunk_id]
        return Candidate(
            chunk_id=chunk_id,
            doc_id=c.doc_id,
            heading=c.heading,
            text=c.text,
            score=float(score),
            vector_rank=vector_rank,
            bm25_rank=bm25_rank,
        )

    def _fuse(
        self,
        vec: list[tuple[int, float]],
        bm: list[tuple[int, float]],
        timings: dict[str, float],
    ) -> list[Candidate]:
        t = time.perf_counter()
        vrank = {cid: r for r, (cid, _) in enumerate(vec, start=1)}
        brank = {cid: r for r, (cid, _) in enumerate(bm, start=1)}
        fused = call_hand_written(
            "app/fusion.py",
            reciprocal_rank_fusion,
            [[cid for cid, _ in vec], [cid for cid, _ in bm]],
            k=self.cfg.rrf_k,
        )
        out = [self._candidate(cid, score, vrank.get(cid), brank.get(cid)) for cid, score in fused]
        timings["fusion_ms"] = _ms(t)
        return out

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.cross_encoder.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return [float(s) for s in np.atleast_1d(scores)]

    def _rerank(self, query: str, pool: list[Candidate], top_k: int, timings: dict[str, float]) -> list[Candidate]:
        t = time.perf_counter()
        # Copies, because rerank() overwrites candidate.score in place.
        candidates = [replace(c) for c in pool]
        out = call_hand_written("app/rerank.py", rerank, query, candidates, predict_fn=self._predict, top_n=top_k)
        timings["rerank_ms"] = _ms(t)
        return list(out)

    # ------------------------------------------------------------------ public

    def search(self, query: str, mode: str, top_k: int) -> tuple[list[Candidate], dict[str, float]]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        timings: dict[str, float] = {}
        start = time.perf_counter()
        if mode == "vector":
            hits = self._vector(query, top_k, timings)
            results = [self._candidate(cid, s, r, None) for r, (cid, s) in enumerate(hits, start=1)]
        elif mode == "bm25":
            hits = self._bm25(query, top_k, timings)
            results = [self._candidate(cid, s, None, r) for r, (cid, s) in enumerate(hits, start=1)]
        else:
            vec = self._vector(query, max(self.cfg.vector_pool, top_k), timings)
            bm = self._bm25(query, max(self.cfg.bm25_pool, top_k), timings)
            fused = self._fuse(vec, bm, timings)
            if mode == "hybrid":
                results = fused[:top_k]
            else:
                pool = fused[: max(self.cfg.rerank_pool, top_k)]
                results = self._rerank(query, pool, top_k, timings)
        timings["total_ms"] = _ms(start)
        return results, timings

    def compare(self, query: str, top_k: int) -> tuple[dict[str, list[Candidate]], dict[str, float]]:
        """vector / hybrid / hybrid_rerank for one query, sharing the retrieval pools."""
        timings: dict[str, float] = {}
        start = time.perf_counter()
        vec = self._vector(query, max(self.cfg.vector_pool, top_k), timings)
        bm = self._bm25(query, max(self.cfg.bm25_pool, top_k), timings)
        brank = {cid: r for r, (cid, _) in enumerate(bm, start=1)}
        vector_results = [
            self._candidate(cid, s, r, brank.get(cid)) for r, (cid, s) in enumerate(vec[:top_k], start=1)
        ]
        fused = self._fuse(vec, bm, timings)
        reranked = self._rerank(query, fused[: max(self.cfg.rerank_pool, top_k)], top_k, timings)
        timings["total_ms"] = _ms(start)
        return {"vector": vector_results, "hybrid": fused[:top_k], "hybrid_rerank": reranked}, timings

    def available_modes(self) -> list[str]:
        """Modes that currently work (probes the hand-written functions cheaply)."""
        modes = ["vector", "bm25"]
        try:
            reciprocal_rank_fusion([[0]], k=self.cfg.rrf_k)
        except NotImplementedError:
            return modes
        except Exception:  # implemented but buggy: let real requests surface the error
            pass
        modes.append("hybrid")
        try:
            probe = Candidate(chunk_id=0, doc_id="", heading="", text="", score=0.0, vector_rank=None, bm25_rank=None)
            rerank("probe", [probe], predict_fn=lambda pairs: [0.0] * len(pairs), top_n=1)
        except NotImplementedError:
            return modes
        except Exception:
            pass
        modes.append("hybrid_rerank")
        return modes
