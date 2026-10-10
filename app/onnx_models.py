"""fastembed (ONNX runtime) model loading and embedding, shared by the app,
scripts/build_index.py and scripts/prefetch_models.py so all three use the same
cache directory, thread count and prefix rules.

Imports only app.config and fastembed, so the Docker model-prefetch layer can copy
this file without the rest of the app.

Query prefix: fastembed does NOT add the bge instruction itself (for bge-small-en-v1.5,
TextEmbedding.query_embed() just calls embed()). It is added here, explicitly and once,
via embed() rather than query_embed(), so a future fastembed release that starts
prefixing inside query_embed() can't double it. Passages go through embed() untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np

from app.config import QUERY_PREFIX, Settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder


def load_embedder(cfg: Settings) -> "TextEmbedding":
    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=cfg.embed_model,
        cache_dir=str(cfg.model_cache_dir),
        threads=cfg.onnx_threads,
    )


def load_cross_encoder(cfg: Settings) -> "TextCrossEncoder":
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(
        model_name=cfg.rerank_model,
        cache_dir=str(cfg.model_cache_dir),
        threads=cfg.onnx_threads,
    )


def embed_query(embedder: "TextEmbedding", query: str) -> np.ndarray:
    """(1, dim) float32, L2-normalized (fastembed normalizes bge output)."""
    vec = next(iter(embedder.embed([QUERY_PREFIX + query], batch_size=1)))
    return np.asarray(vec, dtype=np.float32).reshape(1, -1)


def embed_passages(embedder: "TextEmbedding", texts: Iterable[str], batch_size: int) -> Iterator[np.ndarray]:
    """One (dim,) float32 vector per passage, in input order, no prefix. A generator:
    callers consume it batch by batch instead of materialising every vector at once.
    parallel is left unset so fastembed stays in-process (parallel workers would each
    load their own copy of the model)."""
    for vec in embedder.embed(texts, batch_size=batch_size):
        yield np.asarray(vec, dtype=np.float32)


def embedding_dim(embedder: "TextEmbedding") -> int:
    """Measured from a real forward pass (doubles as a warm-up) rather than a lookup."""
    return int(next(iter(embedder.embed(["warm up"], batch_size=1))).shape[-1])
