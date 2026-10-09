"""Runtime configuration. Every value can be overridden by an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    embed_model: str
    rerank_model: str
    vector_pool: int
    bm25_pool: int
    rerank_pool: int
    rrf_k: int
    max_top_k: int
    max_query_chars: int
    rate_limit_per_min: int
    index_dir: Path
    trust_proxy: bool


def load_settings() -> Settings:
    return Settings(
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        vector_pool=_int("VECTOR_POOL", 30),
        bm25_pool=_int("BM25_POOL", 30),
        rerank_pool=_int("RERANK_POOL", 20),
        rrf_k=_int("RRF_K", 60),
        max_top_k=_int("MAX_TOP_K", 10),
        max_query_chars=_int("MAX_QUERY_CHARS", 300),
        rate_limit_per_min=_int("RATE_LIMIT_PER_MIN", 20),
        index_dir=_path("INDEX_DIR", "index"),
        trust_proxy=_bool("TRUST_PROXY", False),
    )


settings = load_settings()

# bge-small-en-v1.5 retrieval instruction: applied to queries only, never to passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
