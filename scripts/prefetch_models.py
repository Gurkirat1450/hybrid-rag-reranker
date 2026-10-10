"""Download both fastembed (ONNX) models into the model cache so they are baked into
the Docker image and the app does not download them on every cold start.

The cache directory is settings.model_cache_dir (env FASTEMBED_CACHE_PATH, default
./.fastembed_cache), the same directory the app and build_index.py pass to fastembed.

Usage:  python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.onnx_models import embedding_dim, load_cross_encoder, load_embedder  # noqa: E402


def main() -> int:
    print(f"Model cache: {settings.model_cache_dir}")
    print(f"Fetching embed model {settings.embed_model}")
    embedder = load_embedder(settings)
    print(f"  ok, dim={embedding_dim(embedder)}")
    del embedder

    print(f"Fetching re-rank model {settings.rerank_model}")
    cross = load_cross_encoder(settings)
    score = next(iter(cross.rerank("warm up", ["warm up"])))
    print(f"  ok, sample score={float(score):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
