"""Download both models into the Hugging Face cache (HF_HOME) so they are baked into
the Docker image and the Space does not download them on every cold start.

Usage:  python scripts/prefetch_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def main() -> int:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"HF_HOME={os.getenv('HF_HOME', '(default)')}")
    print(f"Fetching embed model {settings.embed_model}")
    embedder = SentenceTransformer(settings.embed_model, device="cpu")
    vec = embedder.encode(["warm up"], normalize_embeddings=True)
    print(f"  ok, dim={vec.shape[1]}")

    print(f"Fetching re-rank model {settings.rerank_model}")
    cross = CrossEncoder(settings.rerank_model, device="cpu", max_length=512)
    score = cross.predict([("warm up", "warm up")])
    print(f"  ok, sample score={float(score[0]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
