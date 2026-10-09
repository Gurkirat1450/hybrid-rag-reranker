"""Build the retrieval index from data/corpus/*.md.

Writes:
    index/faiss.index   IndexFlatIP over normalized bge-small embeddings
    index/chunks.jsonl  one chunk per line: chunk_id, doc_id, heading, text
    index/meta.json     embed model, dim, chunk count, doc count, built_at

BM25 is not stored: the app rebuilds it from chunks.jsonl at startup.

Usage:  python scripts/build_index.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import MAX_CHARS, MIN_CHARS, OVERLAP_CHARS, chunk_corpus, embedding_text  # noqa: E402
from app.config import settings  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"


def main() -> int:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = chunk_corpus(CORPUS_DIR)
    if not chunks:
        print(f"No chunks produced: is {CORPUS_DIR} empty?", file=sys.stderr)
        return 1
    doc_count = len({c.doc_id for c in chunks})
    print(f"Chunked {doc_count} docs into {len(chunks)} chunks.")

    print(f"Embedding with {settings.embed_model} ...")
    model = SentenceTransformer(settings.embed_model, device="cpu")
    vectors = model.encode(
        [embedding_text(c) for c in chunks],  # passages: no query prefix
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    dim = int(vectors.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    with (out / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    meta = {
        "embed_model": settings.embed_model,
        "dim": dim,
        "chunk_count": len(chunks),
        "doc_count": doc_count,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunking": {"max_chars": MAX_CHARS, "overlap_chars": OVERLAP_CHARS, "min_chars": MIN_CHARS},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out / 'faiss.index'} ({index.ntotal} vectors, dim {dim})")
    print(f"chunks: {len(chunks)}  docs: {doc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
