"""Build the retrieval index from data/corpus/*.md.

Writes:
    index/faiss.index   IndexFlatIP over normalized bge-small embeddings
    index/chunks.jsonl  one chunk per line: chunk_id, doc_id, heading, text
    index/meta.json     backend ("fastembed"), embed model, dim, chunk count, doc count, built_at

Embeddings come from fastembed (ONNX runtime), the same runtime the app uses for
queries. BM25 is not stored: the app rebuilds it from chunks.jsonl at startup.

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
from app.config import EMBED_BACKEND, settings  # noqa: E402
from app.onnx_models import embed_passages, load_embedder  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"


def main() -> int:
    import faiss
    import numpy as np

    chunks = chunk_corpus(CORPUS_DIR)
    if not chunks:
        print(f"No chunks produced: is {CORPUS_DIR} empty?", file=sys.stderr)
        return 1
    doc_count = len({c.doc_id for c in chunks})
    print(f"Chunked {doc_count} docs into {len(chunks)} chunks.")

    print(f"Embedding with {settings.embed_model} ({EMBED_BACKEND}, cache {settings.model_cache_dir}) ...")
    model = load_embedder(settings)
    # Passages: no query prefix. Vectors go into FAISS one batch at a time, so the full
    # corpus matrix is never held alongside the index's own copy.
    batch_size = settings.embed_batch_size
    index = None
    pending: list[np.ndarray] = []
    for vec in embed_passages(model, (embedding_text(c) for c in chunks), batch_size=batch_size):
        pending.append(vec)
        if len(pending) == batch_size:
            index = _add_batch(faiss, np, index, pending)
            print(f"  embedded {index.ntotal}/{len(chunks)}")
            pending = []
    if pending:
        index = _add_batch(faiss, np, index, pending)
    if index is None or index.ntotal != len(chunks):
        got = 0 if index is None else index.ntotal
        print(f"Embedding produced {got} vectors for {len(chunks)} chunks.", file=sys.stderr)
        return 1
    dim = int(index.d)

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    with (out / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    meta = {
        "backend": EMBED_BACKEND,
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


def _add_batch(faiss, np, index, batch):
    """Adds one batch of normalized vectors to an IndexFlatIP, creating it on the first
    batch (the dimension comes from the vectors themselves)."""
    matrix = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
    if index is None:
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    return index


if __name__ == "__main__":
    raise SystemExit(main())
