---
title: Hybrid RAG Reranker
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# hybrid-rag-reranker

Hybrid retrieval over the FastAPI docs: BM25 + dense vectors fused with Reciprocal Rank Fusion, then re-ranked by a cross-encoder, with a side-by-side UI and a measured evaluation. Retrieval only, no LLM and no API keys.

Both models run on the ONNX runtime through [fastembed](https://github.com/qdrant/fastembed), with no torch dependency, so the image stays small enough for free CPU hosting.

**Live demo:** https://huggingface.co/spaces/<your-username>/<space-name>

![screenshot](docs/screenshot.png) *(placeholder: add a screenshot after deploying)*

## Architecture

```
                 ┌──────────────────────────────┐
  query ────────►│ BM25 (rank-bm25, in memory)  │── top BM25_POOL (30) ──┐
    │            └──────────────────────────────┘                        │
    │            ┌──────────────────────────────┐                        ▼
    └──────────► │ bge-small-en-v1.5 + FAISS IP │── top VECTOR_POOL (30) ─► Reciprocal Rank Fusion (k=60)
                 └──────────────────────────────┘                        │
                                                                         ▼  top RERANK_POOL (20)
                                                  ms-marco-MiniLM-L-6-v2 (ONNX, fastembed)
                                                                         │
                                                                         ▼
                                                                  top_k results
```

- **Chunking** (`app/chunking.py`): the docs are split on `##`/`###` headings, and each chunk's heading reads "Doc title > section". mkdocs include and admonition lines are removed. Long sections (over about 1800 chars) are split at paragraph breaks with about 200 chars of overlap. Short sections (under about 200 chars) are merged into the previous chunk.
- **Models** (`app/onnx_models.py`): fastembed `TextEmbedding` (`BAAI/bge-small-en-v1.5`, 384-d) and `TextCrossEncoder` (`Xenova/ms-marco-MiniLM-L-6-v2`), each loaded once at startup with a small ONNX thread count (`ONNX_THREADS`, default 1) and small batches (`EMBED_BATCH_SIZE`, `RERANK_BATCH_SIZE`).
- **Vector**: queries get the bge instruction prefix and passages don't. fastembed doesn't add the prefix itself, so the app adds it once. Embeddings are normalized, so FAISS `IndexFlatIP` scores are cosine similarities.
- **BM25**: rebuilt at startup from `index/chunks.jsonl`. Tokens are lowercase `[a-z0-9_]+` (this keeps `check_same_thread` and `exclude_unset` intact). Nothing is pickled.
- **RRF** (`app/fusion.py`), **re-rank** (`app/rerank.py`) and **metrics** (`app/metrics.py`) are written by hand. Until they're implemented the API returns `501` for the modes that need them, while `vector` and `bm25` keep working.

API: `GET /health`, `POST /search` (`mode` = `vector | bm25 | hybrid | hybrid_rerank`), `POST /compare`, `GET /eval/results`. Errors always come back as `{"error", "detail"}`.

## Results

20 hand-written questions over 40 FastAPI tutorial docs, scored at doc level (7 keyword, 7 paraphrase, 6 multi-concept).

| Mode | hit@1 | hit@3 | hit@5 | MRR@10 |  (20 questions, doc-level)
|---|---|---|---|---|
| vector | 0.800 | 0.950 | 1.000 | 0.879 |
| bm25 | 0.950 | 1.000 | 1.000 | 0.967 |
| hybrid | 0.950 | 1.000 | 1.000 | 0.975 |
| hybrid_rerank | 0.950 | 1.000 | 1.000 | 0.975 |

Latency is the mean per query on a local CPU run and will be higher on the free Hugging Face CPU tier.

Results were produced with the fastembed ONNX runtime (not sentence-transformers/torch). fastembed's bge-small is an optimized ONNX export of the same model, so scores can differ slightly from a torch run.

### How to read this

BM25 is a strong baseline here: the FastAPI docs are full of exact identifiers, and even the paraphrased questions share vocabulary with the answer text. Hybrid matches it (MRR 0.975 vs 0.967). Re-ranking produced no net gain while making queries roughly 75x slower than hybrid.

Re-ranking did change individual results:
- **Helped:** q15 ("How do I declare path, query and request body parameters in the same endpoint, and how does FastAPI tell which is which?") moved the correct doc from rank 2 to rank 1. The cross-encoder reads the whole long question against each chunk, which is what multi-concept queries need.
- **Hurt:** q11 ("How can I avoid leaking the plaintext password when returning the user I just created?") moved the correct doc from rank 1 to rank 2, because another doc was a closer topical match to the wording.

These two cancel out in the aggregate.

## Honest limits

- **Small eval:** 20 questions, 6-7 per type. One question moves a within-type metric by 0.14-0.17, so these results are indicative, not statistically significant.
- **Doc-level labels:** a hit means the right *document* was in the top k, not that the right *passage* was. Several docs can partly answer a question, so a "miss" is sometimes a labeling limit.
- **Paraphrase set is mild:** the questions avoid the headings' wording but still share vocabulary with the answers, which favours BM25. A harder paraphrase set would likely favour dense retrieval.
- **Small, technical corpus:** 40 docs (305 chunks) of identifier-heavy text. Fusion and re-ranking are expected to matter more on larger, more varied corpora, which this eval does not test.
- **Retrieval only:** the app returns ranked passages. It does not generate answers.

## Run locally

Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # fastembed + onnxruntime, no torch

pytest                                 # fusion/rerank/metrics tests fail until you implement them
python scripts/prefetch_models.py      # optional: downloads both ONNX models into .fastembed_cache/
python scripts/build_index.py          # writes index/ (prints chunk and doc counts)
uvicorn app.main:app --reload --port 8000   # open http://127.0.0.1:8000
python scripts/run_eval.py             # writes results/eval_results.json and prints the table
```

Configuration comes from environment variables (see `.env.example`): pool sizes, `RRF_K`, `RATE_LIMIT_PER_MIN`, `INDEX_DIR`, `TRUST_PROXY`, `FASTEMBED_CACHE_PATH`, `ONNX_THREADS` and the batch sizes. If you change `EMBED_MODEL`, rebuild the index. The app won't serve an index built with a different model or by a different embedding runtime (it returns 503). An index built before the fastembed switch has no `"backend": "fastembed"` in `meta.json`, so rebuild it once.

## Deploy to Hugging Face Spaces

The Docker build **doesn't build the index**. It copies whatever is in the repo, so build and commit first:

```bash
python scripts/build_index.py
python scripts/run_eval.py
git add index/ results/
git commit -m "Build index and eval results"
```

Then:

1. On huggingface.co, go to **New Space**, pick the **Docker** SDK and the **Blank** template, and choose free **CPU basic** hardware.
2. Create a token with **write** access under *Settings → Access Tokens*.
3. Add the Space as a remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-user>/<your-space>
   git push space main
   # username: your HF username, password: the write token
   ```
   If the Space was created with an initial commit, run `git pull space main --allow-unrelated-histories` first, or use `git push --force space main`.
4. Watch the **Logs** tab. The build installs the requirements (fastembed and onnxruntime, no torch), then downloads both ONNX models into the image.

`index/faiss.index` is a small binary file (chunk count × 384 float32, well under 1 MB). The Hugging Face Hub can reject plain-git binary files on push. If that happens, track it with Git LFS before committing: `git lfs install && git lfs track "index/faiss.index" "*.png"`, then commit `.gitattributes`.

### Verify it works

- [ ] `GET /health` returns `{"status":"ok","chunks":N,...}`, and N matches what `build_index.py` printed.
- [ ] `mode_available` lists all four modes (if it doesn't, a hand-written module is still a stub).
- [ ] Clicking an example chip fills all three columns with results, badges and timings.
- [ ] The "Eval results" panel shows the same numbers as your local `run_eval.py`.
- [ ] Sending more than 20 `/search` calls within a minute returns a 429 with a `Retry-After` header.
- [ ] **Rate limiting behind HF's proxy is an assumption I haven't verified.** With `TRUST_PROXY=1` (set in the Dockerfile) the limiter keys on the **last** `X-Forwarded-For` entry. I assume that's the address HF's nearest proxy appends for the real client. If the logs show every request sharing one IP (so all users share one budget), or if the IP changes on every request, adjust `resolve_client_ip` in `app/ratelimit.py`.
- [ ] Validation: an empty query or `top_k: 11` returns a 422 JSON error.

### Free-tier caveats

- Free CPU Spaces **go to sleep after a period of inactivity**. The first request afterwards triggers a cold start (the container boots and loads both models), which can take tens of seconds.
- Everything runs on shared CPU. Cross-encoder re-ranking of 20 candidates is the slowest stage; check `rerank_ms` in the timings.
- Rate-limit state is in memory, so it resets whenever the Space restarts.

## Corpus attribution

`data/corpus/` contains the top-level tutorial pages of the [FastAPI documentation](https://github.com/fastapi/fastapi/tree/master/docs/en/docs/tutorial), copyright © Sebastián Ramírez, licensed under the [MIT License](https://github.com/fastapi/fastapi/blob/master/LICENSE). The files were copied flat, with no subfolders. This project isn't affiliated with FastAPI.

Models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT) and [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (Apache-2.0), run as the ONNX exports fastembed downloads ([Qdrant/bge-small-en-v1.5-onnx-Q](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q) and [Xenova/ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)).
