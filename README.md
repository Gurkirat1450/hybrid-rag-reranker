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

Hybrid search over the FastAPI tutorial docs. It combines BM25 and dense vectors with reciprocal rank fusion, then re-ranks the results with a cross-encoder, and it shows all three stages side by side. It only does retrieval: there's no LLM and no API keys.

**Live demo:** `https://huggingface.co/spaces/<your-user>/<your-space>` *(placeholder)*

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
                                                     cross-encoder/ms-marco-MiniLM-L-6-v2
                                                                         │
                                                                         ▼
                                                                  top_k results
```

- **Chunking** (`app/chunking.py`): the docs are split on `##`/`###` headings, and each chunk's heading reads "Doc title > section". mkdocs include and admonition lines are removed. Long sections (over about 1800 chars) are split at paragraph breaks with about 200 chars of overlap. Short sections (under about 200 chars) are merged into the previous chunk.
- **Vector**: queries get the bge instruction prefix and passages don't. Embeddings are normalized, so FAISS `IndexFlatIP` scores are cosine similarities.
- **BM25**: rebuilt at startup from `index/chunks.jsonl`. Tokens are lowercase `[a-z0-9_]+` (this keeps `check_same_thread` and `exclude_unset` intact). Nothing is pickled.
- **RRF** (`app/fusion.py`), **re-rank** (`app/rerank.py`) and **metrics** (`app/metrics.py`) are written by hand. Until they're implemented the API returns `501` for the modes that need them, while `vector` and `bm25` keep working.

API: `GET /health`, `POST /search` (`mode` = `vector | bm25 | hybrid | hybrid_rerank`), `POST /compare`, `GET /eval/results`. Errors always come back as `{"error", "detail"}`.

## Results

Paste the table that `python scripts/run_eval.py` prints:

| Mode | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|
| vector | – | – | – | – |
| bm25 | – | – | – | – |
| hybrid | – | – | – | – |
| hybrid_rerank | – | – | – | – |

**How to read it:** hit@k is the share of questions where at least one correct document shows up among the first k *distinct* documents returned. MRR@10 averages 1/rank of the first correct document, so higher is better and 1.0 means the correct doc always came first.

**Honest limits:** labels are doc-level, so a hit means the right page was found, not necessarily the exact paragraph. There are only 20 hand-written questions over a 40-file corpus, so one question moves a metric by 0.05, and small differences between modes are noise. The questions were written by reading the same docs, which can favour keyword overlap even though a third of them are deliberately paraphrased.

## Run locally

Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # pulls torch via sentence-transformers

pytest                                 # fusion/rerank/metrics tests fail until you implement them
python scripts/build_index.py          # writes index/ (prints chunk and doc counts)
uvicorn app.main:app --reload --port 8000   # open http://127.0.0.1:8000
python scripts/run_eval.py             # writes results/eval_results.json and prints the table
```

Configuration comes from environment variables (see `.env.example`): pool sizes, `RRF_K`, `RATE_LIMIT_PER_MIN`, `INDEX_DIR`, `TRUST_PROXY`. If you change `EMBED_MODEL`, rebuild the index. The app won't serve an index built with a different model (it returns 503).

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
4. Watch the **Logs** tab. The build installs CPU torch, then the requirements, then downloads both models into the image.

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

Models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT) and [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (Apache-2.0).
