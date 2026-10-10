"""Run data/eval/questions.jsonl through all four modes and score them.

Metrics (doc-level, from app/metrics.py): hit@1, hit@3, hit@5, MRR@10.
Each mode retrieves the top 10 chunks; metrics dedupe chunk doc_ids first.

Writes results/eval_results.json and prints a markdown table.
A mode whose hand-written function is still a stub is reported and skipped.

Usage:  python scripts/run_eval.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.metrics import hit_at_k, reciprocal_rank  # noqa: E402
from app.retrieval import (  # noqa: E402
    MODES,
    HandWrittenNotImplementedError,
    IndexUnavailableError,
    RerankDisabledError,
    Retriever,
)

QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
OUT = ROOT / "results" / "eval_results.json"
TOP_K = 10
METRIC_KEYS = ("hit@1", "hit@3", "hit@5", "mrr@10")


def load_questions() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def unique_docs(doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def score(doc_ids: list[str], gold: set[str]) -> dict[str, float]:
    return {
        "hit@1": hit_at_k(doc_ids, gold, 1),
        "hit@3": hit_at_k(doc_ids, gold, 3),
        "hit@5": hit_at_k(doc_ids, gold, 5),
        "mrr@10": reciprocal_rank(doc_ids, gold, 10),
    }


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def main() -> int:
    questions = load_questions()

    try:
        score(["probe.md"], {"probe.md"})
    except NotImplementedError:
        print("app/metrics.py is not implemented yet (hit_at_k / reciprocal_rank). Nothing can be scored.")
        return 1

    print(f"Loading retriever ({settings.embed_model}, {settings.rerank_model}) ...")
    try:
        retriever = Retriever(settings)
    except IndexUnavailableError as exc:
        print(f"Cannot load the index: {exc}")
        return 1

    modes_out: dict[str, dict] = {}
    per_question: list[dict] = [
        {"id": q["id"], "type": q["type"], "question": q["question"], "gold_docs": q["gold_docs"], "modes": {}}
        for q in questions
    ]

    for mode in MODES:
        rows: list[dict[str, float]] = []
        latencies: list[float] = []
        try:
            for q, pq in zip(questions, per_question):
                gold = set(q["gold_docs"])
                t = time.perf_counter()
                results, _ = retriever.search(q["question"], mode, TOP_K)
                latencies.append((time.perf_counter() - t) * 1000.0)
                doc_ids = [c.doc_id for c in results]
                m = score(doc_ids, gold)
                rows.append(m)
                rr = m["mrr@10"]
                pq["modes"][mode] = {
                    **m,
                    "first_gold_rank": round(1 / rr) if rr > 0 else None,
                    "top_docs": unique_docs(doc_ids)[:5],
                }
        except HandWrittenNotImplementedError as exc:
            print(f"[skip] mode '{mode}': {exc}")
            modes_out[mode] = {"status": "not_implemented", "detail": str(exc)}
            for pq in per_question:
                pq["modes"].pop(mode, None)
            continue
        except RerankDisabledError as exc:
            # ENABLE_RERANK=0: run the eval with re-ranking enabled to measure this mode.
            print(f"[skip] mode '{mode}': {exc}")
            modes_out[mode] = {"status": "disabled", "detail": str(exc)}
            for pq in per_question:
                pq["modes"].pop(mode, None)
            continue

        agg: dict = {"status": "ok"}
        for key in METRIC_KEYS:
            agg[key] = mean([r[key] for r in rows])
        by_type: dict[str, dict[str, float]] = {}
        for qtype in sorted({q["type"] for q in questions}):
            idx = [i for i, q in enumerate(questions) if q["type"] == qtype]
            by_type[qtype] = {key: mean([rows[i][key] for i in idx]) for key in METRIC_KEYS}
            by_type[qtype]["n"] = len(idx)
        agg["by_type"] = by_type
        agg["mean_latency_ms"] = round(statistics.fmean(latencies), 1) if latencies else None
        modes_out[mode] = agg

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "num_questions": len(questions),
        "top_k_chunks": TOP_K,
        "config": {
            "vector_pool": settings.vector_pool,
            "bm25_pool": settings.bm25_pool,
            "rerank_pool": settings.rerank_pool,
            "rrf_k": settings.rrf_k,
        },
        "modes": modes_out,
        "questions": per_question,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"| Mode | hit@1 | hit@3 | hit@5 | MRR@10 |  ({len(questions)} questions, doc-level)")
    print("|---|---|---|---|---|")
    for mode in MODES:
        m = modes_out[mode]
        if m["status"] != "ok":
            print(f"| {mode} | n/a | n/a | n/a | n/a |")
        else:
            print(f"| {mode} | {m['hit@1']:.3f} | {m['hit@3']:.3f} | {m['hit@5']:.3f} | {m['mrr@10']:.3f} |")
    print()
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
