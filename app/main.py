"""FastAPI app: search API, side-by-side comparison, eval results and a static UI."""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import PROJECT_ROOT, settings
from app.ratelimit import SlidingWindowRateLimiter, resolve_client_ip
from app.retrieval import HandWrittenNotImplementedError, IndexUnavailableError, Retriever
from app.types import Candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hybrid_rag")

STATIC_DIR = Path(__file__).resolve().parent / "static"
EVAL_RESULTS = PROJECT_ROOT / "results" / "eval_results.json"

SEARCH_COST = 1
COMPARE_COST = 3

Mode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class RateLimitedError(Exception):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = None
    app.state.startup_error = None
    try:
        app.state.retriever = Retriever(settings)
        logger.info("Retriever ready: %d chunks", len(app.state.retriever.chunks))
    except IndexUnavailableError as exc:
        # Keep serving so the UI and /health can explain the problem (503).
        app.state.startup_error = str(exc)
        logger.error("STARTUP FAILED: %s", exc)
    yield


app = FastAPI(title="hybrid-rag-reranker", version="1.0.0", lifespan=lifespan)
limiter = SlidingWindowRateLimiter(limit=settings.rate_limit_per_min, window_seconds=60.0)


# ---------------------------------------------------------------------- schemas


class _QueryModel(BaseModel):
    query: str = Field(..., min_length=1, max_length=settings.max_query_chars)
    top_k: int = Field(5, ge=1, le=settings.max_top_k)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SearchRequest(_QueryModel):
    mode: Mode = "hybrid_rerank"


class CompareRequest(_QueryModel):
    pass


class ResultItem(BaseModel):
    rank: int
    doc_id: str
    heading: str
    text: str
    score: float
    vector_rank: int | None
    bm25_rank: int | None


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[ResultItem]
    timings_ms: dict[str, float]


class CompareResponse(BaseModel):
    query: str
    vector: list[ResultItem]
    hybrid: list[ResultItem]
    hybrid_rerank: list[ResultItem]
    timings_ms: dict[str, float]


def _items(candidates: list[Candidate]) -> list[ResultItem]:
    return [
        ResultItem(
            rank=i,
            doc_id=c.doc_id,
            heading=c.heading,
            text=c.text,
            score=round(float(c.score), 6),
            vector_rank=c.vector_rank,
            bm25_rank=c.bm25_rank,
        )
        for i, c in enumerate(candidates, start=1)
    ]


# ---------------------------------------------------------------------- helpers


def _error(status: int, error: str, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail}, headers=headers)


def _client_ip(request: Request) -> str:
    forwarded = ",".join(request.headers.getlist("x-forwarded-for")) or None
    host = request.client.host if request.client else None
    return resolve_client_ip(host, forwarded, settings.trust_proxy)


def _charge(request: Request, cost: int) -> None:
    allowed, retry_after = limiter.hit(_client_ip(request), cost)
    if not allowed:
        raise RateLimitedError(retry_after)


def _retriever(request: Request) -> Retriever:
    r = request.app.state.retriever
    if r is None:
        raise IndexUnavailableError(request.app.state.startup_error or "Index is not loaded.")
    return r


# ---------------------------------------------------------------------- error handlers


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return _error(422, "validation_error", "; ".join(parts) or "Invalid request.")


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error(exc.status_code, "http_error", str(exc.detail), headers=getattr(exc, "headers", None))


@app.exception_handler(RateLimitedError)
async def _rate_limit_handler(request: Request, exc: RateLimitedError) -> JSONResponse:
    seconds = max(1, math.ceil(exc.retry_after))
    return _error(
        429,
        "rate_limited",
        f"Rate limit of {settings.rate_limit_per_min} units/minute exceeded "
        f"(/search costs {SEARCH_COST}, /compare costs {COMPARE_COST}). Retry in {seconds}s.",
        headers={"Retry-After": str(seconds)},
    )


@app.exception_handler(IndexUnavailableError)
async def _index_handler(request: Request, exc: IndexUnavailableError) -> JSONResponse:
    return _error(503, "index_unavailable", str(exc))


@app.exception_handler(HandWrittenNotImplementedError)
async def _not_implemented_handler(request: Request, exc: HandWrittenNotImplementedError) -> JSONResponse:
    return _error(501, "not_implemented", f"{exc} Implement it in {exc.filename} and restart the server.")


@app.exception_handler(Exception)
async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return _error(500, "internal_error", "Internal server error. See server logs.")


# ---------------------------------------------------------------------- routes


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health(request: Request):
    r = _retriever(request)
    return {"status": "ok", "chunks": len(r.chunks), "mode_available": r.available_modes()}


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    _charge(request, SEARCH_COST)
    r = _retriever(request)
    results, timings = r.search(body.query, body.mode, body.top_k)
    return SearchResponse(query=body.query, mode=body.mode, results=_items(results), timings_ms=timings)


@app.post("/compare", response_model=CompareResponse)
def compare(body: CompareRequest, request: Request) -> CompareResponse:
    _charge(request, COMPARE_COST)
    r = _retriever(request)
    columns, timings = r.compare(body.query, body.top_k)
    return CompareResponse(
        query=body.query,
        vector=_items(columns["vector"]),
        hybrid=_items(columns["hybrid"]),
        hybrid_rerank=_items(columns["hybrid_rerank"]),
        timings_ms=timings,
    )


@app.get("/eval/results")
async def eval_results():
    if not EVAL_RESULTS.exists():
        return _error(
            404,
            "not_found",
            "results/eval_results.json not found. Run `python scripts/run_eval.py` and commit results/.",
        )
    return FileResponse(EVAL_RESULTS, media_type="application/json")
