#!/usr/bin/env python3
"""FastAPI backend + static web UI for the Indian Legal Precedents RAG demo.

Serves the HTML/CSS/JS front-end in web/ and a small JSON API:
    GET  /api/status     corpus size, models, Gemini budget left today
    GET  /api/examples   sample questions from the benchmark
    POST /api/search     run the retrieval funnel (+ optional grounded answer)
    GET  /api/benchmark  latest saved benchmark report

Usage (on the server):
    python server.py --port 8000
Then open http://localhost:8000 (via SSH tunnel: ssh -L 8000:localhost:8000 <user>@<server>).
"""

import argparse
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_settings
from src.llm_budget import get_budget
from src.rag_pipeline import IndianLegalRAGPipeline
from src.resource_guard import ResourceGuard

WEB_DIR = Path(__file__).resolve().parent / "web"

settings = get_settings()
ResourceGuard(
    max_ram_gb=settings.max_ram_gb,
    warn_ram_gb=settings.warn_ram_gb,
    max_storage_gb=settings.max_storage_gb,
    vram_safety_fraction=settings.vram_safety_fraction,
    num_cpu_threads=settings.num_threads,
).initialize_hardware()

pipeline = IndianLegalRAGPipeline(settings=settings)
# One GPU, one model instance: serialise requests so concurrent clicks cannot OOM the GPU
_pipeline_lock = threading.Lock()

app = FastAPI(title="Indian Legal Precedents RAG")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=10)
    generate_answer: bool = False
    one_per_case: bool = True


def _clean_title(title: Any) -> str:
    """Drop Kaggle's duplicate-file suffix ('..._on_8_January_2025_1.pdf' -> trailing ' 1')."""
    return re.sub(r"\s+\d{1,2}$", "", str(title or "")).strip()


def _display_role(role: Any) -> str:
    """The heuristic role tagger never positively detects obiter: 'Obiter Dicta' is its fallback
    when no cue matched, so it is shown honestly as unclassified."""
    role = str(role or "")
    return "Unclassified role" if role == "Obiter Dicta" else role


def _doc_key(candidate: Dict[str, Any]) -> str:
    meta = candidate.get("metadata") or {}
    source = str(candidate.get("source_path") or meta.get("source_path") or "")
    return Path(source.replace("\\", "/")).stem or str(candidate.get("case_title") or meta.get("case_title"))


def _format_candidate(rank: int, c: Dict[str, Any]) -> Dict[str, Any]:
    meta = c.get("metadata") or {}
    authority = (
        float(c.get("court_tier_weight", 1.0)) * float(c.get("role_weight", 1.0)) * float(c.get("judge_affinity", 1.0))
    )
    return {
        "rank": rank,
        "case_title": _clean_title(c.get("case_title") or meta.get("case_title", "Unknown case")),
        "court": c.get("court") or meta.get("court", ""),
        "year": c.get("year") or meta.get("year", ""),
        "role": _display_role(c.get("role") or meta.get("role", "")),
        "rerank_score": round(float(c.get("rerank_score", 0.0)), 4),
        "cross_encoder_raw": round(float(c.get("raw_cross_encoder_score", 0.0)), 3),
        "authority_multiplier": round(authority, 3),
        "authority_factor": round(pipeline._authority_factor(authority), 4),
        "dense_rank": c.get("dense_rank"),
        "bm25_rank": c.get("bm25_rank"),
        "document": _doc_key(c),
        "snippet": str(meta.get("raw_text") or c.get("text", ""))[:600],
    }


@app.get("/api/status")
def status() -> Dict[str, Any]:
    collection = pipeline.vector_store.get_or_create_collection()
    budget = get_budget(settings)
    return {
        "collection": settings.collection_name,
        "chunks_indexed": collection.count(),
        "parents_indexed": pipeline.sqlite_store.count_parents(),
        "embedding_model": settings.local_embedding_model
        if settings.embedding_provider == "local"
        else settings.gemini_embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_route": budget.llm_label(pipeline.gemini_client),
        "llm_available": budget.llm_label(pipeline.gemini_client) != "none configured",
        "gemini_requests_left_today": budget.remaining_today(),
    }


@app.get("/api/examples")
def examples(limit: int = 8) -> List[str]:
    path = settings.benchmarks_dir / "golden_queries.json"
    try:
        queries = json.loads(path.read_text(encoding="utf-8"))
        return [q.get("query_text") or q.get("query") for q in queries[:limit]]
    except Exception:
        return []


@app.post("/api/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    query = req.query.strip()
    with _pipeline_lock:
        # All 30 fused candidates are cross-encoder scored; asking for all of them lets us
        # show the best chunk from each distinct judgment rather than 5 chunks of one case.
        retrieve_k = 30 if req.one_per_case else req.top_k
        funnel = pipeline.retrieve_hybrid(query=query, top_k=retrieve_k, candidate_pool_size=60)

        candidates = funnel["reranked_candidates"]
        if req.one_per_case:
            seen, distinct = set(), []
            for c in candidates:
                key = _doc_key(c)
                if key not in seen:
                    seen.add(key)
                    distinct.append(c)
            candidates = distinct
        candidates = candidates[: req.top_k]

        answer = None
        llm_timings: Dict[str, Any] = {}
        if req.generate_answer:
            result = pipeline.generate_answer(query=query, top_k=req.top_k, candidate_pool_size=60)
            answer = result.get("answer")
            llm_timings = {"llm_generation_sec": result.get("timings", {}).get("llm_generation_sec")}

    return {
        "query": query,
        "expanded_query": funnel.get("expanded_query", query),
        "results": [_format_candidate(i, c) for i, c in enumerate(candidates, start=1)],
        "timings": {**funnel["timings"], **{k: v for k, v in llm_timings.items() if v is not None}},
        "funnel_counts": {**funnel.get("funnel_counts", {}), "returned": len(candidates)},
        "answer": answer,
        "gemini_requests_left_today": get_budget(settings).remaining_today(),
    }


@app.get("/api/benchmark")
def benchmark() -> Dict[str, Any]:
    files = sorted(settings.benchmarks_dir.glob("benchmark_results_*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="No benchmark results yet. Run evaluate_pipeline.py.")
    report = json.loads(files[-1].read_text(encoding="utf-8"))
    details = report.get("raw_results", {}).get("Config 4 (Full Legal Funnel)", {}).get("query_details", [])
    return {
        "file": files[-1].name,
        "timestamp": report.get("timestamp"),
        "queries_count": report.get("queries_count"),
        "collection_name": report.get("collection_name"),
        "summary_table": report.get("summary_table", {}),
        "subset_metrics": report.get("subset_metrics", {}),
        "per_query": [
            {
                **{k: q.get(k) for k in ("query_id", "query", "rank", "latency_ms", "score")},
                "target_case": _clean_title(q.get("target_case")),
                "top_1_case": _clean_title(q.get("top_1_case")),
            }
            for q in details
        ],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Indian Legal RAG web demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("[INFO] Warming up embedding model and cross-encoder...")
    try:
        pipeline.retrieve_hybrid(query="warm-up query on Article 21", top_k=1)
    except Exception as err:
        print(f"[WARN] Warm-up failed: {err}")
    uvicorn.run(app, host=args.host, port=args.port)
