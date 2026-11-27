#!/usr/bin/env python3
"""FastAPI backend + static web UI (web/) for the SC criminal precedents RAG demo.

    GET  /api/status     corpus size, models, LLM route
    GET  /api/examples   demo queries (data/demo_queries.json)
    POST /api/search     judgment-level retrieval (+ optional grounded answer and citation check)
    GET  /api/results    benchmark v2 results.json (for the Evaluation page)

Usage (server; never at the same time as indexing):
    nohup python server.py --port 8000 > ~/Pratik/workspace/logs/server.log 2>&1 &
"""

import argparse
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.citation_check import check_citations
from src.config import get_settings
from src.llm_budget import get_budget
from src.rag_pipeline import IndianLegalRAGPipeline, _field, bench_label
from src.resource_guard import ResourceGuard

# Presentation-layer citation normalisation. The answer model sometimes writes 【Precedent 1†Passage 2】
# (full-width brackets, dagger, passage number) instead of [Precedent 1]. Display form: "[Precedent N, passage M]"
# or "[Precedent N]". The existing citation check (src/citation_check.py, unchanged) only knows "[Precedent N]",
# so it is run on the same text with the ", passage M" part removed.
CITATION_FORM_RE = re.compile(
    r"[【\[]\s*(Precedents?)\s+(\d+(?:\s*(?:,|and|&|-)\s*\d+)*)\s*(?:†\s*Passage\s*(\d+))?\s*[】\]]",
    re.IGNORECASE)
PASSAGE_SUFFIX_RE = re.compile(r"(\[Precedents? [^\]]*?), passage \d+\]", re.IGNORECASE)


def normalize_answer_citations(text: str) -> str:
    """【Precedent N†Passage M】, 【Precedent N】, [Precedent N†Passage M] -> "[Precedent N, passage M]" / "[Precedent N]"."""
    def repl(m: "re.Match") -> str:
        word = "Precedents" if m.group(1).lower() == "precedents" else "Precedent"
        return f"[{word} {m.group(2)}, passage {m.group(3)}]" if m.group(3) else f"[{word} {m.group(2)}]"

    return CITATION_FORM_RE.sub(repl, text or "")


def citations_for_check(text: str) -> str:
    """Normalised answer -> the "[Precedent N]" form the existing citation check understands."""
    return PASSAGE_SUFFIX_RE.sub(r"\1]", text or "")


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"
DEMO_QUERIES = PROJECT_ROOT / "data" / "demo_queries.json"

settings = get_settings()
ResourceGuard(
    max_ram_gb=settings.max_ram_gb,
    warn_ram_gb=settings.warn_ram_gb,
    max_storage_gb=settings.max_storage_gb,
    vram_safety_fraction=settings.vram_safety_fraction,
    num_cpu_threads=settings.num_threads,
).initialize_hardware()

pipeline = IndianLegalRAGPipeline(settings=settings)
# One model instance: serialise requests so concurrent clicks cannot exceed the RAM budget
_pipeline_lock = threading.Lock()

app = FastAPI(title="SC Criminal Precedents RAG (academic prototype)")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=10)
    generate_answer: bool = False
    year_from: Optional[int] = Field(None, ge=1950, le=2100)
    year_to: Optional[int] = Field(None, ge=1950, le=2100)
    min_bench: Optional[int] = Field(None, ge=1, le=7)

    def filters(self) -> Optional[Dict[str, Any]]:
        f = {"year_from": self.year_from, "year_to": self.year_to, "min_bench": self.min_bench}
        f = {k: v for k, v in f.items() if v is not None}
        return f or None


def judgment_card(g: Dict[str, Any]) -> Dict[str, Any]:
    """One card per judgment: metadata chips + up to 2 passages with role labels."""
    import re

    meta = pipeline.judgment_meta(g)
    # Official PDF in the public AWS Open Data bucket, same key format as scripts/download_aws_sc.py
    # (local_pdf_path: <root>/year=YYYY/<path>_EN.pdf; s3_key: data/pdf/year=YYYY/english/<path>_EN.pdf).
    # The full relative <path> after year=YYYY/ is kept, so a nested path still maps correctly ("" if not parseable).
    m = re.search(r"year=(\d{4})/(.+_EN\.pdf)$", str(meta.get("source_path") or "").replace("\\", "/"))
    pdf_url = (f"https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year={m.group(1)}/english/{m.group(2)}"
               if m else "")
    return {
        "rank": g["rank"],
        "doc_id": g["doc_id"],
        "case_title": meta.get("case_title") or "Untitled",
        "citation": meta.get("citation") or "",
        "decision_date": meta.get("decision_date") or "",
        "year": meta.get("year") or "",
        "bench_label": bench_label(meta.get("bench_strength")),
        "bench_strength": meta.get("bench_strength"),
        "judges": meta.get("judges") or "",
        "author_judge": meta.get("author_judge") or "",
        "author_source": meta.get("author_source") or "",
        "disposal_nature": meta.get("disposal_nature") or "",
        "statutes_cited": meta.get("statutes_cited") or "",
        "pdf_url": pdf_url,
        "score": round(float(g["score"]), 4),
        "relevance": round(float(g["relevance"]), 4),
        "authority_factor": round(float(g["authority_factor"]), 4),
        "passages": [{"role": _field(p, "role", "Unclassified"), "section_kind": _field(p, "section_kind", ""),
                      "text": str(_field(p, "raw_text") or p.get("text", ""))[:900]} for p in g["passages"]],
    }


@app.get("/api/status")
def status() -> Dict[str, Any]:
    budget = get_budget(settings)
    route = budget.llm_label(pipeline.gemini_client)
    return {
        "collection": pipeline.collection_name,
        "chunks_indexed": pipeline.vector_store.get_or_create_collection().count(),
        "parents_indexed": pipeline.sqlite_store.count_parents(),
        "judgments_indexed": pipeline.sqlite_store.count_documents(),
        "embedding_model": settings.local_embedding_model,
        "reranker_model": settings.reranker_model,
        "min_relevance": settings.min_relevance,
        "llm_route": route,
        "llm_available": route != "none configured",
    }


@app.get("/api/examples")
def examples() -> List[Dict[str, Any]]:
    try:
        return json.loads(DEMO_QUERIES.read_text(encoding="utf-8"))
    except Exception:
        return []


@app.post("/api/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    query = req.query.strip()
    with _pipeline_lock:
        if req.generate_answer:
            res = pipeline.generate_answer(query=query, top_k=req.top_k, candidate_pool_size=60, filters=req.filters())
            groups, timings = res["judgment_groups"], res["timings"]
            expanded, finfo = res["expanded_query"], res["filters"]
            low_confidence, best_relevance = res["low_confidence"], res["best_relevance"]
            # normalise citation forms for display, then re-run the existing check on the normalised text
            answer = normalize_answer_citations(res["answer"])
            citation_check = check_citations(citations_for_check(answer), res["precedents"])
        else:
            funnel = pipeline.retrieve_hybrid(query=query, top_k=req.top_k, candidate_pool_size=60, filters=req.filters())
            groups, timings = funnel["judgments"], funnel["timings"]
            expanded, finfo = funnel["expanded_query"], funnel["filters"]
            answer, citation_check = None, None
            best_relevance = max((g["relevance"] for g in groups), default=None)
            low_confidence = bool(groups) and best_relevance < settings.min_relevance
    return {
        "query": query,
        "expanded_query": expanded,
        "filters": finfo,
        "results": [judgment_card(g) for g in groups],
        "low_confidence": low_confidence,
        "best_relevance": best_relevance,
        "min_relevance": settings.min_relevance,
        "answer": answer,
        "citation_check": citation_check,
        "timings": timings,
    }


@app.get("/api/results")
def results() -> JSONResponse:
    for path in (settings.benchmarks_dir / "v2" / "results.json", PROJECT_ROOT / "results" / "v2" / "results.json"):
        if path.exists():
            return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    raise HTTPException(status_code=404, detail="No results.json yet: run scripts/evaluate_v2.py.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SC criminal precedents RAG web demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("[INFO] Warming up embedding model and cross-encoder...")
    try:
        pipeline.retrieve_hybrid(query="warm-up: bail under Section 439 CrPC", top_k=1)
    except Exception as err:
        print(f"[WARN] Warm-up failed: {err}")
    uvicorn.run(app, host=args.host, port=args.port)
