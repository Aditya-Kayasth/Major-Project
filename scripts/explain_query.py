#!/usr/bin/env python3
"""Show what the system stores and how a query moves through the retrieval funnel.

    python scripts/explain_query.py --show-record
        Prints one stored chunk exactly as ChromaDB holds it (id, embedded text, metadata,
        embedding vector) and the parent section it points to in SQLite.

    python scripts/explain_query.py "Can a bail condition require sharing live GPS location?"
        Traces the query stage by stage: concordance expansion, dense top hits, BM25 top hits,
        Reciprocal Rank Fusion, cross-encoder reranking and the parent sections handed to the LLM.
        No LLM call is made.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.concordance import StatutoryConcordance  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.rag_pipeline import IndianLegalRAGPipeline  # noqa: E402
from src.resource_guard import configure_torch_backends  # noqa: E402

LINE = "-" * 100


def _title(hit: Dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    return str(hit.get("case_title") or meta.get("case_title", "?"))[:60]


def show_record(pipeline: IndianLegalRAGPipeline, offset: int) -> None:
    collection = pipeline.vector_store.get_or_create_collection()
    print(LINE)
    print(f"ChromaDB collection '{collection.name}': {collection.count():,} chunks | "
          f"SQLite parent sections: {pipeline.sqlite_store.count_parents():,}")
    print(LINE)

    rec = collection.get(limit=1, offset=offset, include=["documents", "metadatas", "embeddings"])
    chunk_id, document, meta = rec["ids"][0], rec["documents"][0], dict(rec["metadatas"][0])
    vector = rec["embeddings"][0]
    raw_text = meta.pop("raw_text", "")

    print(f"CHUNK ID  : {chunk_id}")
    print("            = sha1(file name)[:16] - p<parent index> - c<child index>  (deterministic)\n")
    print("EMBEDDED TEXT (what the embedding model and BM25 actually see):")
    print(f"  {document}\n")
    print("METADATA (stored next to the vector; used for citations, filters and the authority prior):")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n  raw_text (snippet shown in the UI): {raw_text[:160]}...")
    print(f"\nEMBEDDING: {len(vector)} numbers, first 8 = {[round(float(x), 4) for x in vector[:8]]}")

    parent = pipeline.retrieve_parent_context(meta["parent_id"])
    print(LINE)
    print(f"PARENT SECTION in SQLite (parent_id = {meta['parent_id']}), sent to the LLM as context:")
    if parent:
        print(json.dumps({k: v for k, v in parent["metadata"].items() if k != "source_path"}, indent=2, ensure_ascii=False))
        print(f"\n  text ({len(parent['text'])} chars): {parent['text'][:500]}...")
    print(LINE)


def _print_hits(label: str, hits: List[Dict[str, Any]], score_key: str, n: int = 5) -> None:
    print(f"\n{label}")
    for rank, hit in enumerate(hits[:n], start=1):
        print(f"  {rank:>2}. {_title(hit):<62} {score_key}={float(hit.get(score_key, 0.0)):.4f}")


def trace_query(pipeline: IndianLegalRAGPipeline, query: str, top_k: int) -> None:
    vs = pipeline.vector_store
    expanded = StatutoryConcordance.expand_query(query)
    print(LINE)
    print(f"QUERY    : {query}")
    print(f"EXPANDED : {expanded}" if expanded != query else "EXPANDED : (no statute to map)")
    print(LINE)

    dense = vs.query_dense(expanded, top_k=60)
    bm25 = vs.query_bm25(expanded, top_k=60)
    fused = pipeline.compute_rrf(dense, bm25, k=60, query=query)
    _print_hits("STAGE 2a  Dense (meaning) search - top 5 of 60 by cosine similarity:", dense, "similarity_score")
    _print_hits("STAGE 2b  BM25 (keyword) search - top 5 of 60 by BM25 score:", bm25, "bm25_score")
    _print_hits(f"STAGE 3   Reciprocal Rank Fusion - {len(fused)} unique chunks, top 30 go to the reranker:", fused, "rrf_score")

    out = pipeline.retrieve_hybrid(query=query, top_k=top_k, candidate_pool_size=60)
    _print_hits(f"STAGE 4   Cross-encoder rerank (+ bounded authority prior) - final top {top_k}:",
                out["reranked_candidates"], "rerank_score", n=top_k)

    print(f"\nSTAGE 5   Parent sections handed to the LLM ({len(out['retrieved_parents'])} unique):")
    for parent in out["retrieved_parents"]:
        meta = parent.get("metadata", {})
        print(f"  - {str(parent.get('case_title'))[:60]:<62} {meta.get('court', '')}, {meta.get('year', '')} "
              f"({len(parent.get('text', ''))} chars)")
    print("\nTIMINGS (ms): " + ", ".join(f"{k.replace('_sec', '')}={v * 1000:.0f}" for k, v in out["timings"].items()))
    print(LINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored chunks and trace the retrieval funnel")
    parser.add_argument("query", nargs="?", help="Legal question to trace")
    parser.add_argument("--show-record", action="store_true", help="Print one stored chunk and its parent")
    parser.add_argument("--offset", type=int, default=1234, help="Which stored chunk to show")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not args.query and not args.show_record:
        parser.error("give a query and/or --show-record")

    configure_torch_backends()
    pipeline = IndianLegalRAGPipeline(settings=get_settings())
    if args.show_record:
        show_record(pipeline, args.offset)
    if args.query:
        trace_query(pipeline, args.query, args.top_k)


if __name__ == "__main__":
    main()
