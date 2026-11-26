#!/usr/bin/env python3
"""Inspect stored chunks and run questions through the retrieval funnel.

    python scripts/explain_query.py --collection sc_criminal_cpu_test --show-record
        One stored chunk exactly as Chroma holds it (id, embedded text, metadata, vector) and its parent.

    python scripts/explain_query.py --collection sc_criminal_cpu_test "question 1" "question 2"
        Compact top-5 per question (title | citation | bench | author | role | score) and per-stage
        latency in ms, including BM25 (to measure BM25 p50 on the full index).

    ... --year-from 2020 --min-bench 3      filters (SQLite documents -> doc_ids -> Chroma $in + BM25)
    ... --answer                            also a grounded LLM answer + citation check for the FIRST question
    ... --trace                             the old stage-by-stage trace (dense / BM25 / RRF tops)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.rag_pipeline import IndianLegalRAGPipeline, bench_label  # noqa: E402
from src.resource_guard import configure_torch_backends  # noqa: E402

LINE = "-" * 100


def _m(hit: Dict[str, Any], key: str, default: Any = "") -> Any:
    v = hit.get(key)
    return v if v not in (None, "") else (hit.get("metadata") or {}).get(key, default)


def show_record(pipeline: IndianLegalRAGPipeline, offset) -> None:
    collection = pipeline.vector_store.get_or_create_collection()
    print(LINE)
    print(f"Chroma collection '{collection.name}': {collection.count():,} chunks | SQLite parents: "
          f"{pipeline.sqlite_store.count_parents():,} | documents: {pipeline.sqlite_store.count_documents():,}")
    print(LINE)
    if collection.count() == 0:
        print("collection is empty")
        return
    if offset is None or offset >= collection.count():
        offset = collection.count() // 2
    rec = collection.get(limit=1, offset=offset, include=["documents", "metadatas", "embeddings"])
    chunk_id, document, meta = rec["ids"][0], rec["documents"][0], dict(rec["metadatas"][0])
    vector = rec["embeddings"][0]
    raw_text = meta.pop("raw_text", "")
    print(f"CHUNK ID : {chunk_id}   (= doc_id - p<parent> - c<child>, deterministic)")
    print(f"EMBEDDED TEXT: {document[:400]}")
    print("METADATA:")
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"raw_text: {raw_text[:160]}...")
    print(f"EMBEDDING: {len(vector)} dims, first 6 = {[round(float(x), 4) for x in vector[:6]]}")
    parent = pipeline.retrieve_parent_context(meta["parent_id"])
    if parent:
        print(f"PARENT {meta['parent_id']}: {len(parent['text'])} chars | role={parent['metadata'].get('role')} | "
              f"section_kind={parent['metadata'].get('section_kind')}")
    print(LINE)


def compact(pipeline: IndianLegalRAGPipeline, query: str, top_k: int, filters) -> Dict[str, float]:
    out = pipeline.retrieve_hybrid(query=query, top_k=top_k, candidate_pool_size=60, filters=filters)
    print(LINE)
    print(f"Q: {query}")
    if out["expanded_query"] != query:
        print(f"   expanded: {out['expanded_query'][len(query):].strip()}")
    if filters:
        print(f"   filters: {out['filters']['applied']} -> matching docs {out['filters']['matching_docs']}")
    for g in out["judgments"]:  # one line per JUDGMENT (best passage score; up to 2 passages kept)
        c = g["best"]
        roles = ", ".join(dict.fromkeys(str(_m(p, "role")) for p in g["passages"]))
        print(f"  {g['rank']}. {str(_m(c, 'case_title'))[:50]:<50} | {_m(c, 'citation', '-'):<14} | "
              f"{bench_label(_m(c, 'bench_strength', None)):<25} | {str(_m(c, 'author_judge', '-'))[:22]:<22} | "
              f"{roles[:28]:<28} | {len(g['passages'])}p | score {g['score']:.4f} "
              f"(rel {g['relevance']:.4f} x auth {g['authority_factor']:.4f})")
    t = out["timings"]
    print("   ms: " + ", ".join(f"{k.replace('_sec', '')}={v * 1000:.0f}" for k, v in t.items()))
    return t


def trace(pipeline: IndianLegalRAGPipeline, query: str) -> None:
    vs = pipeline.vector_store
    from src.concordance import StatutoryConcordance

    expanded = StatutoryConcordance.expand_query(query)
    dense = vs.query_dense(expanded, top_k=60)
    bm25 = vs.query_bm25(expanded, top_k=60)
    fused = pipeline.compute_rrf(dense, bm25, k=60)
    for label, hits, key in (("dense", dense, "similarity_score"), ("bm25", bm25, "bm25_score"), ("rrf", fused, "rrf_score")):
        print(f"  {label}: " + " | ".join(f"{str(_m(h, 'case_title'))[:30]} {float(h.get(key, 0)):.3f}" for h in hits[:5]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored chunks and run questions through the funnel")
    parser.add_argument("queries", nargs="*", help="Legal questions")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--show-record", action="store_true")
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--min-bench", type=int, default=None)
    parser.add_argument("--answer", action="store_true", help="Grounded LLM answer for the first question")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not args.queries and not args.show_record:
        parser.error("give one or more questions and/or --show-record")

    configure_torch_backends()
    pipeline = IndianLegalRAGPipeline(settings=get_settings(), collection_name=args.collection)
    filters = {k: v for k, v in {"year_from": args.year_from, "year_to": args.year_to,
                                 "min_bench": args.min_bench}.items() if v is not None} or None
    if args.show_record:
        show_record(pipeline, args.offset)
    if args.queries:
        pipeline.retrieve_hybrid(query="warm-up: bail under Section 439 CrPC", top_k=1)  # load models first
        timings: List[Dict[str, float]] = []
        for q in args.queries:
            timings.append(compact(pipeline, q, args.top_k, filters))
            if args.trace:
                trace(pipeline, q)
        if len(timings) > 1:
            for key in ("bm25_retrieval_sec", "dense_retrieval_sec", "reranking_sec", "funnel_latency_sec"):
                vals = [t[key] * 1000 for t in timings if key in t]
                if vals:
                    print(f"   {key.replace('_sec', '')}: p50 {statistics.median(vals):.0f} ms, max {max(vals):.0f} ms "
                          f"(n={len(vals)})")
    if args.answer and args.queries:
        res = pipeline.generate_answer(query=args.queries[0], top_k=args.top_k, filters=filters)
        print(LINE)
        print("PRECEDENT HEADERS SENT TO THE LLM:")
        for p in res["precedents"]:
            print("  " + p["header"][:200])
        print(LINE)
        print("ANSWER:")
        print(res["answer"])
        print(LINE)
        cc = res["citation_check"]
        print(f"LOW CONFIDENCE: {res['low_confidence']} (best relevance {res['best_relevance']}, "
              f"MIN_RELEVANCE {res['min_relevance']})")
        print(f"CITATION CHECK: precision={cc['citation_precision']} | cited={cc['cited_precedents']} of "
              f"{cc['n_precedents']} | unknown numbers={cc['unknown_precedent_numbers']} | "
              f"unknown external={cc['unknown_external_citations']} | refused={cc['refused']} | "
              f"ends with verify={cc['ends_with_verify']} | llm {res['timings'].get('llm_generation_sec', 0):.1f}s")
    print(LINE)


if __name__ == "__main__":
    main()
