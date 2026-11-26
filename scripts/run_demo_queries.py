#!/usr/bin/env python3
"""Run data/demo_queries.json through the full answer path (warms the LLM cache for the live demo).

Prints one line per query: top judgment, best relevance, low_confidence, citation precision,
refusal, expected behaviour. Writes the full results to $BENCHMARKS_DIR/v2/demo_results.json.

Usage (server; not while indexing):
    python scripts/run_demo_queries.py --collection sc_criminal_cpu
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the demo queries through the answer path")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--file", default=str(PROJECT_ROOT / "data" / "demo_queries.json"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    from src.rag_pipeline import IndianLegalRAGPipeline

    settings = get_settings()
    pipe = IndianLegalRAGPipeline(settings=settings, collection_name=args.collection)
    demos = json.loads(Path(args.file).read_text(encoding="utf-8"))
    out = []
    print(f"DEMO QUERIES | collection {pipe.collection_name} | MIN_RELEVANCE {settings.min_relevance}")
    for d in demos:
        res = pipe.generate_answer(query=d["query"], top_k=args.top_k, filters=d.get("filters") or None)
        cc = res["citation_check"]
        top = res["judgments"][0] if res["judgments"] else {}
        line = (f"{d['id']} {d['label'][:32]:<32} | top: {str(top.get('case_title', '-'))[:40]:<40} "
                f"{top.get('citation') or '-':<14} | rel {res['best_relevance'] if res['best_relevance'] is None else round(res['best_relevance'], 3)} "
                f"| low_conf {res['low_confidence']} | precision {cc['citation_precision']} | refused {cc['refused']} "
                f"| expect: {d.get('expect')}")
        if res["filters"].get("applied"):
            line += f" | filter {res['filters']['applied']} -> {res['filters']['matching_docs']} docs"
        print(line)
        out.append({"id": d["id"], "query": d["query"], "filters": d.get("filters"), "answer": res["answer"],
                    "low_confidence": res["low_confidence"], "best_relevance": res["best_relevance"],
                    "citation_check": cc, "judgments": res["judgments"], "expanded_query": res["expanded_query"],
                    "timings": res["timings"]})
    path = settings.benchmarks_dir / "v2" / "demo_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved: $BENCHMARKS_DIR/v2/demo_results.json ({len(out)} queries; answers are cached for the live demo)")


if __name__ == "__main__":
    main()
