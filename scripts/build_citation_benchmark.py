#!/usr/bin/env python3
"""Benchmark v3: held-out CITATION benchmark on unseen 2026 judgments (build step; run with nohup).

1. Downloads metadata/parquet/year=2026/metadata.parquet (if missing) to $METADATA_DIR/sc/year=2026/,
   selects criminal judgments with the Phase 1 parser (case number contains CRIMINAL/CRL) and downloads
   their English PDFs to $RAW_DATA_DIR/sc_aws_2026/ (resumable, %PDF check). Writes
   $METADATA_DIR/manifest_2026.parquet. These judgments are NEVER indexed.
2. For each 2026 judgment: QUERY = its "Issue for Consideration" (fallback: first headnote paragraph),
   scrubbed of case names, citations, case numbers, dates, non-Act years and its own party/judge names;
   skipped if the leakage filter still fires or fewer than 8 words remain.
3. TARGETS = judgments it cites that are in our corpus (manifest status ok/exists): SCR citation
   (year+volume+start page), neutral citation, and (unless --no-name-match) a conservative case-name
   match flagged match_type=name. SCC/AIR citations are counted only.
4. Keeps queries with >= 1 target -> $BENCHMARKS_DIR/v3/citation_queries_v3.json + build_report_v3.json.

Usage (server):
    nohup python scripts/build_citation_benchmark.py > ~/Pratik/workspace/logs/bench_v3_build.log 2>&1 &
"""

import argparse
import datetime
import json
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import download_aws_sc as dl  # noqa: E402
import pandas as pd  # noqa: E402

from src import benchmark_v2 as bv  # noqa: E402
from src import citation_benchmark as cb  # noqa: E402
from src import judgment_text as jt  # noqa: E402
from src.config import get_settings  # noqa: E402

YEAR = 2026


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the v3 citation benchmark from unseen 2026 judgments")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-name-match", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None, help="Tests only")
    args = ap.parse_args()

    settings = get_settings()
    t0 = time.perf_counter()
    md_dir = settings.metadata_dir
    pq = md_dir / "sc" / f"year={YEAR}" / "metadata.parquet"
    if not pq.exists():
        pq.parent.mkdir(parents=True, exist_ok=True)
        url = f"{dl.S3_BASE}/metadata/parquet/year={YEAR}/metadata.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": dl.USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            pq.write_bytes(resp.read())
        print(f"[v3] downloaded {YEAR} metadata ({pq.stat().st_size / 1024:.0f} KB)", flush=True)

    dest_root = settings.raw_data_dir / f"sc_aws_{YEAR}"
    rows, per_year, dropped = dl.select_criminal(md_dir, [YEAR], dest_root)
    if args.max_docs:
        rows = rows[: args.max_docs]
    print(f"[v3] {YEAR} criminal judgments selected: {len(rows)} (duplicates dropped {len(dropped)})", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda r: dl.download_one(r["s3_key"], Path(r["local_path"]), 4, 60.0), rows))
    for r, (status, nbytes, err) in zip(rows, results):
        r.update(status=status, bytes=nbytes, error=err)
    pd.DataFrame(rows).to_parquet(md_dir / f"manifest_{YEAR}.parquet", index=False)
    dl_status = Counter(r["status"] for r in rows)
    print(f"[v3] downloads: {dict(dl_status)} | {time.perf_counter() - t0:.0f}s", flush=True)

    corpus = pd.read_parquet(settings.manifest_path)
    corpus = corpus[corpus["status"].isin(["ok", "exists"])]
    index = cb.build_corpus_index(corpus.to_dict("records"))
    titles = dict(zip(corpus["doc_id"], corpus["title"]))
    corpus_ids = set(corpus["doc_id"])

    stats: Counter = Counter()
    cite_counts: Counter = Counter()
    mapped_counts: Counter = Counter()
    scrubbed: Counter = Counter()
    queries = []
    for r in rows:
        if r["status"] not in ("ok", "exists"):
            continue
        stats["pdf_ok"] += 1
        try:
            pages = jt.extract_pages(r["local_path"])
        except Exception:
            stats["pdf_unreadable"] += 1
            continue
        cleaned = jt.clean_pages(pages)
        text = "\n".join(cleaned["lines"])
        sec = jt.split_sections(cleaned["lines"])
        source, issue = cb.issue_text("\n".join(sec["headnote"]), text)
        stats[f"query_source:{source}"] += 1
        if not issue:
            continue
        parties = bv.party_tokens(r["title"], r["petitioner"], r["respondent"])
        query, removed = cb.scrub(issue, parties, bv.judge_tokens(r["judges"]))
        scrubbed.update(removed)
        if bv.leakage_reasons(query) or len(query.split()) < 8:
            stats["query_rejected_after_scrub"] += 1
            continue
        body = "\n".join(sec["body"]) or text
        cites = cb.extract_citations(body)
        cite_counts.update({"scr": len(cites["scr"]), "neutral": len(cites["insc"]), "scc": cites["scc"],
                            "air": cites["air"], "case_names": len(cites["names"])})
        if cites["scr"] or cites["insc"] or cites["scc"] or cites["air"]:
            stats["judgments_citing_anything"] += 1
        targets = [t for t in cb.map_targets(cites, index, use_names=not args.no_name_match) if t["doc_id"] in corpus_ids]
        mapped_counts.update(t["match_type"] for t in targets)
        if not targets:
            stats["no_target_in_corpus"] += 1
            continue
        queries.append({
            "query_id": f"CQ-{len(queries) + 1:03d}", "type": "citation", "query": query, "query_source": source,
            "target_doc_ids": [t["doc_id"] for t in targets], "targets": targets,
            "target_title": titles.get(targets[0]["doc_id"]), "target_citation": targets[0]["citation"],
            "year": YEAR, "source_cnr": r["cnr"], "source_title": r["title"], "source_citation": r["citation"],
        })

    out_dir = settings.benchmarks_dir / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "citation_queries_v3.json").write_text(json.dumps(queries, indent=1, ensure_ascii=False), encoding="utf-8")
    n_t = [len(q["target_doc_ids"]) for q in queries]
    report = {
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"), "year": YEAR,
        "criminal_judgments": len(rows), "downloads": dict(dl_status), "stats": dict(stats),
        "citations_found": dict(cite_counts), "targets_mapped_by_type": dict(mapped_counts),
        "scrubbed": dict(scrubbed), "queries_kept": len(queries),
        "mean_targets_per_query": round(sum(n_t) / len(n_t), 2) if n_t else None,
        "max_targets_per_query": max(n_t) if n_t else None,
        "name_match": not args.no_name_match, "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "build_report_v3.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("=" * 100)
    print(f"BENCHMARK v3 BUILD (citation, unseen {YEAR}) | {report['built_at']} | {report['elapsed_sec']}s")
    print(f"{YEAR} criminal judgments {len(rows)} | downloads {dict(dl_status)} | PDFs read {stats['pdf_ok']}")
    print("query source: " + " | ".join(f"{k.split(':')[1]}:{v}" for k, v in stats.items() if k.startswith("query_source")))
    print(f"rejected after scrub {stats['query_rejected_after_scrub']} | scrubbed: {dict(scrubbed)}")
    print(f"citations found (in kept-issue judgments): {dict(cite_counts)} | judgments citing anything "
          f"{stats['judgments_citing_anything']}")
    print(f"mapped to corpus by type: {dict(mapped_counts)} | no target in corpus {stats['no_target_in_corpus']}")
    print(f"QUERIES KEPT {len(queries)} | mean targets/query {report['mean_targets_per_query']} | max {report['max_targets_per_query']}")
    for q in queries[:3]:
        print(f"  {q['query_id']} [{q['query_source']}] Q: {q['query'][:150]}")
        print(f"      targets: " + "; ".join(f"{t['match_type']}:{t['citation']}" for t in q["targets"][:4])
              + f" | e.g. {str(q['target_title'])[:60]}")
    print("files: $BENCHMARKS_DIR/v3/citation_queries_v3.json, build_report_v3.json; $METADATA_DIR/manifest_2026.parquet")


if __name__ == "__main__":
    main()
