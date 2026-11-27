#!/usr/bin/env python3
"""Citation benchmarks (build step; run with nohup).

v3a  HELD-OUT: unseen 2026 Supreme Court criminal judgments (downloaded to $RAW_DATA_DIR/sc_aws_2026/,
     NEVER indexed). Query = the court-written "Issue for Consideration" (fallback: first headnote paragraph).
v3b  IN-CORPUS: our indexed 2023-2025 judgments that have an "Issue for Consideration" headnote.
     Query = that issue text; the citing judgment itself is excluded from the targets AND (at evaluation,
     --exclude-self) from every config's ranking. Seed 42, at most --cap-v3b queries.

For both: the query is scrubbed of case names, citations, case numbers, dates, non-Act years and the
citing case's own party/judge names; TARGETS = cited judgments that are in sc_criminal_cpu, matched by
SCR citation (year+volume+start page), neutral citation, or a conservative case-name match (both party
keys + year 2018-2025 + unique hit; match_type=name). SCC/AIR citations are counted only.

Outputs: $BENCHMARKS_DIR/v3a/citation_queries_v3a.json + build_report_v3a.json,
         $BENCHMARKS_DIR/v3b/citation_queries_v3b.json + build_report_v3b.json.

Usage (server):
    nohup python scripts/build_citation_benchmark.py --mode both > ~/Pratik/workspace/logs/bench_v3_build.log 2>&1 &
"""

import argparse
import datetime
import json
import random
import sqlite3
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import download_aws_sc as dl  # noqa: E402
import pandas as pd  # noqa: E402

from src import benchmark_v2 as bv  # noqa: E402
from src import citation_benchmark as cb  # noqa: E402
from src import judgment_text as jt  # noqa: E402
from src.config import get_settings  # noqa: E402

YEAR_V3A = 2026
SEED = 42


class Builder:
    """Shared query/target logic and counters for one benchmark."""

    def __init__(self, tag: str, index: Dict[str, Any], titles: Dict[str, str], corpus_ids: set, use_names: bool):
        self.tag, self.index, self.titles, self.corpus_ids, self.use_names = tag, index, titles, corpus_ids, use_names
        self.stats: Counter = Counter()
        self.cites: Counter = Counter()
        self.mapped: Counter = Counter()
        self.scrubbed: Counter = Counter()
        self.unmatched: List[str] = []
        self.queries: List[Dict[str, Any]] = []

    def add(self, headnote: str, fallback: str, body: str, parties: List[str], judges: List[str],
            meta: Dict[str, Any], exclude: Tuple[str, ...] = ()) -> None:
        source, issue = cb.issue_text(headnote, fallback)
        self.stats[f"query_source:{source}"] += 1
        if not issue:
            return
        query, removed = cb.scrub(issue, parties, judges)
        self.scrubbed.update(removed)
        if bv.leakage_reasons(query) or len(query.split()) < 8:
            self.stats["query_rejected_after_scrub"] += 1
            return
        cites = cb.extract_citations(body)
        self.cites.update({"scr": len(cites["scr"]), "neutral": len(cites["insc"]), "scc": cites["scc"],
                           "air": cites["air"], "case_names": len(cites["names"])})
        if cites["scr"] or cites["insc"] or cites["scc"] or cites["air"] or cites["names"]:
            self.stats["judgments_citing_anything"] += 1
        all_targets = cb.map_targets(cites, self.index, use_names=self.use_names)
        self_hits = [t for t in all_targets if t["doc_id"] in exclude]
        self.stats["self_citations_excluded"] += len(self_hits)
        targets = [t for t in all_targets if t["doc_id"] in self.corpus_ids and t["doc_id"] not in exclude]
        self.mapped.update(t["match_type"] for t in targets)
        if len(self.unmatched) < 5:
            self.unmatched.extend(cb.unmatched_names(cites, self.index)[: 5 - len(self.unmatched)])
        if not targets:
            self.stats["no_target_in_corpus"] += 1
            return
        self.queries.append({
            "query_id": f"{self.tag.upper()}-{len(self.queries) + 1:03d}", "type": f"citation_{self.tag}",
            "query": query, "query_source": source, "target_doc_ids": [t["doc_id"] for t in targets],
            "targets": targets, "target_title": self.titles.get(targets[0]["doc_id"]),
            "target_citation": targets[0]["citation"], "exclude_doc_ids": list(exclude), **meta,
        })

    def write(self, out_dir: Path, extra: Dict[str, Any], t0: float) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"citation_queries_{self.tag}.json").write_text(json.dumps(self.queries, indent=1, ensure_ascii=False),
                                                                    encoding="utf-8")
        n_t = [len(q["target_doc_ids"]) for q in self.queries]
        report = {"built_at": datetime.datetime.now().isoformat(timespec="seconds"), "tag": self.tag, **extra,
                  "stats": dict(self.stats), "citations_found": dict(self.cites), "targets_mapped_by_type": dict(self.mapped),
                  "scrubbed": dict(self.scrubbed), "queries_kept": len(self.queries),
                  "mean_targets_per_query": round(sum(n_t) / len(n_t), 2) if n_t else None,
                  "max_targets_per_query": max(n_t) if n_t else None, "unmatched_name_examples": self.unmatched,
                  "name_match": self.use_names, "elapsed_sec": round(time.perf_counter() - t0, 1)}
        (out_dir / f"build_report_{self.tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return report

    def print_report(self, report: Dict[str, Any], header: str) -> None:
        s = self.stats
        print("=" * 100)
        print(f"BENCHMARK {self.tag} BUILD ({header}) | {report['built_at']} | {report['elapsed_sec']}s")
        print(" | ".join(f"{k}: {v}" for k, v in report.items() if k in ("judgments", "downloads", "pdfs_read",
                                                                         "candidates", "sampled_order")))
        print("query source: " + " | ".join(f"{k.split(':')[1]}:{v}" for k, v in s.items() if k.startswith("query_source")))
        print(f"rejected after scrub {s['query_rejected_after_scrub']} | scrubbed: {dict(self.scrubbed)}")
        print(f"citations found: {dict(self.cites)} | judgments citing anything {s['judgments_citing_anything']}")
        print(f"mapped to corpus by type: {dict(self.mapped)} | self-citations excluded {s['self_citations_excluded']} | "
              f"no target in corpus {s['no_target_in_corpus']}")
        print(f"QUERIES KEPT {len(self.queries)} | mean targets/query {report['mean_targets_per_query']} | "
              f"max {report['max_targets_per_query']}")
        for q in self.queries[:3]:
            print(f"  {q['query_id']} [{q['query_source']}] Q: {q['query'][:150]}")
            print("      targets: " + "; ".join(f"{t['match_type']}:{t['citation']}" for t in q["targets"][:4])
                  + f" | e.g. {str(q['target_title'])[:60]}")
        print("unmatched name examples (corpus-year names that did not match):")
        for u in self.unmatched[:5]:
            print(f"  - {u[:140]}")


def build_v3a(settings, index, titles, corpus_ids, args) -> None:
    t0 = time.perf_counter()
    md_dir = settings.metadata_dir
    pq = md_dir / "sc" / f"year={YEAR_V3A}" / "metadata.parquet"
    if not pq.exists():
        pq.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f"{dl.S3_BASE}/metadata/parquet/year={YEAR_V3A}/metadata.parquet",
                                     headers={"User-Agent": dl.USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            pq.write_bytes(resp.read())
    dest_root = settings.raw_data_dir / f"sc_aws_{YEAR_V3A}"
    rows, _, dropped = dl.select_criminal(md_dir, [YEAR_V3A], dest_root)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda r: dl.download_one(r["s3_key"], Path(r["local_path"]), 4, 60.0), rows))
    for r, (status, nbytes, err) in zip(rows, results):
        r.update(status=status, bytes=nbytes, error=err)
    pd.DataFrame(rows).to_parquet(md_dir / f"manifest_{YEAR_V3A}.parquet", index=False)
    b = Builder("v3a", index, titles, corpus_ids, not args.no_name_match)
    pdfs = 0
    for r in rows:
        if r["status"] not in ("ok", "exists"):
            continue
        try:
            cleaned = jt.clean_pages(jt.extract_pages(r["local_path"]))
        except Exception:
            b.stats["pdf_unreadable"] += 1
            continue
        pdfs += 1
        text = "\n".join(cleaned["lines"])
        sec = jt.split_sections(cleaned["lines"])
        b.add("\n".join(sec["headnote"]), text, "\n".join(sec["body"]) or text,
              bv.party_tokens(r["title"], r["petitioner"], r["respondent"]), bv.judge_tokens(r["judges"]),
              {"year": YEAR_V3A, "source_cnr": r["cnr"], "source_title": r["title"], "source_citation": r["citation"]})
    report = b.write(settings.benchmarks_dir / "v3a", {"judgments": len(rows), "downloads": dict(Counter(r["status"] for r in rows)),
                                                       "pdfs_read": pdfs, "duplicates_dropped": len(dropped)}, t0)
    b.print_report(report, f"held-out, unseen {YEAR_V3A} judgments")


def build_v3b(settings, index, titles, corpus_ids, manifest_rows, args) -> None:
    t0 = time.perf_counter()
    db = settings.sqlite_path_for(args.collection)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    docs = [dict(r) for r in conn.execute(
        "SELECT doc_id, year, headnote, judges, case_title FROM documents WHERE year BETWEEN 2023 AND 2025 "
        "AND headnote LIKE '%ssue for %onsideration%' ORDER BY doc_id")]
    rng = random.Random(SEED)
    rng.shuffle(docs)
    by_id = {r["doc_id"]: r for r in manifest_rows}
    b = Builder("v3b", index, titles, corpus_ids, not args.no_name_match)
    for d in docs:
        if len(b.queries) >= args.cap_v3b:
            break
        body = "\n\n".join(r[0] for r in conn.execute(
            "SELECT text FROM parent_chunks WHERE doc_id = ? AND metadata_json LIKE '%\"section_kind\": \"body\"%' "
            "ORDER BY parent_id", (d["doc_id"],)))
        m = by_id.get(d["doc_id"], {})
        b.add(d["headnote"] or "", "", body,
              bv.party_tokens(m.get("title") or d["case_title"], m.get("petitioner"), m.get("respondent")),
              bv.judge_tokens(d["judges"]),
              {"year": d["year"], "source_doc_id": d["doc_id"], "source_title": d["case_title"],
               "source_citation": m.get("citation")}, exclude=(d["doc_id"],))
    conn.close()
    report = b.write(settings.benchmarks_dir / "v3b", {"candidates": len(docs), "seed": SEED, "cap": args.cap_v3b}, t0)
    b.print_report(report, "in-corpus 2023-2025, citing judgment excluded")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the citation benchmarks v3a (2026 held-out) and v3b (in-corpus)")
    ap.add_argument("--mode", choices=["v3a", "v3b", "both"], default="both")
    ap.add_argument("--collection", default="sc_criminal_cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap-v3b", type=int, default=150)
    ap.add_argument("--no-name-match", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    corpus = pd.read_parquet(settings.manifest_path)
    corpus = corpus[corpus["status"].isin(["ok", "exists"])]
    rows = corpus.to_dict("records")
    index = cb.build_corpus_index(rows)
    titles = dict(zip(corpus["doc_id"], corpus["title"]))
    corpus_ids = set(corpus["doc_id"])
    if args.mode in ("v3a", "both"):
        build_v3a(settings, index, titles, corpus_ids, args)
    if args.mode in ("v3b", "both"):
        build_v3b(settings, index, titles, corpus_ids, rows, args)


if __name__ == "__main__":
    main()
