#!/usr/bin/env python3
"""Benchmark v2 query generation (Phase 6, step 1; run with nohup).

From the indexed collection's SQLite store (documents + parent sections) it builds:
- 100 legal-issue queries from BODY sections (Ratio/Analysis preferred, then Unclassified / Facts
  ranked by holding cues), stratified by year (proportional allocation), seed 42
- 50 fact-pattern queries from BODY Facts sections of OTHER judgments, briefed the way a lawyer
  briefs facts (no names, places, dates, case numbers, citations)
- 5 hand-written out-of-scope queries (expected: refusal)
- $BENCHMARKS_DIR/v2/human_queries.json template with 20 empty slots (never overwritten)

Leakage filter (src/benchmark_v2.leakage_reasons): party names, judge names, case numbers, citations,
dates and non-Act years. Rejections are counted and reported. Judgments whose headnote split is
'none' / 'unsplit' are excluded (headnote and body could not be separated). Headnote parents are
NEVER used (leakage rule, docs/phase0_audit.md).

LLM calls go through src/llm_budget.py (Groq first, rate limited by LLM_RPM, every response cached),
so a re-run with the same seed costs nothing.

Usage (server):
    nohup python scripts/generate_benchmark_v2.py --collection sc_criminal_cpu > ~/Pratik/workspace/logs/bench_gen.log 2>&1 &
"""

import argparse
import datetime
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import benchmark_v2 as bv  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm_budget import gen_config, get_budget, make_gemini_client  # noqa: E402

LEGAL_PROMPT = """You are building a retrieval benchmark for an Indian criminal-law research engine over Supreme Court judgments.

Below is a passage from a judgment. Write ONE research question that an advocate would type into a legal search engine and that THIS passage answers.

Strict rules:
- Do NOT mention party names, the case title, judge names, place names, citations, case numbers or dates.
- Ask about the legal issue in general terms (statute section, doctrine, evidence rule, procedure, sentencing), 15-40 words.
- Statute names and section numbers are allowed (e.g. "Section 304B IPC", "Section 27 of the Evidence Act").
- The answer must be supported by the passage.

Return strictly valid JSON: {{"query": "...", "answer": "1-2 sentence answer grounded in the passage", "statutes": ["..."]}}

Passage:
\"\"\"{passage}\"\"\"
"""

FACT_PROMPT = """You are building a retrieval benchmark for an Indian criminal-law research engine over Supreme Court judgments.

Below are the facts from a judgment. Rewrite them as a lawyer would brief the facts of a NEW matter to a colleague and ask for precedent.

Strict rules:
- 50-120 words, plain prose, describing what happened, what the accused is charged with, the stage of the proceedings and what is being sought.
- Use generic roles only ("the accused", "the deceased", "the complainant", "the trial court", "the High Court").
- Do NOT include any names of people or places, dates, years, case numbers, FIR numbers, citations or judge names.
- End with one question, e.g. "Which precedents apply?" or a specific legal question.

Return strictly valid JSON: {{"query": "...", "issue": "the legal issue in one sentence"}}

Facts:
\"\"\"{passage}\"\"\"
"""


def load_corpus(db_path: Path) -> Dict[str, Dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    docs: Dict[str, Dict[str, Any]] = {}
    for r in conn.execute("SELECT doc_id, year, case_title, petitioner, respondent, judges, citation, cnr, "
                          "headnote_split FROM documents"):
        docs[r["doc_id"]] = {**dict(r), "parents": []}
    for r in conn.execute("SELECT parent_id, doc_id, text, metadata_json FROM parent_chunks ORDER BY parent_id"):
        if r["doc_id"] in docs:
            meta = json.loads(r["metadata_json"] or "{}")
            docs[r["doc_id"]]["parents"].append({"parent_id": r["parent_id"], "text": r["text"] or "",
                                                 "role": meta.get("role"), "kind": meta.get("section_kind")})
    conn.close()
    return docs


def cue_score(text: str, cues) -> int:
    low = text.lower()
    return sum(low.count(c) for c in cues)


def pick_legal(doc) -> Optional[Dict[str, Any]]:
    body = [p for p in doc["parents"] if p["kind"] == "body" and len(p["text"]) >= 600]
    ratio = [p for p in body if p["role"] == "Ratio/Analysis"]
    pool = ratio or [p for p in body if p["role"] in ("Unclassified", "Facts")]
    return max(pool, key=lambda p: cue_score(p["text"], bv.HOLDING_CUES)) if pool else None


def pick_facts(doc) -> Optional[Dict[str, Any]]:
    body = [p for p in doc["parents"] if p["kind"] == "body" and len(p["text"]) >= 600]
    facts = [p for p in body if p["role"] == "Facts"]
    if facts:
        return facts[0]
    cand = [p for p in body if p["role"] == "Unclassified" and cue_score(p["text"], bv.FACT_CUES) >= 2]
    return cand[0] if cand else None


def ask(client, model: str, prompt: str) -> Optional[Dict[str, Any]]:
    raw = get_budget().generate(client=client, models=[model], contents=prompt,
                                config=gen_config(response_mime_type="application/json", temperature=0.3))
    try:
        data = json.loads(raw.strip()) if raw else None
    except Exception:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) and data.get("query") else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate benchmark v2 queries")
    ap.add_argument("--collection", default="sc_criminal_cpu")
    ap.add_argument("--legal", type=int, default=100)
    ap.add_argument("--facts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    settings = get_settings()
    out_dir = settings.benchmarks_dir / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_gemini_client(settings.gemini_api_key)
    route = get_budget().llm_label(client)
    if route == "none configured":
        sys.exit("[ERROR] No LLM configured (LLM_API_KEY / GEMINI_API_KEY).")

    t0 = time.perf_counter()
    docs = load_corpus(settings.sqlite_path_for(args.collection))
    excluded = [d for d in docs.values() if d["headnote_split"] in ("none", "unsplit")]
    eligible = [d for d in docs.values() if d["headnote_split"] not in ("none", "unsplit")]
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in sorted(eligible, key=lambda d: d["doc_id"]):
        by_year[int(d["year"])].append(d)
    rng = random.Random(args.seed)
    for y in by_year:
        rng.shuffle(by_year[y])
    counts = {y: len(v) for y, v in sorted(by_year.items())}
    alloc_legal = bv.allocate_by_year(counts, args.legal)
    alloc_facts = bv.allocate_by_year(counts, args.facts)
    print(f"[bench-gen] {len(docs)} judgments | eligible {len(eligible)} | excluded (headnote split none/unsplit) "
          f"{len(excluded)} | LLM route: {route}", flush=True)

    queries: List[Dict[str, Any]] = []
    rejected: Counter = Counter()
    used_docs = set()
    partial = out_dir / "queries_v2.partial.json"

    def run_kind(kind: str, alloc: Dict[int, int], picker, prompt_tpl: str) -> None:
        for year in sorted(alloc):
            need, got, fails = alloc[year], 0, 0
            for doc in by_year[year]:
                if got >= need:
                    break
                if doc["doc_id"] in used_docs:
                    continue
                passage = picker(doc)
                if passage is None:
                    rejected[f"{kind}:no_passage"] += 1
                    continue
                data = ask(client, settings.gemini_model, prompt_tpl.format(passage=passage["text"][:2500]))
                if not data:
                    rejected[f"{kind}:llm_failed"] += 1
                    fails += 1
                    if fails >= 8:
                        print(f"[bench-gen] {kind} {year}: 8 LLM failures, moving on", flush=True)
                        break
                    continue
                q = " ".join(str(data["query"]).split())
                reasons = bv.leakage_reasons(q, bv.party_tokens(doc["case_title"], doc["petitioner"], doc["respondent"]),
                                             bv.judge_tokens(doc["judges"]))
                words = len(q.split())
                if not (8 <= words <= (60 if kind == "legal_issue" else 160)):
                    reasons.append("length")
                if reasons:
                    for r in reasons:
                        rejected[f"{kind}:{r.split(':')[0]}"] += 1
                    continue
                used_docs.add(doc["doc_id"])
                got += 1
                queries.append({
                    "query_id": f"{'LQ' if kind == 'legal_issue' else 'FQ'}-{sum(1 for x in queries if x['type'] == kind) + 1:03d}",
                    "type": kind, "query": q, "target_doc_ids": [doc["doc_id"]], "target_cnr": doc["cnr"],
                    "target_title": doc["case_title"], "target_citation": doc["citation"], "year": year,
                    "source_parent_id": passage["parent_id"], "source_role": passage["role"],
                    "answer": str(data.get("answer") or data.get("issue") or ""), "statutes": data.get("statutes") or [],
                    "source_passage": passage["text"][:800], "verified": False,
                })
                partial.write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[bench-gen] {kind} {year}: {got}/{need} | total {len(queries)} | {time.perf_counter() - t0:.0f}s",
                  flush=True)

    run_kind("legal_issue", alloc_legal, pick_legal, LEGAL_PROMPT)
    run_kind("fact_pattern", alloc_facts, pick_facts, FACT_PROMPT)
    for i, q in enumerate(bv.OUT_OF_SCOPE_QUERIES, start=1):
        queries.append({"query_id": f"OOS-{i:02d}", "type": "out_of_scope", "query": q, "target_doc_ids": [],
                        "expected": "refuse"})

    (out_dir / "queries_v2.json").write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
    human = out_dir / "human_queries.json"
    if not human.exists():
        human.write_text(json.dumps(bv.human_template(), indent=1), encoding="utf-8")
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "collection": args.collection,
        "seed": args.seed, "llm_route": route, "eligible_judgments": len(eligible), "excluded_headnote_split": len(excluded),
        "allocation_legal": alloc_legal, "allocation_facts": alloc_facts,
        "counts": dict(Counter(q["type"] for q in queries)), "rejections": dict(rejected),
        "source_roles": dict(Counter(f"{q['type']}:{q.get('source_role')}" for q in queries if q["type"] != "out_of_scope")),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "generation_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("=" * 90)
    print(f"BENCHMARK v2 GENERATION | {report['generated_at']} | seed {args.seed} | {report['elapsed_sec']}s")
    print(f"counts: {report['counts']} (targets: legal {args.legal}, facts {args.facts}, out_of_scope 5)")
    print(f"eligible judgments {len(eligible)} | excluded (headnote split none/unsplit) {len(excluded)}")
    print("allocation legal: " + " ".join(f"{y}:{n}" for y, n in alloc_legal.items()))
    print("allocation facts: " + " ".join(f"{y}:{n}" for y, n in alloc_facts.items()))
    print("rejections: " + (" | ".join(f"{k}:{v}" for k, v in sorted(rejected.items())) or "none"))
    print("source roles: " + " | ".join(f"{k}:{v}" for k, v in sorted(report["source_roles"].items())))
    for kind in ("legal_issue", "fact_pattern"):
        ex = [q for q in queries if q["type"] == kind][:2]
        for q in ex:
            print(f"  e.g. {q['query_id']}: {q['query'][:160]}")
    print("files: $BENCHMARKS_DIR/v2/queries_v2.json, generation_report.json, human_queries.json (template)")


if __name__ == "__main__":
    main()
