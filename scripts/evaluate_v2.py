#!/usr/bin/env python3
"""Benchmark v2 evaluation (Phase 6, step 2; run with nohup after generate_benchmark_v2.py).

Configs (all ranked at JUDGMENT level, top 10, ground truth = target doc_id):
  dense          Chroma cosine on the raw query (bge-small + BGE query prefix), chunks -> judgments
  bm25           rank_bm25 on the raw query, chunks -> judgments
  hybrid_rrf     dense top 60 + BM25 top 60 fused by Reciprocal Rank Fusion, -> judgments
  full           concordance + hybrid + cross-encoder rerank (top RERANK_TOP_N) x bounded bench prior,
                 judgments by best passage; positions after the reranked judgments are filled from the fused order
  full_no_prior  as full, with the authority prior switched off (ablation)

Metrics: Hit@1/3/5/10, MRR@10, nDCG@10, Recall@10 with 95% bootstrap CIs (1,000 resamples, seed 42),
by query type and (full) by year; latency mean/p50/p95 per config; paired bootstrap deltas.
Out-of-scope queries: best cross-encoder relevance and refusal at MIN_RELEVANCE, plus a threshold sweep
(also over in-scope queries) to calibrate MIN_RELEVANCE.
Generation (optional): grounded answers (Groq) for 15 in-scope + 5 out-of-scope queries; Gemini judge
on the 15 (groundedness, ratio/obiter, statutory accuracy, reasoning); citation precision and refusal
rate from the citation check. All LLM calls are cached. Nothing is estimated: anything not run is "not run".

Outputs in $BENCHMARKS_DIR/v2/: results.json (compact), results.md, run_manifest.json, per_query.json.

Usage (server):
    nohup python scripts/evaluate_v2.py --collection sc_criminal_cpu > ~/Pratik/workspace/logs/eval_v2.log 2>&1 &
"""

import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import benchmark_v2 as bv  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm_budget import gen_config, get_budget, make_gemini_client  # noqa: E402

CONFIGS = ["dense", "bm25", "hybrid_rrf", "full", "full_no_prior"]
LABELS = {"dense": "Dense", "bm25": "BM25", "hybrid_rrf": "Hybrid RRF", "full": "Full funnel",
          "full_no_prior": "Full without authority prior"}
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SEED = 42

JUDGE_SYSTEM = """You are an expert reviewer of legal research answers on Indian criminal law. You receive a question,
the numbered Supreme Court precedents that were given to the answering system, and its answer.
Score the answer from 0 to 10 on each criterion, judging ONLY against the provided precedents:
- groundedness: every claim and every [Precedent N] citation is supported by the provided precedents (10 = nothing unsupported)
- ratio_obiter: correctly separates the binding holding (ratio decidendi) from obiter dicta / submissions
- statutory_accuracy: statutory sections are correct, incl. the IPC/BNS (CrPC/BNSS, Evidence Act/BSA) rule by date of offence
- reasoning: the legal reasoning from the precedents to the answer is sound and answers the question
Return strictly valid JSON: {"groundedness": n, "ratio_obiter": n, "statutory_accuracy": n, "reasoning": n, "feedback": "one sentence"}"""
JUDGE_DIMS = ("groundedness", "ratio_obiter", "statutory_accuracy", "reasoning")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_queries(out_dir: Path, store) -> List[Dict[str, Any]]:
    path = out_dir / "queries_v2.json"
    if not path.exists():
        sys.exit("[ERROR] queries_v2.json not found: run scripts/generate_benchmark_v2.py first.")
    queries = json.loads(path.read_text(encoding="utf-8"))
    human = out_dir / "human_queries.json"
    if human.exists():
        import sqlite3

        conn = sqlite3.connect(f"file:{store.db_path}?mode=ro", uri=True)
        for h in json.loads(human.read_text(encoding="utf-8")):
            ids = list(h.get("target_doc_ids") or [])
            for cit in h.get("target_citations") or []:
                ids += [r[0] for r in conn.execute("SELECT doc_id FROM documents WHERE citation = ? OR case_id = ?",
                                                   (cit.strip(), cit.strip()))]
            if str(h.get("query") or "").strip() and ids:
                queries.append({"query_id": h["query_id"], "type": "human", "query": h["query"].strip(),
                                "target_doc_ids": sorted(set(ids)), "year": None})
        conn.close()
    return queries


def rank(pipe, cfg: str, query: str, k: int = 10) -> Dict[str, Any]:
    from src.rag_pipeline import dedupe_by_judgment, judgment_key

    vs = pipe.vector_store
    t0 = time.perf_counter()
    extra: Dict[str, Any] = {}
    if cfg == "dense":
        ranking = [judgment_key(h) for h in dedupe_by_judgment(vs.query_dense(query, top_k=100))][:k]
    elif cfg == "bm25":
        ranking = [judgment_key(h) for h in dedupe_by_judgment(vs.query_bm25(query, top_k=100))][:k]
    elif cfg == "hybrid_rrf":
        fused = pipe.compute_rrf(vs.query_dense(query, top_k=60), vs.query_bm25(query, top_k=60), k=60)
        ranking = [judgment_key(h) for h in dedupe_by_judgment(fused)][:k]
    else:
        out = pipe.retrieve_hybrid(query=query, top_k=k, candidate_pool_size=60,
                                   use_authority=False if cfg == "full_no_prior" else None)
        ranking = bv.full_ranking([g["doc_id"] for g in out["judgments"]], out["fused_doc_order"], k)
        extra["best_relevance"] = max((g["relevance"] for g in out["judgments"]), default=0.0)
        extra["n_reranked_judgments"] = len(out["judgments"])
    extra["ms"] = (time.perf_counter() - t0) * 1000.0
    extra["ranking"] = ranking
    return extra


def judge(client, settings, question: str, context: str, answer: str) -> Optional[Dict[str, Any]]:
    prompt = (f"Question:\n{question}\n\nPrecedents given to the system:\n{context[:6000]}\n\n"
              f"Answer to score:\n{answer}\n\nReturn the JSON scores.")
    raw = get_budget(settings).generate(client=client, models=[settings.gemini_model, "gemini-flash-latest"],
                                        contents=prompt, providers=["gemini"],
                                        config=gen_config(system_instruction=JUDGE_SYSTEM, temperature=0.0,
                                                          response_mime_type="application/json"))
    try:
        data = json.loads(raw.strip()) if raw else None
    except Exception:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or not all(isinstance(data.get(d), (int, float)) for d in JUDGE_DIMS):
        return None
    return {d: float(data[d]) for d in JUDGE_DIMS} | {"feedback": str(data.get("feedback", ""))[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate benchmark v2")
    ap.add_argument("--collection", default="sc_criminal_cpu")
    ap.add_argument("--skip-llm", action="store_true", help="Retrieval metrics only (no answers, no judge)")
    ap.add_argument("--judge-n", type=int, default=15)
    ap.add_argument("--max-queries", type=int, default=None, help="Smoke test only")
    args = ap.parse_args()

    import torch

    from src.rag_pipeline import IndianLegalRAGPipeline
    from src.resource_guard import ResourceGuard

    settings = get_settings()
    torch.set_num_threads(settings.num_threads)
    guard = ResourceGuard(max_ram_gb=settings.max_ram_gb, warn_ram_gb=settings.warn_ram_gb, num_cpu_threads=settings.num_threads)
    out_dir = settings.benchmarks_dir / "v2"
    pipe = IndianLegalRAGPipeline(settings=settings, collection_name=args.collection)
    queries = load_queries(out_dir, pipe.sqlite_store)
    if args.max_queries:
        queries = queries[: args.max_queries] + [q for q in queries if q["type"] == "out_of_scope"][:2]
    in_scope = [q for q in queries if q["type"] != "out_of_scope"]
    oos = [q for q in queries if q["type"] == "out_of_scope"]
    t_start = time.perf_counter()

    # warm-up (model loading is not latency)
    pipe.retrieve_hybrid(query="warm-up: bail under Section 439 CrPC", top_k=1)
    pipe.vector_store.query_bm25("warm-up bail", top_k=1)

    per_query: List[Dict[str, Any]] = []
    for i, q in enumerate(in_scope, start=1):
        rec = {"query_id": q["query_id"], "type": q["type"], "year": q.get("year"), "targets": q["target_doc_ids"],
               "query": q["query"][:200], "target_title": q.get("target_title"), "target_citation": q.get("target_citation")}
        for cfg in CONFIGS:
            r = rank(pipe, cfg, q["query"])
            m = bv.query_metrics(r["ranking"], q["target_doc_ids"])
            rec[cfg] = {**m, "ms": round(r["ms"], 1), "top3": r["ranking"][:3],
                        **({"best_relevance": round(r["best_relevance"], 4)} if "best_relevance" in r else {})}
        per_query.append(rec)
        guard.get_ram_usage_gb()
        if i % 25 == 0 or i == len(in_scope):
            print(f"[eval] {i}/{len(in_scope)} queries | {time.perf_counter() - t_start:.0f}s | "
                  f"full hit@5 so far {statistics.mean(p['full']['hit@5'] for p in per_query):.3f} | "
                  f"RAM peak {guard.peak_ram_gb:.1f} GB", flush=True)

    oos_rows = []
    for q in oos:
        r = rank(pipe, "full", q["query"])
        oos_rows.append({"query_id": q["query_id"], "query": q["query"], "best_relevance": round(r["best_relevance"], 4),
                         "refused_at_min_relevance": r["best_relevance"] < settings.min_relevance})

    # ------------------------------------------------------------------ aggregate retrieval
    types = sorted({p["type"] for p in per_query})
    configs_out: Dict[str, Any] = {}
    for cfg in CONFIGS:
        rows = [p[cfg] for p in per_query]
        configs_out[cfg] = {
            "label": LABELS[cfg],
            "overall": bv.summarize(rows, with_ci=True, seed=SEED),
            "by_type": {t: bv.summarize([p[cfg] for p in per_query if p["type"] == t], with_ci=False) for t in types},
            "latency_ms": bv.latency_stats([r["ms"] for r in rows]),
        }
    by_year = {}
    for y in sorted({p["year"] for p in per_query if p["year"]}):
        rows = [p["full"] for p in per_query if p["year"] == y]
        by_year[str(y)] = {"n": len(rows), "hit@5": round(statistics.mean(r["hit@5"] for r in rows), 4),
                           "mrr@10": round(statistics.mean(r["mrr@10"] for r in rows), 4)}
    deltas = {}
    for a, b in (("full", "full_no_prior"), ("full", "hybrid_rrf"), ("hybrid_rrf", "bm25"), ("hybrid_rrf", "dense")):
        for metric in ("mrr@10", "hit@5"):
            deltas[f"{a}-{b}:{metric}"] = bv.paired_bootstrap_delta([p[a][metric] for p in per_query],
                                                                    [p[b][metric] for p in per_query], seed=SEED)

    in_rel = [p["full"]["best_relevance"] for p in per_query]
    answerable = [p["full"]["best_relevance"] for p in per_query if p["full"]["hit@1"]]
    calibration = {
        "current_min_relevance": settings.min_relevance,
        "in_scope_best_relevance": {"p05": round(sorted(in_rel)[max(0, int(0.05 * len(in_rel)) - 1)], 4) if in_rel else None,
                                    "median": round(statistics.median(in_rel), 4) if in_rel else None},
        "oos_best_relevance": [r["best_relevance"] for r in oos_rows],
        "sweep": [{"threshold": t,
                   "oos_refused": round(sum(r["best_relevance"] < t for r in oos_rows) / len(oos_rows), 4) if oos_rows else None,
                   "in_scope_refused": round(sum(v < t for v in in_rel) / len(in_rel), 4) if in_rel else None,
                   "hit1_refused": round(sum(v < t for v in answerable) / len(answerable), 4) if answerable else None}
                  for t in THRESHOLDS],
    }

    # ------------------------------------------------------------------ generation + judge
    generation: Dict[str, Any] = {"status": "not run (--skip-llm)"}
    if not args.skip_llm:
        client = make_gemini_client(settings.gemini_api_key)
        picked = [q for q in in_scope if q["type"] == "legal_issue"][: max(0, args.judge_n - 5)] + \
                 [q for q in in_scope if q["type"] == "fact_pattern"][:5]
        gen_rows, judged = [], []
        for q in picked:
            res = pipe.generate_answer(query=q["query"], top_k=5)
            cc = res["citation_check"]
            row = {"query_id": q["query_id"], "type": q["type"], "low_confidence": res["low_confidence"],
                   "citation_precision": cc["citation_precision"], "unknown_citations": len(cc["unknown_precedent_numbers"]) +
                   len(cc["unknown_external_citations"]), "refused": cc["refused"], "ends_with_verify": cc["ends_with_verify"],
                   "target_in_precedents": any(p["doc_id"] in q["target_doc_ids"] for p in res["precedents"]),
                   "llm_sec": res["timings"].get("llm_generation_sec")}
            if client is not None and not res["low_confidence"] and res["context"]:
                score = judge(client, settings, q["query"], res["context"], res["answer"])
                if score:
                    row["judge"] = score
                    judged.append(score)
            gen_rows.append(row)
            print(f"[gen] {q['query_id']} precision={row['citation_precision']} judged={'judge' in row}", flush=True)
        oos_gen = [pipe.generate_answer(query=q["query"], top_k=5) for q in oos]
        precisions = [r["citation_precision"] for r in gen_rows if r["citation_precision"] is not None]
        generation = {
            "status": "run",
            "answer_model": get_budget(settings).llm_label(client),
            "judge": "gemini (" + settings.gemini_model + ")" if client is not None else "not run (no GEMINI_API_KEY)",
            "n_answers": len(gen_rows),
            "n_judged": len(judged),
            "judge_means": {d: round(statistics.mean(j[d] for j in judged), 2) for d in JUDGE_DIMS} if judged else None,
            "citation_precision_mean": round(statistics.mean(precisions), 4) if precisions else None,
            "answers_with_unknown_citations": sum(r["unknown_citations"] > 0 for r in gen_rows),
            "ends_with_verify": sum(r["ends_with_verify"] for r in gen_rows),
            "in_scope_refused": sum(r["refused"] for r in gen_rows),
            "target_in_precedents": sum(r["target_in_precedents"] for r in gen_rows),
            "oos_refusal_rate": round(sum(g["citation_check"]["refused"] for g in oos_gen) / len(oos_gen), 4) if oos_gen else None,
            "oos_low_confidence": sum(g["low_confidence"] for g in oos_gen),
        }
        (out_dir / "generation_rows.json").write_text(json.dumps(gen_rows, indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ write outputs
    vs = pipe.vector_store
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_threads": settings.num_threads,
        "collection": args.collection,
        "corpus": {"chunks": vs.get_or_create_collection(args.collection).count(),
                   "judgments": pipe.sqlite_store.count_documents(), "parents": pipe.sqlite_store.count_parents()},
        "models": {"embedding": settings.local_embedding_model, "query_instruction": settings.query_instruction,
                   "reranker": settings.reranker_model, "answer_llm": settings.llm_model, "judge_llm": settings.gemini_model},
        "chunking": {"parent": settings.parent_chunk_size, "child": settings.child_chunk_size, "overlap": settings.chunk_overlap},
        "retrieval": {"rerank_top_n": settings.rerank_top_n, "passages_per_judgment": settings.passages_per_judgment,
                      "authority_strength": settings.authority_strength, "authority_max_bonus": settings.authority_max_bonus,
                      "min_relevance": settings.min_relevance, "candidate_pool": 60, "k": 10},
        "seed": SEED, "bootstrap_resamples": 1000,
        "queries": {"file_sha1": hashlib.sha1((out_dir / "queries_v2.json").read_bytes()).hexdigest()[:12],
                    "counts": {t: sum(1 for q in queries if q["type"] == t) for t in sorted({q["type"] for q in queries})}},
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
    }
    results = {"run": {k: manifest[k] for k in ("timestamp", "git_commit", "device", "collection", "corpus", "queries")},
               "configs": configs_out, "full_by_year": by_year, "deltas": deltas,
               "out_of_scope": {"n": len(oos_rows), "refused_at_min_relevance": sum(r["refused_at_min_relevance"] for r in oos_rows)},
               "calibration": calibration, "generation": generation}
    (out_dir / "results.json").write_text(json.dumps(results, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out_dir / "per_query.json").write_text(json.dumps({"in_scope": per_query, "out_of_scope": oos_rows}, indent=1),
                                            encoding="utf-8")
    write_markdown(out_dir / "results.md", results, manifest)

    # ------------------------------------------------------------------ summary (<= 60 lines)
    size_kb = (out_dir / "results.json").stat().st_size / 1024
    print("=" * 100)
    print(f"EVAL v2 | {manifest['timestamp']} | commit {manifest['git_commit']} | {args.collection} | "
          f"{manifest['corpus']['judgments']} judgments | queries {manifest['queries']['counts']} | {manifest['elapsed_sec']}s")
    print(f"{'config':<30} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR@10':>7} {'nDCG@10':>8}  {'MRR 95% CI':>16} {'p50 ms':>7} {'p95 ms':>7}")
    for cfg in CONFIGS:
        o, lat = configs_out[cfg]["overall"], configs_out[cfg]["latency_ms"]
        print(f"{LABELS[cfg]:<30} {o['hit@1']['value']:>7.3f} {o['hit@3']['value']:>7.3f} {o['hit@5']['value']:>7.3f} "
              f"{o['hit@10']['value']:>7.3f} {o['mrr@10']['value']:>7.3f} {o['ndcg@10']['value']:>8.3f}  "
              f"[{o['mrr@10']['ci'][0]:.3f}, {o['mrr@10']['ci'][1]:.3f}] {lat['p50']:>7.0f} {lat['p95']:>7.0f}")
    for t in types:
        print(f"-- {t}: " + " | ".join(f"{LABELS[c]} H@5 {configs_out[c]['by_type'][t]['hit@5']:.3f} "
                                       f"MRR {configs_out[c]['by_type'][t]['mrr@10']:.3f}" for c in ("bm25", "hybrid_rrf", "full"))
              + f" (n={configs_out['full']['by_type'][t]['n']})")
    print("full by year (H@5): " + " ".join(f"{y}:{v['hit@5']:.2f}(n{v['n']})" for y, v in by_year.items()))
    for key in ("full-full_no_prior:mrr@10", "full-hybrid_rrf:mrr@10", "hybrid_rrf-bm25:mrr@10", "hybrid_rrf-dense:mrr@10"):
        d = deltas[key]
        print(f"delta {key}: {d['delta']:+.4f} CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    print(f"out-of-scope best relevance: {calibration['oos_best_relevance']} | refused at MIN_RELEVANCE "
          f"{settings.min_relevance}: {results['out_of_scope']['refused_at_min_relevance']}/{len(oos_rows)}")
    print(f"in-scope best relevance p05 {calibration['in_scope_best_relevance']['p05']} median "
          f"{calibration['in_scope_best_relevance']['median']}")
    print("sweep (threshold: oos_refused / in_scope_refused / hit1_refused): " + " | ".join(
        f"{s['threshold']}: {s['oos_refused']}/{s['in_scope_refused']}/{s['hit1_refused']}" for s in calibration["sweep"]))
    g = generation
    if g.get("status") == "run":
        print(f"generation: {g['n_answers']} answers ({g['answer_model']}) | judged {g['n_judged']} by {g['judge']} | "
              f"judge means {g['judge_means']}")
        print(f"citation precision mean {g['citation_precision_mean']} | answers with unknown citations "
              f"{g['answers_with_unknown_citations']} | ends with verify {g['ends_with_verify']}/{g['n_answers']} | "
              f"in-scope refused {g['in_scope_refused']} | target among precedents {g['target_in_precedents']}/{g['n_answers']} | "
              f"OOS refusal rate {g['oos_refusal_rate']}")
    else:
        print(f"generation: {g['status']}")
    # sanity check of the ground-truth matching: one legal-issue, one fact-pattern, one Full miss
    samples = [next((p for p in per_query if p["type"] == "legal_issue"), None),
               next((p for p in per_query if p["type"] == "fact_pattern"), None),
               next((p for p in per_query if p["full"]["rank"] is None), None)]
    titles = {}
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{pipe.sqlite_store.db_path}?mode=ro", uri=True)
        ids = {d for p in samples if p for d in p["full"]["top3"][:1]}
        titles = {r[0]: r[1] for r in conn.execute(
            f"SELECT doc_id, case_title FROM documents WHERE doc_id IN ({','.join('?' * len(ids))})", list(ids))} if ids else {}
        conn.close()
    except Exception:
        pass
    for label, p in zip(("sample legal", "sample facts", "sample miss"), samples):
        if p is None:
            print(f"{label}: (none)")
            continue
        top1 = p["full"]["top3"][0] if p["full"]["top3"] else None
        print(f"{label}: {p['query_id']} | Q: {p['query'][:110]}")
        print(f"    target {p['targets'][0]} {str(p.get('target_title'))[:45]} ({p.get('target_citation')}) | Full rank "
              f"{p['full']['rank']} | Full top1 {top1} {str(titles.get(top1, ''))[:40]} | BM25 rank {p['bm25']['rank']}")
    print(f"files: $BENCHMARKS_DIR/v2/results.json ({size_kb:.1f} KB), results.md, run_manifest.json, per_query.json")


def write_markdown(path: Path, results: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    L = [f"# Benchmark v2 results", "",
         f"Run {manifest['timestamp']}, commit `{manifest['git_commit']}`, {manifest['device']}, collection "
         f"`{manifest['collection']}` ({manifest['corpus']['judgments']} judgments, {manifest['corpus']['chunks']} chunks). "
         f"Queries: {manifest['queries']['counts']}. Judgment-level ranking, k = 10, 95% bootstrap CIs (1,000 resamples, seed 42).", "",
         "| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 [95% CI] | nDCG@10 | p50 ms | p95 ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for cfg, c in results["configs"].items():
        o, lat = c["overall"], c["latency_ms"]
        L.append(f"| {c['label']} | {o['hit@1']['value']:.3f} | {o['hit@3']['value']:.3f} | {o['hit@5']['value']:.3f} | "
                 f"{o['hit@10']['value']:.3f} | {o['mrr@10']['value']:.3f} [{o['mrr@10']['ci'][0]:.3f}, {o['mrr@10']['ci'][1]:.3f}] | "
                 f"{o['ndcg@10']['value']:.3f} | {lat['p50']:.0f} | {lat['p95']:.0f} |")
    L += ["", "## By query type (Hit@5 / MRR@10)", "", "| Type | n | " + " | ".join(c["label"] for c in results["configs"].values()) + " |",
          "|---|---|" + "---|" * len(results["configs"])]
    types = next(iter(results["configs"].values()))["by_type"].keys()
    for t in types:
        n = results["configs"]["full"]["by_type"][t]["n"]
        L.append(f"| {t} | {n} | " + " | ".join(f"{c['by_type'][t]['hit@5']:.3f} / {c['by_type'][t]['mrr@10']:.3f}"
                                                for c in results["configs"].values()) + " |")
    L += ["", "## Paired deltas (bootstrap 95% CI)", ""]
    L += [f"- {k}: {v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" for k, v in results["deltas"].items()]
    cal = results["calibration"]
    L += ["", "## Refusal calibration (MIN_RELEVANCE)", "",
          f"Out-of-scope best relevance: {cal['oos_best_relevance']}; current MIN_RELEVANCE {cal['current_min_relevance']}.", "",
          "| Threshold | OOS refused | In-scope refused | Refused among Hit@1 queries |", "|---|---|---|---|"]
    L += [f"| {s['threshold']} | {s['oos_refused']} | {s['in_scope_refused']} | {s['hit1_refused']} |" for s in cal["sweep"]]
    L += ["", "## Generation", "", "```json", json.dumps(results["generation"], indent=1), "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
