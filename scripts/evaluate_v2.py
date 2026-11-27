#!/usr/bin/env python3
"""Benchmark v2 evaluation (Phase 6), in three resumable steps. Run each with nohup.

  1. retrieval  (default, or --skip-llm)  all 5 configs on every in-scope query + out-of-scope relevance.
                Writes per_query.json and run_manifest.json IMMEDIATELY, then results.json / results.md.
  2. --llm-only                           grounded answers (Groq) for 5 out-of-scope + N in-scope queries
                                          (default 15), reusing per_query.json (no retrieval re-run).
                                          Appends to generation_answers.json after EVERY answer; resumes.
  3. --judge-only                         Gemini judge on at most --judge-n answers (default 10), with
                                          --judge-delay seconds between calls; appends to judge_rows.json
                                          after every call; resumes. Loads no models.

Every step finishes by rebuilding results.json / results.md from whatever files exist, so a stopped step
loses at most the item in progress. Anything not run is reported as "not run". All LLM calls are cached.

Configs (all ranked at JUDGMENT level, top 10, ground truth = target doc_id):
  dense          Chroma cosine on the raw query (bge-small + BGE query prefix), chunks -> judgments
  bm25           rank_bm25 on the raw query, chunks -> judgments
  hybrid_rrf     dense top 60 + BM25 top 60 fused by Reciprocal Rank Fusion, -> judgments
  full           concordance + hybrid + cross-encoder rerank (top RERANK_TOP_N) x bounded bench prior,
                 judgments by best passage; positions after the reranked judgments are filled from the fused order
  full_no_prior  as full, with the authority prior switched off (ablation)

Usage (server):
    nohup python scripts/evaluate_v2.py --skip-llm   > ~/Pratik/workspace/logs/eval_v2_retrieval.log 2>&1 &
    nohup python scripts/evaluate_v2.py --llm-only   > ~/Pratik/workspace/logs/eval_v2_gen.log 2>&1 &
    nohup python scripts/evaluate_v2.py --judge-only > ~/Pratik/workspace/logs/eval_v2_judge.log 2>&1 &
"""

import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import time
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


# --------------------------------------------------------------------------- helpers
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def write_json(path: Path, data: Any, compact: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) if compact
                   else json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def queries_path(out_dir: Path, name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else out_dir / p


def load_queries(out_dir: Path, db_path: Path, name: str = "queries_v2.json") -> List[Dict[str, Any]]:
    path = queries_path(out_dir, name)
    if not path.exists():
        sys.exit(f"[ERROR] {path.name} not found in the benchmark folder (run the generator / builder first).")
    queries = json.loads(path.read_text(encoding="utf-8"))
    human = out_dir / "human_queries.json"
    if human.exists():
        import sqlite3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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


def rank(pipe, cfg: str, query: str, k: int = 10, exclude=()) -> Dict[str, Any]:
    from src.rag_pipeline import dedupe_by_judgment, judgment_key

    vs = pipe.vector_store
    ex = set(exclude)
    kk = k + len(ex)  # fetch extra so that removing the excluded judgment still leaves k
    t0 = time.perf_counter()
    extra: Dict[str, Any] = {}
    if cfg == "dense":
        ranking = [judgment_key(h) for h in dedupe_by_judgment(vs.query_dense(query, top_k=100))][:kk]
    elif cfg == "bm25":
        ranking = [judgment_key(h) for h in dedupe_by_judgment(vs.query_bm25(query, top_k=100))][:kk]
    elif cfg == "hybrid_rrf":
        fused = pipe.compute_rrf(vs.query_dense(query, top_k=60), vs.query_bm25(query, top_k=60), k=60)
        ranking = [judgment_key(h) for h in dedupe_by_judgment(fused)][:kk]
    else:
        out = pipe.retrieve_hybrid(query=query, top_k=kk, candidate_pool_size=60,
                                   use_authority=False if cfg == "full_no_prior" else None)
        ranking = bv.full_ranking([g["doc_id"] for g in out["judgments"]], out["fused_doc_order"], kk)
        extra["best_relevance"] = max((g["relevance"] for g in out["judgments"] if g["doc_id"] not in ex), default=0.0)
    extra["ms"] = (time.perf_counter() - t0) * 1000.0
    extra["ranking"] = bv.exclude_and_cut(ranking, ex, k)
    return extra


def judge_prompt(question: str, context: str, answer: str, max_chars: int = 24000) -> Dict[str, Any]:
    """The judge sees EXACTLY the generator's precedent blocks (all N; passages shortened only if needed)."""
    fit = bv.fit_context(context, max_chars=max_chars)
    n = fit["n_precedents"]
    prompt = (f"Question:\n{question}\n\n"
              f"The answering system was given exactly {n} precedents, numbered [Precedent 1] to [Precedent {n}]. "
              f"Any citation [Precedent k] with 1 <= k <= {n} refers to a precedent shown below and is NOT invented"
              f"{' (passages are shortened for length; all precedents are present)' if fit['shortened'] else ''}.\n\n"
              f"Precedents (valid range: Precedents 1..{n}):\n{fit['text']}\n\n"
              f"Answer to score:\n{answer}\n\nReturn the JSON scores.")
    return {"prompt": prompt, "n_precedents": n, "shortened": fit["shortened"]}


def judge(client, settings, question: str, context: str, answer: str, provider: str = "gemini",
          model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Score one answer. provider 'gemini' (GEMINI_MODEL) or 'groq' (exactly `model` on the OpenAI-compatible
    route, never the generator or its fallbacks)."""
    prompt = judge_prompt(question, context, answer)["prompt"]
    cfg = gen_config(system_instruction=JUDGE_SYSTEM, temperature=0.0, response_mime_type="application/json")
    if provider == "groq":
        raw = get_budget(settings).generate(client=None, models=[], contents=prompt, providers=["openai"],
                                            compat_models=[model], config=cfg)
    else:
        raw = get_budget(settings).generate(client=client, models=[model or settings.gemini_model, "gemini-flash-latest"],
                                            contents=prompt, providers=["gemini"], config=cfg)
    try:
        data = json.loads(raw.strip()) if raw else None
    except Exception:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or not all(isinstance(data.get(d), (int, float)) for d in JUDGE_DIMS):
        return None
    return {d: float(data[d]) for d in JUDGE_DIMS} | {"feedback": str(data.get("feedback", ""))[:200]}


# --------------------------------------------------------------------------- step 1: retrieval
def run_retrieval(args, settings, out_dir: Path) -> None:
    import torch

    from src.rag_pipeline import IndianLegalRAGPipeline
    from src.resource_guard import ResourceGuard

    torch.set_num_threads(settings.num_threads)
    guard = ResourceGuard(max_ram_gb=settings.max_ram_gb, warn_ram_gb=settings.warn_ram_gb,
                          num_cpu_threads=settings.num_threads)
    pipe = IndianLegalRAGPipeline(settings=settings, collection_name=args.collection)
    queries = load_queries(out_dir, pipe.sqlite_store.db_path, args.queries)
    if args.max_queries:
        queries = queries[: args.max_queries] + [q for q in queries if q["type"] == "out_of_scope"][:2]
    in_scope = [q for q in queries if q["type"] != "out_of_scope"]
    oos = [q for q in queries if q["type"] == "out_of_scope"]
    t0 = time.perf_counter()
    pipe.retrieve_hybrid(query="warm-up: bail under Section 439 CrPC", top_k=1)  # model loading is not latency
    pipe.vector_store.query_bm25("warm-up bail", top_k=1)

    per_query: List[Dict[str, Any]] = []
    for i, q in enumerate(in_scope, start=1):
        rec = {"query_id": q["query_id"], "type": q["type"], "year": q.get("year"), "targets": q["target_doc_ids"],
               "query": q["query"][:200], "target_title": q.get("target_title"), "target_citation": q.get("target_citation")}
        for cfg in CONFIGS:
            r = rank(pipe, cfg, q["query"], exclude=q.get("exclude_doc_ids") or () if args.exclude_self else ())
            m = bv.query_metrics(r["ranking"], q["target_doc_ids"])
            rec[cfg] = {**m, "ms": round(r["ms"], 1), "top3": r["ranking"][:3],
                        **({"best_relevance": round(r["best_relevance"], 4)} if "best_relevance" in r else {})}
        per_query.append(rec)
        guard.get_ram_usage_gb()
        if i % 25 == 0 or i == len(in_scope):
            print(f"[eval] {i}/{len(in_scope)} queries | {time.perf_counter() - t0:.0f}s | full hit@5 so far "
                  f"{statistics.mean(p['full']['hit@5'] for p in per_query):.3f} | RAM peak {guard.peak_ram_gb:.1f} GB",
                  flush=True)
    oos_rows = []
    for q in oos:
        r = rank(pipe, "full", q["query"])
        oos_rows.append({"query_id": q["query_id"], "query": q["query"], "best_relevance": round(r["best_relevance"], 4)})

    # saved IMMEDIATELY, before anything else can fail
    write_json(out_dir / "per_query.json", {"in_scope": per_query, "out_of_scope": oos_rows})
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
                      "min_relevance": settings.min_relevance, "candidate_pool": 60, "k": 10,
                      "exclude_self": bool(args.exclude_self)},
        "seed": SEED, "bootstrap_resamples": 1000,
        "queries": {"file_sha1": hashlib.sha1(queries_path(out_dir, args.queries).read_bytes()).hexdigest()[:12],
                    "counts": {t: sum(1 for q in queries if q["type"] == t) for t in sorted({q["type"] for q in queries})}},
        "retrieval_elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(f"[eval] retrieval saved: per_query.json + run_manifest.json ({manifest['retrieval_elapsed_sec']}s)", flush=True)


# --------------------------------------------------------------------------- step 2: answers
def run_generation(args, settings, out_dir: Path) -> None:
    from src.rag_pipeline import IndianLegalRAGPipeline

    if not (out_dir / "per_query.json").exists():
        sys.exit("[ERROR] per_query.json missing: run the retrieval step first (--skip-llm).")
    path = out_dir / "generation_answers.json"
    rows: List[Dict[str, Any]] = read_json(path, [])
    if args.redo:  # e.g. --redo out_of_scope: drop those saved rows (and their judge rows) and generate them again
        redo = set(args.redo)
        dropped = {r["query_id"] for r in rows if "all" in redo or r["type"] in redo or r["query_id"] in redo}
        rows = [r for r in rows if r["query_id"] not in dropped]
        write_json(path, rows)
        jpath = out_dir / "judge_rows.json"
        if jpath.exists():
            write_json(jpath, [j for j in read_json(jpath, []) if j["query_id"] not in dropped])
        print(f"[gen] --redo {sorted(redo)}: dropped {len(dropped)} saved answers", flush=True)
    done = {r["query_id"] for r in rows}
    pipe = IndianLegalRAGPipeline(settings=settings, collection_name=args.collection)
    queries = load_queries(out_dir, pipe.sqlite_store.db_path, args.queries)
    oos = [q for q in queries if q["type"] == "out_of_scope"]
    legal = [q for q in queries if q["type"] == "legal_issue"]
    facts = [q for q in queries if q["type"] == "fact_pattern"]
    n_facts = min(5, args.gen_n // 3)
    picked = oos + legal[: args.gen_n - n_facts] + facts[:n_facts]  # out-of-scope first: mostly no LLM call
    print(f"[gen] {len(picked)} queries ({len(oos)} out-of-scope, {args.gen_n - n_facts} legal, {n_facts} facts); "
          f"{len(done)} already done; route {get_budget(settings).llm_label(pipe.gemini_client)}", flush=True)
    for q in picked:
        if q["query_id"] in done:
            continue
        t0 = time.perf_counter()
        res = pipe.generate_answer(query=q["query"], top_k=args.gen_top_k)
        cc = res["citation_check"]
        rows.append({
            "query_id": q["query_id"], "type": q["type"], "query": q["query"], "answer": res["answer"],
            "context": res["context"], "low_confidence": res["low_confidence"], "best_relevance": res["best_relevance"],
            "generator_model": res.get("llm_model"),
            "generator_attribution": "no_llm_call" if res["low_confidence"] or not res["context"] else
                                     ("recorded" if res.get("llm_model") else "unknown (cache hit before tracking)"),
            "citation_check": cc,
            "target_in_precedents": any(p["doc_id"] in (q.get("target_doc_ids") or []) for p in res["precedents"]),
            "llm_sec": res["timings"].get("llm_generation_sec"), "wall_sec": round(time.perf_counter() - t0, 1),
        })
        write_json(path, rows)  # after EVERY answer
        print(f"[gen] {q['query_id']} {q['type']} | low_conf {res['low_confidence']} | precision "
              f"{cc['citation_precision']} | refused {cc['refused']} | llm {rows[-1]['llm_sec']}s | wall "
              f"{rows[-1]['wall_sec']}s", flush=True)


# --------------------------------------------------------------------------- step 3: judge
def run_judge(args, settings, out_dir: Path) -> None:
    answers = read_json(out_dir / "generation_answers.json", [])
    if not answers:
        sys.exit("[ERROR] generation_answers.json missing or empty: run --llm-only first.")
    provider = args.judge_provider
    client = None
    if provider == "gemini":
        model = args.judge_model or settings.gemini_model
        client = make_gemini_client(settings.gemini_api_key)
        if client is None:
            sys.exit("[ERROR] Gemini is not configured (GEMINI_API_KEY): the judge cannot run.")
    else:
        model = args.judge_model
        if not model:
            sys.exit("[ERROR] --judge-provider groq needs --judge-model (e.g. qwen/qwen3.8-27b).")
        if model == settings.llm_model:
            sys.exit(f"[ERROR] judge model {model} is the answer generator; pick a different model.")
        if not settings.llm_api_key:
            sys.exit("[ERROR] LLM_API_KEY (Groq) is not configured: the judge cannot run.")
    path = out_dir / "judge_rows.json"
    rows: List[Dict[str, Any]] = read_json(path, [])
    if args.redo:  # --judge-only --redo all | <types> | <ids>: drop those judge rows and judge again
        redo = set(args.redo)
        rows = [] if "all" in redo else [r for r in rows if r["type"] not in redo and r["query_id"] not in redo]
    failed = [r for r in rows if not r.get("scores")]
    rows = [r for r in rows if r.get("scores")]  # FAILED rows are always retried
    write_json(path, rows)
    print(f"[judge] kept {len(rows)} judged rows | retrying {len(failed)} FAILED | provider {provider} | model {model}",
          flush=True)
    print_oos(answers)
    done = {r["query_id"] for r in rows}
    todo = [a for a in answers if a["type"] != "out_of_scope" and not a["low_confidence"] and a["context"]
            and not a["citation_check"]["refused"]][: args.judge_n]
    print(f"[judge] {len(todo)} answers to judge (max {args.judge_n}); {len(done)} already done; delay "
          f"{args.judge_delay}s", flush=True)
    first = True
    for a in todo:
        if a["query_id"] in done:
            continue
        if not first:
            time.sleep(args.judge_delay)
        first = False
        n_gen = a["citation_check"]["n_precedents"]
        jp = judge_prompt(a["query"], a["context"], a["answer"])
        score = judge(client, settings, a["query"], a["context"], a["answer"], provider=provider, model=model)
        rows.append({"query_id": a["query_id"], "type": a["type"], "scores": score,
                     "judge_provider": provider, "judge_model": model,
                     "n_precedents_generator": n_gen, "n_precedents_judge": jp["n_precedents"],
                     "passages_shortened": jp["shortened"]})
        write_json(path, rows)  # after EVERY call
        flag = "" if n_gen == jp["n_precedents"] else "  MISMATCH"
        print(f"[judge] {a['query_id']} | precedents generator={n_gen} judge={jp['n_precedents']}{flag} | "
              f"shortened={jp['shortened']} -> {score if score else 'FAILED (no valid JSON / quota)'}", flush=True)


def attribute_from_log(out_dir: Path, settings, log_path: Path) -> None:
    """Fill generator_model for saved answers that lack a recorded one, from the --llm-only log's fallback trace."""
    path = out_dir / "generation_answers.json"
    rows = read_json(path, [])
    if not rows or not log_path.exists():
        print(f"[attr] nothing to do (answers: {len(rows)}, log exists: {log_path.exists()})")
        return
    chain = [settings.llm_model] + [m for m in settings.llm_fallback_models if m != settings.llm_model]
    found = bv.attribute_generators_from_log(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), chain)
    n = 0
    for r in rows:
        if r.get("generator_attribution") == "recorded":
            continue
        a = found.get(r["query_id"])
        if a is None:
            r["generator_model"], r["generator_attribution"] = "unknown", "unknown (no [gen] line in the log)"
        else:
            r["generator_model"], r["generator_attribution"] = a["model"], f"{a['source']} (probable)"
            n += 1
    write_json(path, rows)
    counts = {}
    for r in rows:
        counts[str(r.get("generator_model"))] = counts.get(str(r.get("generator_model")), 0) + 1
    print(f"[attr] chain {chain} | attributed {n} from {log_path.name} | generator models: {counts}", flush=True)


def print_oos(answers: List[Dict[str, Any]]) -> None:
    """The out-of-scope queries, as answered: id, refused, low_confidence, query text."""
    for a in [a for a in answers if a["type"] == "out_of_scope"]:
        print(f"[oos] {a['query_id']} | refused {a['citation_check']['refused']} | low_conf {a['low_confidence']} | "
              f"best_rel {a['best_relevance'] if a['best_relevance'] is None else round(a['best_relevance'], 3)} | "
              f"Q: {a['query']}", flush=True)


# --------------------------------------------------------------------------- results (from saved files)
def aggregate(out_dir: Path, settings) -> Optional[Dict[str, Any]]:
    pq = read_json(out_dir / "per_query.json", None)
    if not pq:
        return None
    per_query, oos_rows = pq["in_scope"], pq["out_of_scope"]
    manifest = read_json(out_dir / "run_manifest.json", {})
    types = sorted({p["type"] for p in per_query})
    configs_out: Dict[str, Any] = {}
    for cfg in CONFIGS:
        rows = [p[cfg] for p in per_query]
        configs_out[cfg] = {"label": LABELS[cfg], "overall": bv.summarize(rows, with_ci=True, seed=SEED),
                            "by_type": {t: bv.summarize([p[cfg] for p in per_query if p["type"] == t], with_ci=False)
                                        for t in types},
                            "latency_ms": bv.latency_stats([r["ms"] for r in rows])}
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
    min_rel = manifest.get("retrieval", {}).get("min_relevance", settings.min_relevance)
    in_rel = sorted(p["full"]["best_relevance"] for p in per_query)
    answerable = [p["full"]["best_relevance"] for p in per_query if p["full"]["hit@1"]]
    calibration = {
        "current_min_relevance": min_rel,
        "in_scope_best_relevance": {"p05": round(in_rel[max(0, int(0.05 * len(in_rel)) - 1)], 4) if in_rel else None,
                                    "median": round(statistics.median(in_rel), 4) if in_rel else None},
        "oos_best_relevance": [r["best_relevance"] for r in oos_rows],
        "sweep": [{"threshold": t,
                   "oos_refused": round(sum(r["best_relevance"] < t for r in oos_rows) / len(oos_rows), 4) if oos_rows else None,
                   "in_scope_refused": round(sum(v < t for v in in_rel) / len(in_rel), 4) if in_rel else None,
                   "hit1_refused": round(sum(v < t for v in answerable) / len(answerable), 4) if answerable else None}
                  for t in THRESHOLDS],
    }
    return {"per_query": per_query, "oos_rows": oos_rows, "manifest": manifest, "types": types, "min_rel": min_rel,
            "results": {"run": {k: manifest.get(k) for k in ("timestamp", "git_commit", "device", "collection", "corpus", "queries")},
                        "configs": configs_out, "full_by_year": by_year, "deltas": deltas,
                        "out_of_scope": {"n": len(oos_rows),
                                         "refused_at_min_relevance": sum(r["best_relevance"] < min_rel for r in oos_rows)},
                        "calibration": calibration}}


def generation_summary(out_dir: Path, settings) -> Dict[str, Any]:
    answers = read_json(out_dir / "generation_answers.json", [])
    judge_rows = read_json(out_dir / "judge_rows.json", [])
    judged = [r for r in judge_rows if r.get("scores")]
    judges = sorted({f"{r.get('judge_provider', 'gemini')}:{r.get('judge_model', settings.gemini_model)}" for r in judged})
    if not answers:
        return {"status": "not run"}
    ins = [a for a in answers if a["type"] != "out_of_scope"]
    oos = [a for a in answers if a["type"] == "out_of_scope"]
    prec = [a["citation_check"]["citation_precision"] for a in ins if a["citation_check"]["citation_precision"] is not None]
    split = bv.judge_exclusion(judge_rows, {a["query_id"]: a for a in answers})
    used = split["included"]
    gen_models: Dict[str, int] = {}
    for a in answers:
        k = str(a.get("generator_model")) if "generator_model" in a else "not recorded"
        gen_models[k] = gen_models.get(k, 0) + 1
    attributions = sorted({str(a.get("generator_attribution", "not recorded")) for a in answers})
    return {
        "status": "run",
        "answer_model": settings.llm_model,
        "generator_models": gen_models,
        "generator_attribution": attributions,
        "n_answers": len(ins),
        "n_low_confidence": sum(a["low_confidence"] for a in ins),
        "citation_precision_mean": round(statistics.mean(prec), 4) if prec else None,
        "answers_with_unknown_citations": sum(len(a["citation_check"]["unknown_precedent_numbers"]) +
                                              len(a["citation_check"]["unknown_external_citations"]) > 0 for a in ins),
        "ends_with_verify": sum(a["citation_check"]["ends_with_verify"] for a in ins),
        "in_scope_refused": sum(a["citation_check"]["refused"] for a in ins),
        "target_in_precedents": sum(a["target_in_precedents"] for a in ins),
        "llm_sec_median": round(statistics.median([a["llm_sec"] for a in ins if a["llm_sec"]]), 1) if ins else None,
        "oos_n": len(oos),
        "oos_refusal_rate": round(sum(a["citation_check"]["refused"] for a in oos) / len(oos), 4) if oos else None,
        "oos_low_confidence": sum(a["low_confidence"] for a in oos),
        "judge": ", ".join(judges) if judges else "not run",
        "judge_provider": sorted({r.get("judge_provider", "gemini") for r in judged}) or None,
        "judge_model": sorted({r.get("judge_model", settings.gemini_model) for r in judged}) or None,
        "n_judge_attempted": len(judge_rows),
        "n_judged": len(judged),
        "n_judge_excluded_same_model": len(split["excluded"]),
        "n_judge_unknown_generator": len(split["unknown"]),
        "n_judge_in_means": len(used),
        # means EXCLUDE answers whose generator model == judge model (unknown-generator answers stay in, flagged)
        "judge_means": {d: round(statistics.mean(r["scores"][d] for r in used), 2) for d in JUDGE_DIMS} if used else None,
    }


def write_results(out_dir: Path, settings) -> Optional[Dict[str, Any]]:
    agg = aggregate(out_dir, settings)
    if agg is None:
        print("[results] per_query.json missing: nothing to write yet")
        return None
    results = {**agg["results"], "generation": generation_summary(out_dir, settings)}
    write_json(out_dir / "results.json", results, compact=True)
    write_markdown(out_dir / "results.md", results, agg["manifest"])
    agg["results"] = results
    return agg


def print_summary(agg: Dict[str, Any], out_dir: Path) -> None:
    R, manifest, per_query, types = agg["results"], agg["manifest"], agg["per_query"], agg["types"]
    cfgs, cal, g = R["configs"], R["calibration"], R["generation"]
    print("=" * 100)
    print(f"EVAL v2 | {manifest.get('timestamp')} | commit {manifest.get('git_commit')} | {manifest.get('collection')} | "
          f"{(manifest.get('corpus') or {}).get('judgments')} judgments | queries {(manifest.get('queries') or {}).get('counts')}")
    print(f"{'config':<30} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR@10':>7} {'nDCG@10':>8} {'R@10':>6}  {'MRR 95% CI':>16} {'p50 ms':>7} {'p95 ms':>7}")
    for cfg in CONFIGS:
        o, lat = cfgs[cfg]["overall"], cfgs[cfg]["latency_ms"]
        print(f"{LABELS[cfg]:<30} {o['hit@1']['value']:>7.3f} {o['hit@3']['value']:>7.3f} {o['hit@5']['value']:>7.3f} "
              f"{o['hit@10']['value']:>7.3f} {o['mrr@10']['value']:>7.3f} {o['ndcg@10']['value']:>8.3f} {o['recall@10']['value']:>6.3f}  "
              f"[{o['mrr@10']['ci'][0]:.3f}, {o['mrr@10']['ci'][1]:.3f}] {lat['p50']:>7.0f} {lat['p95']:>7.0f}")
    for t in types:
        print(f"-- {t}: " + " | ".join(f"{LABELS[c]} H@5 {cfgs[c]['by_type'][t]['hit@5']:.3f} MRR {cfgs[c]['by_type'][t]['mrr@10']:.3f}"
                                       for c in ("bm25", "hybrid_rrf", "full")) + f" (n={cfgs['full']['by_type'][t]['n']})")
    print("full by year (H@5): " + " ".join(f"{y}:{v['hit@5']:.2f}(n{v['n']})" for y, v in R["full_by_year"].items()))
    for key in ("full-full_no_prior:mrr@10", "full-hybrid_rrf:mrr@10", "hybrid_rrf-bm25:mrr@10", "hybrid_rrf-dense:mrr@10"):
        d = R["deltas"][key]
        print(f"delta {key}: {d['delta']:+.4f} CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    print(f"out-of-scope best relevance: {cal['oos_best_relevance']} | refused at MIN_RELEVANCE {agg['min_rel']}: "
          f"{R['out_of_scope']['refused_at_min_relevance']}/{R['out_of_scope']['n']}")
    print(f"in-scope best relevance p05 {cal['in_scope_best_relevance']['p05']} median {cal['in_scope_best_relevance']['median']}")
    print("sweep (threshold: oos_refused / in_scope_refused / hit1_refused): " + " | ".join(
        f"{s['threshold']}: {s['oos_refused']}/{s['in_scope_refused']}/{s['hit1_refused']}" for s in cal["sweep"]))
    if g.get("status") == "run":
        print(f"generation: {g['n_answers']} in-scope answers ({g['answer_model']}, median LLM {g['llm_sec_median']}s) | "
              f"low-confidence {g['n_low_confidence']} | citation precision mean {g['citation_precision_mean']} | unknown "
              f"citations in {g['answers_with_unknown_citations']} | ends with verify {g['ends_with_verify']}/{g['n_answers']} | "
              f"target among precedents {g['target_in_precedents']}/{g['n_answers']}")
        print(f"out-of-scope via answer path: refusal rate {g['oos_refusal_rate']} (n={g['oos_n']}, low-confidence "
              f"{g['oos_low_confidence']})")
        print(f"judge: judged {g['n_judged']}, excluded {g['n_judge_excluded_same_model']} (same model as judge), "
              f"unknown generator {g['n_judge_unknown_generator']} (kept, flagged) | n judged = {g['n_judged']} of "
              f"{g['n_judge_attempted']}, judge = {g['judge']} | means over {g['n_judge_in_means']}: {g['judge_means']}")
        print(f"generator models: {g['generator_models']} | attribution: {g['generator_attribution']}")
    else:
        print("generation: not run")
    # sanity check of the ground-truth matching: first query of each type + first Full miss
    samples = [(f"sample {t}", next(p for p in per_query if p["type"] == t)) for t in types]
    samples.append(("sample miss", next((p for p in per_query if p["full"]["rank"] is None), None)))
    for label, p in samples:
        if p is None:
            print(f"{label}: (none)")
            continue
        top1 = p["full"]["top3"][0] if p["full"]["top3"] else None
        print(f"{label}: {p['query_id']} | Q: {str(p.get('query'))[:110]}")
        print(f"    targets {len(p['targets'])} (first {p['targets'][0]} {str(p.get('target_title'))[:40]}, "
              f"{p.get('target_citation')}) | Full rank {p['full']['rank']} recall@10 {p['full']['recall@10']:.2f} | "
              f"Full top1 {top1} | BM25 rank {p['bm25']['rank']}")
    size_kb = (out_dir / "results.json").stat().st_size / 1024
    print_oos(read_json(out_dir / "generation_answers.json", []))
    print(f"files: $BENCHMARKS_DIR/{out_dir.name}/results.json ({size_kb:.1f} KB), results.md, run_manifest.json, per_query.json")


def write_markdown(path: Path, results: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    corpus = manifest.get("corpus") or {}
    L = ["# Benchmark v2 results", "",
         f"Run {manifest.get('timestamp')}, commit `{manifest.get('git_commit')}`, {manifest.get('device')}, collection "
         f"`{manifest.get('collection')}` ({corpus.get('judgments')} judgments, {corpus.get('chunks')} chunks). "
         f"Queries: {(manifest.get('queries') or {}).get('counts')}. Judgment-level ranking, k = 10, 95% bootstrap CIs "
         f"(1,000 resamples, seed 42).", "",
         "| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 [95% CI] | nDCG@10 | Recall@10 | p50 ms | p95 ms |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in results["configs"].values():
        o, lat = c["overall"], c["latency_ms"]
        L.append(f"| {c['label']} | {o['hit@1']['value']:.3f} | {o['hit@3']['value']:.3f} | {o['hit@5']['value']:.3f} | "
                 f"{o['hit@10']['value']:.3f} | {o['mrr@10']['value']:.3f} [{o['mrr@10']['ci'][0]:.3f}, {o['mrr@10']['ci'][1]:.3f}] | "
                 f"{o['ndcg@10']['value']:.3f} | {o['recall@10']['value']:.3f} | {lat['p50']:.0f} | {lat['p95']:.0f} |")
    types = list(next(iter(results["configs"].values()))["by_type"].keys())
    L += ["", "## By query type (Hit@5 / MRR@10)", "", "| Type | n | " + " | ".join(c["label"] for c in results["configs"].values()) + " |",
          "|---|---|" + "---|" * len(results["configs"])]
    for t in types:
        L.append(f"| {t} | {results['configs']['full']['by_type'][t]['n']} | " + " | ".join(
            f"{c['by_type'][t]['hit@5']:.3f} / {c['by_type'][t]['mrr@10']:.3f}" for c in results["configs"].values()) + " |")
    L += ["", "## Paired deltas (bootstrap 95% CI)", ""]
    L += [f"- {k}: {v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" for k, v in results["deltas"].items()]
    cal = results["calibration"]
    L += ["", "## Refusal calibration (MIN_RELEVANCE)", "",
          f"Out-of-scope best relevance: {cal['oos_best_relevance']}; current MIN_RELEVANCE {cal['current_min_relevance']}.", "",
          "| Threshold | OOS refused | In-scope refused | Refused among Hit@1 queries |", "|---|---|---|---|"]
    L += [f"| {s['threshold']} | {s['oos_refused']} | {s['in_scope_refused']} | {s['hit1_refused']} |" for s in cal["sweep"]]
    g = results["generation"]
    L += ["", "## Generation", ""]
    if g.get("status") == "run":
        attr = g.get("generator_attribution") or []
        if any(a.startswith("unknown") or a == "not recorded" for a in attr) or g.get("n_judge_unknown_generator"):
            L.append("**Generator-model attribution is incomplete.** For some answers the model that produced them "
                     "(gpt-oss-120b or one of its fallbacks) could not be determined; those answers are kept in the "
                     "judge means and flagged (`n_judge_unknown_generator`). Attribution sources: " + ", ".join(attr) + ".")
        elif any("probable" in a for a in attr):
            L.append("Generator models were reconstructed from the fallback warnings in the generation log (probable, not "
                     "recorded at call time). Attribution sources: " + ", ".join(attr) + ".")
        L.append(f"Judge means exclude {g.get('n_judge_excluded_same_model', 0)} answer(s) whose generator model is the "
                 f"judge model; means are over {g.get('n_judge_in_means')} of {g.get('n_judged')} judged answers.")
        L.append("")
    L += ["```json", json.dumps(g, indent=1), "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate benchmark v2 (resumable steps)")
    ap.add_argument("--collection", default="sc_criminal_cpu")
    ap.add_argument("--queries", default="queries_v2.json", help="Queries file in the benchmark folder (v3: citation_queries_v3.json)")
    ap.add_argument("--tag", default="v2", help="Benchmark folder under BENCHMARKS_DIR (v2, v3a, v3b)")
    ap.add_argument("--exclude-self", action="store_true",
                    help="Remove each query's exclude_doc_ids (v3b: the citing judgment) from every ranking before scoring")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--skip-llm", action="store_true", help="Step 1 only: retrieval metrics")
    mode.add_argument("--llm-only", action="store_true", help="Step 2 only: answers, reusing per_query.json")
    mode.add_argument("--judge-only", action="store_true", help="Step 3 only: Gemini judge on saved answers")
    mode.add_argument("--results-only", action="store_true", help="Rebuild results.json/.md from saved files")
    ap.add_argument("--gen-n", type=int, default=15, help="In-scope answers to generate (step 2)")
    ap.add_argument("--gen-top-k", type=int, default=5, help="Judgments given to the LLM per answer")
    ap.add_argument("--redo", nargs="+", default=None,
                    help="With --llm-only or --judge-only: 'all', query types or ids whose saved answers / judge rows are dropped and redone "
                         "(e.g. --redo out_of_scope, or --redo out_of_scope legal_issue fact_pattern for all)")
    ap.add_argument("--judge-n", type=int, default=10, help="Max answers to judge (step 3)")
    ap.add_argument("--judge-delay", type=float, default=20.0, help="Seconds between judge calls")
    ap.add_argument("--judge-provider", choices=["gemini", "groq"], default="gemini",
                    help="groq = the OpenAI-compatible LLM_BASE_URL route with ONLY --judge-model (no fallbacks)")
    ap.add_argument("--judge-model", default=None, help="Judge model (gemini: default GEMINI_MODEL; groq: required)")
    ap.add_argument("--max-queries", type=int, default=None, help="Smoke test only")
    ap.add_argument("--attribute-from-log", default=None,
                    help="With --results-only: fill generator_model for saved answers from this --llm-only log "
                         "(fallback-warning trace; probable, not recorded)")
    args = ap.parse_args()

    settings = get_settings()
    out_dir = settings.benchmarks_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.attribute_from_log:
        attribute_from_log(out_dir, settings, Path(args.attribute_from_log).expanduser())
    if args.llm_only:
        run_generation(args, settings, out_dir)
    elif args.judge_only:
        run_judge(args, settings, out_dir)
    elif not args.results_only:
        run_retrieval(args, settings, out_dir)  # default and --skip-llm: the LLM steps are separate now
    agg = write_results(out_dir, settings)
    if agg:
        print_summary(agg, out_dir)


if __name__ == "__main__":
    main()
