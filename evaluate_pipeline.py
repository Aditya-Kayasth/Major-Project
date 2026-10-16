#!/usr/bin/env python3
"""Automated Benchmarking and Ablation Testing Suite for Indian Legal RAG.

Evaluates 4 retrieval configurations across 10 golden legal queries:
  Config 1: Naive Dense (ChromaDB cosine)
  Config 2: BM25 Lexical (BM25Okapi keyword)
  Config 3: Hybrid RRF (Dense + BM25 + Reciprocal Rank Fusion + Parent Expansion)
  Config 4: Full Legal Funnel (Hybrid RRF + BGE Cross-Encoder + Parent Expansion)

Computes retrieval metrics (Hit@1, Hit@3, Hit@5, MRR, Latency p50/p95)
and generation metrics via LLM-as-a-Judge (Factual Groundedness, Ratio Decidendi,
Statutory Concordance, Legal Reasoning Quality).
"""

import argparse
import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tqdm import tqdm

from src.config import Settings, get_settings
from src.llm_budget import get_budget
from src.rag_pipeline import GROUNDED_LEGAL_SYSTEM_INSTRUCTION, IndianLegalRAGPipeline
from src.vector_store import VectorStoreManager

from src.resource_guard import ResourceGuard

_settings = get_settings()
_guard = ResourceGuard(
    max_ram_gb=_settings.max_ram_gb,
    warn_ram_gb=_settings.warn_ram_gb,
    max_storage_gb=_settings.max_storage_gb,
    vram_safety_fraction=_settings.vram_safety_fraction,
    num_cpu_threads=_settings.num_threads,
)
_guard.initialize_hardware()

# Configure logger
logger = logging.getLogger("IndianLegalRAG.Evaluator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

LLM_JUDGE_SYSTEM_INSTRUCTION = """You are a High Court Judicial Research Judge evaluating AI-generated legal analyses against gold-standard Indian legal benchmarks and retrieved context.

Score the generated answer on a strict 0.0 to 10.0 scale across these 4 distinct judicial dimensions:
1. factual_groundedness: 0-10 (10 = zero hallucinated citations, judge names, or statutory sections; 0 = completely fabricated).
2. ratio_decidendi_isolation: 0-10 (10 = clearly isolates the binding legal holding; 0 = confuses with passing remarks or misses the core question of law).
3. statutory_concordance: 0-10 (10 = precise statutory sections and accurate IPC <-> BNS cross-references where applicable; 0 = erroneous or missing sections).
4. legal_reasoning_quality: 0-10 (10 = flawless judicial deduction from facts to legal holding; 0 = illogical or contradictory).

Output MUST be strictly valid JSON in this exact structure:
{
  "factual_groundedness": float,
  "ratio_decidendi_isolation": float,
  "statutory_concordance": float,
  "legal_reasoning_quality": float,
  "overall_score": float,
  "feedback": "Concise judicial rationale"
}"""

JUDGE_DIMENSIONS = [
    "factual_groundedness",
    "ratio_decidendi_isolation",
    "statutory_concordance",
    "legal_reasoning_quality",
    "overall_score",
]
JUDGE_UNAVAILABLE: Dict[str, Any] = {dim: None for dim in JUDGE_DIMENSIONS}

# Corpus case aliases to resolve numeric stems and abbreviations to canonical benchmark targets
CORPUS_CASE_ALIASES = {
    "654b1c982df7e33902d05c9e": ["chandrapal", "chandrapal singh", "2023:ahc:212021-fb"],
    "2026041142": ["minor child k", "2025:dhc:1142", "minor victim", "vulnerable witness"],
    "2026071765": ["pila pahan", "2026 insc 604", "jharkhand", "reserved judgment"],
    "2026092958": ["basudev", "2026 insc 831", "sanjay kumar", "lakshmi ram bhuyan"],
    "v_r_sanal_kumar": ["dr. v.r. sanal kumar", "sanal kumar", "2023 insc 524"],
    "2026020987": ["ram swaroop gupta", "2026:dhc:987"],
    "cybercrime": ["cyber forensics", "2024:hc:cyber-01", "cyber crime", "sop for cyber forensics"],
}


def chunk_doc_stem(chunk: Dict[str, Any]) -> str:
    """Return the source file stem of a retrieved chunk (document identity for evaluation)."""
    meta = chunk.get("metadata") or {}
    source_path = str(chunk.get("source_path") or meta.get("source_path") or "")
    # Normalise Windows paths so stems resolve identically on Linux
    return Path(source_path.replace("\\", "/")).stem if source_path else ""


def is_case_match(
    retrieved_chunk: Dict[str, Any],
    target_case_title: str,
    target_citations: Optional[List[str]] = None,
    target_doc_stem: Optional[str] = None,
) -> bool:
    """Determine if a retrieved candidate matches the target gold benchmark case.

    When the golden query carries a ``target_doc_stem`` the match is exact on document
    identity (the source file stem). This is the only mode used for reported metrics:
    the fuzzy title heuristics below are a legacy fallback and produce false positives
    on common surnames (e.g. any "... Singh" case matching "Chandrapal Singh").
    """
    if target_doc_stem:
        return chunk_doc_stem(retrieved_chunk).lower() == target_doc_stem.lower()

    if not target_case_title:
        return False

    meta = retrieved_chunk.get("metadata") or {}
    case_title = str(retrieved_chunk.get("case_title") or meta.get("case_title") or "").lower()
    source_path = str(retrieved_chunk.get("source_path") or meta.get("source_path") or "").lower()
    text = str(retrieved_chunk.get("text") or "").lower()
    stem = Path(source_path).stem.lower() if source_path else ""

    target_clean = target_case_title.lower().strip()

    # 1. Canonical Corpus Alias Mapping (resolves numeric stems e.g. 654b..., 2026041142)
    for key, aliases in CORPUS_CASE_ALIASES.items():
        if key in source_path or key in case_title or key in stem:
            if any(a in target_clean for a in aliases) or any(target_clean in a for a in aliases):
                return True
            if target_citations:
                for cit in target_citations:
                    c_clean = cit.lower().strip()
                    if any(a in c_clean or c_clean in a for a in aliases):
                        return True

    # 2. Direct substring match on combined metadata
    combined_meta = f"{case_title} {source_path} {stem}".lower()
    if target_clean in combined_meta:
        return True

    # 3. Check citations in metadata or text
    if target_citations:
        for cit in target_citations:
            c_clean = cit.lower().strip()
            if c_clean in combined_meta or c_clean in text:
                return True

    # 4. Token overlap of significant legal names (excluding stopwords)
    stopwords = {"vs", "versus", "state", "union", "india", "of", "and", "the", "in", "appeal", "no"}
    target_words = set(re.findall(r"\b\w{3,}\b", target_clean)) - stopwords
    meta_words = set(re.findall(r"\b\w{3,}\b", combined_meta)) - stopwords

    if target_words:
        intersection = target_words.intersection(meta_words)
        if len(intersection) >= max(1, len(target_words) // 2):
            return True

    # 5. Check text for target title or rare citations
    if target_clean in text[:2000]:
        return True

    citations = re.findall(r"\b(?:19|20)\d{2}\s+INSC\s+\d+\b", target_clean)
    for cit in citations:
        if cit in text:
            return True

    return False


def compute_retrieval_metrics(
    rankings: List[Optional[int]],
) -> Dict[str, float]:
    """Calculate Hit@1, Hit@3, Hit@5, and MRR for a list of target rank positions.

    Args:
        rankings: List of 1-based rank positions where target appeared (or None if not in top K).
    """
    total = len(rankings)
    if total == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}

    hits_1 = sum(1 for r in rankings if r is not None and r == 1)
    hits_3 = sum(1 for r in rankings if r is not None and r <= 3)
    hits_5 = sum(1 for r in rankings if r is not None and r <= 5)
    reciprocal_ranks = [1.0 / r if (r is not None and 1 <= r <= 5) else 0.0 for r in rankings]

    return {
        "hit@1": round(hits_1 / total, 4),
        "hit@3": round(hits_3 / total, 4),
        "hit@5": round(hits_5 / total, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
    }


def compute_latency_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate mean, p50, and p95 retrieval latency in milliseconds."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    }


class BenchmarkEvaluator:
    """Automated evaluation harness for Indian Legal RAG ablation configurations."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: Optional[str] = None,
        skip_llm_judge: bool = False,
        judge_configs: Optional[List[int]] = None,
        judge_limit: int = 15,
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name or self.settings.collection_name
        self.skip_llm_judge = skip_llm_judge
        # Free-tier budget: synthesis + judging costs 2 Gemini calls per (query, config), so
        # only the selected configs are judged, and only on the first `judge_limit` queries.
        # Retrieval metrics (Hit@k, MRR, latency) never call the LLM and cover every query.
        self.judge_configs = set(judge_configs or [4])
        self.judge_limit = judge_limit
        self.budget = get_budget(self.settings)

        self.vector_store = VectorStoreManager(
            settings=self.settings, collection_name=self.collection_name
        )
        self.pipeline = IndianLegalRAGPipeline(
            settings=self.settings, vector_store=self.vector_store
        )
        self.client = self.pipeline.gemini_client

    def evaluate_llm_judge(
        self,
        query: str,
        ground_truth: str,
        retrieved_context: str,
        generated_answer: str,
    ) -> Dict[str, Any]:
        """Grade a generated legal answer using Gemini 3.5 Flash as an automated judicial judge.

        Returns None scores when the judge is skipped or fails, so that placeholder values
        are never averaged into reported metrics.
        """
        if self.skip_llm_judge:
            return dict(JUDGE_UNAVAILABLE, feedback="LLM Judge evaluation skipped.")

        prompt = f"""Legal Query:
{query}

Gold-Standard Authoritative Answer:
{ground_truth}

Retrieved Legal Context:
{retrieved_context[:4000]}

Generated Legal Answer to Grade:
{generated_answer}

Evaluate strictly according to the system rubric and return JSON:"""

        raw = self.budget.generate(
            client=self.client,
            models=[self.settings.gemini_model],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=LLM_JUDGE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        try:
            parsed = json.loads(raw.strip()) if raw else None
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else None
            if isinstance(parsed, dict):
                dims = [parsed.get(d) for d in JUDGE_DIMENSIONS[:-1]]
                if "overall_score" not in parsed and all(isinstance(v, (int, float)) for v in dims):
                    parsed["overall_score"] = round(float(np.mean(dims)), 2)
                return parsed
        except Exception as err:
            logger.warning(f"Judge parsing error: {err}")

        return dict(JUDGE_UNAVAILABLE, feedback="LLM Judge call failed; excluded from averages.")

    def _judged_eval(
        self, config_num: int, q_index: int, query: str, ground_truth: str, context: str
    ) -> Dict[str, Any]:
        """Synthesize + judge only when this (config, query) is inside the LLM budget scope."""
        if self.skip_llm_judge or config_num not in self.judge_configs or q_index >= self.judge_limit:
            return dict(JUDGE_UNAVAILABLE, feedback="Not judged (outside LLM budget scope).")
        answer = self._synthesize_answer(query, context)
        if answer is None:
            return dict(JUDGE_UNAVAILABLE, feedback="Synthesis unavailable (Gemini budget/rate limit).")
        return self.evaluate_llm_judge(query, ground_truth, context, answer)

    def run_benchmark(
        self,
        queries_path: Optional[Path] = None,
        top_k: int = 5,
        champion_only: bool = False,
    ) -> Dict[str, Any]:
        """Run ablation benchmark across all 4 retrieval configurations (or champion only)."""
        bench_file = Path(queries_path or self.settings.benchmarks_dir / "golden_queries.json")
        if not bench_file.exists():
            raise FileNotFoundError(f"Golden benchmark dataset not found at {bench_file}")

        with open(bench_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        logger.info(f"Loaded {len(queries)} gold-standard legal queries for evaluation.")

        if champion_only:
            configs = ["Config 4 (Full Legal Funnel)"]
            logger.info("Evaluation Mode: CHAMPION-ONLY (Testing strictly Config 4: Full Legal Funnel).")
        else:
            configs = [
                "Config 1 (Naive Dense)",
                "Config 2 (BM25 Lexical)",
                "Config 3 (Hybrid RRF)",
                "Config 4 (Full Legal Funnel)",
            ]

        results_by_config: Dict[str, Dict[str, Any]] = {
            cfg: {
                "rankings": [],
                "latencies_ms": [],
                "judge_scores": {
                    "factual_groundedness": [],
                    "ratio_decidendi_isolation": [],
                    "statutory_concordance": [],
                    "legal_reasoning_quality": [],
                    "overall_score": [],
                },
                "query_details": [],
            }
            for cfg in configs
        }

        # Warm-up: load the embedding model and cross-encoder BEFORE any timed query, so that
        # one-time model initialisation is not reported as retrieval latency.
        logger.info("Warming up embedding model and cross-encoder (excluded from latency metrics)...")
        try:
            self.pipeline.retrieve_hybrid(
                query="warm-up query on Article 21", top_k=1, collection_name=self.collection_name
            )
        except Exception as warm_err:
            logger.warning(f"Warm-up query failed ({warm_err}); first-query latency may include model loading.")

        print("\n" + "=" * 80)
        mode_str = "CHAMPION COMBINATION (CONFIG 4 ONLY)" if champion_only else "AUTOMATED BENCHMARKING & ABLATION TEST SUITE"
        print(f" PHASE 4: {mode_str} ".center(80))
        print("=" * 80 + "\n")

        with tqdm(total=len(queries), desc="Evaluating Golden Queries", unit="query") as pbar:
            for q_index, item in enumerate(queries):
                q_id = item["query_id"]
                q_text = item.get("query_text") or item.get("query")
                target_case = item["target_case_title"]
                target_cits = item.get("target_citations") or []
                target_stem = item.get("target_doc_stem")
                gt_answer = item.get("ground_truth_answer") or item.get("expected_ratio") or ""

                if not champion_only:
                    # =========================================================
                    # Config 1: Naive Dense (Only Dense Cosine Retrieval)
                    # =========================================================
                    t0 = time.perf_counter()
                    try:
                        dense_hits = self.vector_store.query_dense(
                            query=q_text, top_k=top_k, collection_name=self.collection_name
                        )
                    except Exception as d_err:
                        logger.warning(f"Dense query failed for {q_id} ({d_err}). Recording empty candidates.")
                        dense_hits = []
                    t_dense = (time.perf_counter() - t0) * 1000.0

                    rank_c1 = None
                    for idx, hit in enumerate(dense_hits, start=1):
                        if is_case_match(hit, target_case, target_cits, target_stem):
                            rank_c1 = idx
                            break

                    c1_context = "\n\n".join(h.get("text", "") for h in dense_hits[:top_k])
                    c1_judge = self._judged_eval(1, q_index, q_text, gt_answer, c1_context)

                    results_by_config["Config 1 (Naive Dense)"]["rankings"].append(rank_c1)
                    results_by_config["Config 1 (Naive Dense)"]["latencies_ms"].append(t_dense)
                    self._record_judge_scores(results_by_config["Config 1 (Naive Dense)"], c1_judge)
                    results_by_config["Config 1 (Naive Dense)"]["query_details"].append(
                        {"query_id": q_id, "rank": rank_c1, "latency_ms": t_dense, "score": c1_judge["overall_score"]}
                    )

                    # =========================================================
                    # Config 2: BM25 Lexical (Only BM25Okapi Search)
                    # =========================================================
                    t0 = time.perf_counter()
                    bm25_hits = self.vector_store.query_bm25(
                        query=q_text, top_k=top_k, collection_name=self.collection_name
                    )
                    t_bm25 = (time.perf_counter() - t0) * 1000.0

                    rank_c2 = None
                    for idx, hit in enumerate(bm25_hits, start=1):
                        if is_case_match(hit, target_case, target_cits, target_stem):
                            rank_c2 = idx
                            break

                    c2_context = "\n\n".join(h.get("text", "") for h in bm25_hits[:top_k])
                    c2_judge = self._judged_eval(2, q_index, q_text, gt_answer, c2_context)

                    results_by_config["Config 2 (BM25 Lexical)"]["rankings"].append(rank_c2)
                    results_by_config["Config 2 (BM25 Lexical)"]["latencies_ms"].append(t_bm25)
                    self._record_judge_scores(results_by_config["Config 2 (BM25 Lexical)"], c2_judge)
                    results_by_config["Config 2 (BM25 Lexical)"]["query_details"].append(
                        {"query_id": q_id, "rank": rank_c2, "latency_ms": t_bm25, "score": c2_judge["overall_score"]}
                    )

                    # =========================================================
                    # Config 3: Hybrid RRF (Dense + BM25 + RRF + Parent Expansion)
                    # =========================================================
                    t0 = time.perf_counter()
                    try:
                        d_cands = self.vector_store.query_dense(
                            query=q_text, top_k=60, collection_name=self.collection_name
                        )
                    except Exception:
                        d_cands = []
                    b_cands = self.vector_store.query_bm25(
                        query=q_text, top_k=60, collection_name=self.collection_name
                    )
                    rrf_hits = self.pipeline.compute_rrf(d_cands, b_cands, k=60)[:top_k]
                    t_rrf = (time.perf_counter() - t0) * 1000.0

                    rank_c3 = None
                    for idx, hit in enumerate(rrf_hits, start=1):
                        if is_case_match(hit, target_case, target_cits, target_stem):
                            rank_c3 = idx
                            break

                    # Parent expansion
                    c3_parents = []
                    for h in rrf_hits:
                        pid = h.get("parent_id") or (h.get("metadata") or {}).get("parent_id")
                        if pid:
                            pdoc = self.pipeline.retrieve_parent_context(pid)
                            if pdoc:
                                c3_parents.append(pdoc.get("text", ""))

                    c3_context = "\n\n".join(c3_parents) if c3_parents else "\n\n".join(h.get("text", "") for h in rrf_hits)
                    c3_judge = self._judged_eval(3, q_index, q_text, gt_answer, c3_context)

                    results_by_config["Config 3 (Hybrid RRF)"]["rankings"].append(rank_c3)
                    results_by_config["Config 3 (Hybrid RRF)"]["latencies_ms"].append(t_rrf)
                    self._record_judge_scores(results_by_config["Config 3 (Hybrid RRF)"], c3_judge)
                    results_by_config["Config 3 (Hybrid RRF)"]["query_details"].append(
                        {"query_id": q_id, "rank": rank_c3, "latency_ms": t_rrf, "score": c3_judge["overall_score"]}
                    )

                # =========================================================
                # Config 4: Full Legal Funnel (Hybrid + RRF + BGE Reranker + Parents)
                # =========================================================
                t0 = time.perf_counter()
                full_output = self.pipeline.retrieve_hybrid(
                    query=q_text, top_k=top_k, candidate_pool_size=60, collection_name=self.collection_name
                )
                t_full = (time.perf_counter() - t0) * 1000.0

                c4_reranked = full_output.get("reranked_candidates", [])
                rank_c4 = None
                for idx, hit in enumerate(c4_reranked, start=1):
                    if is_case_match(hit, target_case, target_cits, target_stem):
                        rank_c4 = idx
                        break

                c4_parents = [p.get("text", "") for p in full_output.get("retrieved_parents", [])]
                c4_context = "\n\n".join(c4_parents) if c4_parents else "\n\n".join(c.get("text", "") for c in c4_reranked)
                c4_judge = self._judged_eval(4, q_index, q_text, gt_answer, c4_context)

                top_1_case = "None"
                if c4_reranked:
                    h0 = c4_reranked[0]
                    h0_m = h0.get("metadata") or {}
                    top_1_case = str(h0.get("case_title") or h0_m.get("case_title") or chunk_doc_stem(h0))
                top_docs = [chunk_doc_stem(h) for h in c4_reranked]

                results_by_config["Config 4 (Full Legal Funnel)"]["rankings"].append(rank_c4)
                results_by_config["Config 4 (Full Legal Funnel)"]["latencies_ms"].append(t_full)
                self._record_judge_scores(results_by_config["Config 4 (Full Legal Funnel)"], c4_judge)
                results_by_config["Config 4 (Full Legal Funnel)"]["query_details"].append(
                    {
                        "query_id": q_id,
                        "query": q_text,
                        "source": item.get("source", "curated"),
                        "target_case": target_case,
                        "target_doc_stem": target_stem,
                        "rank": rank_c4,
                        "top_1_case": top_1_case,
                        "top_docs": top_docs,
                        "hit_1": 1 if rank_c4 == 1 else 0,
                        "hit_5": 1 if (rank_c4 is not None and rank_c4 <= 5) else 0,
                        "latency_ms": t_full,
                        "score": c4_judge["overall_score"],
                    }
                )

                pbar.update(1)

        # -------------------------------------------------------------
        # Aggregate Metric Summary Table
        # -------------------------------------------------------------
        summary_table: Dict[str, Dict[str, Any]] = {}
        for cfg in configs:
            ret_metrics = compute_retrieval_metrics(results_by_config[cfg]["rankings"])
            lat_metrics = compute_latency_metrics(results_by_config[cfg]["latencies_ms"])
            js = results_by_config[cfg]["judge_scores"]

            def _judge_mean(dim: str) -> Optional[float]:
                vals = [v for v in js[dim] if isinstance(v, (int, float))]
                return round(float(np.mean(vals)), 2) if vals else None

            avg_groundedness = _judge_mean("factual_groundedness")
            avg_ratio = _judge_mean("ratio_decidendi_isolation")
            avg_statutory = _judge_mean("statutory_concordance")
            avg_reasoning = _judge_mean("legal_reasoning_quality")
            avg_overall = _judge_mean("overall_score")

            summary_table[cfg] = {
                "Hit@1": ret_metrics["hit@1"],
                "Hit@3": ret_metrics["hit@3"],
                "Hit@5": ret_metrics["hit@5"],
                "MRR": ret_metrics["mrr"],
                "Latency Mean (ms)": lat_metrics["mean_ms"],
                "Latency p50 (ms)": lat_metrics["p50_ms"],
                "Latency p95 (ms)": lat_metrics["p95_ms"],
                "Groundedness (0-10)": avg_groundedness,
                "Ratio Decidendi (0-10)": avg_ratio,
                "Statutory Concordance (0-10)": avg_statutory,
                "Reasoning Quality (0-10)": avg_reasoning,
                "Overall Judicial Score": avg_overall,
            }

        # Print formatted ASCII table
        self._print_comparison_table(summary_table)

        # Print per-query breakdown for Full Legal Funnel
        self._print_query_breakdown(results_by_config["Config 4 (Full Legal Funnel)"]["query_details"])

        # Retrieval metrics per query subset (e.g. curated vs. synthetic) for every configuration
        subset_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        subsets = sorted({q.get("source", "curated") for q in queries})
        for subset in subsets:
            idxs = [i for i, q in enumerate(queries) if q.get("source", "curated") == subset]
            subset_metrics[subset] = {
                cfg: dict(
                    compute_retrieval_metrics([results_by_config[cfg]["rankings"][i] for i in idxs]),
                    n=len(idxs),
                )
                for cfg in configs
            }
        if len(subsets) > 1:
            print("=" * 80)
            print(" RETRIEVAL METRICS BY QUERY SUBSET ".center(80))
            print("=" * 80)
            for subset, per_cfg in subset_metrics.items():
                for cfg, m in per_cfg.items():
                    print(
                        f"  {subset:<10} (n={m['n']:>3}) | {cfg:<30} | Hit@1 {m['hit@1']:.3f} | "
                        f"Hit@5 {m['hit@5']:.3f} | MRR {m['mrr']:.3f}"
                    )
            print()

        # Persist results to JSON
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.settings.benchmarks_dir / f"benchmark_results_{timestamp}.json"
        report_data = {
            "timestamp": timestamp,
            "queries_count": len(queries),
            "collection_name": self.collection_name,
            "summary_table": summary_table,
            "subset_metrics": subset_metrics,
            "raw_results": results_by_config,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n[+] Full benchmark report saved to: {output_file}\n")
        return report_data

    def _synthesize_answer(self, query: str, context: str) -> Optional[str]:
        """Contextual answer synthesis for judge evaluation (None if the LLM is unavailable)."""
        if not context.strip():
            return "Insufficient authoritative precedent found in the indexed corpus."

        prompt = f"Legal Query:\n{query}\n\nContext:\n{context[:3500]}\n\nProvide authoritative legal answer citing precedents and statutes:"
        return self.budget.generate(
            client=self.client,
            models=[self.settings.gemini_model],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GROUNDED_LEGAL_SYSTEM_INSTRUCTION, temperature=0.1
            ),
        )

    def _record_judge_scores(self, config_dict: Dict[str, Any], judge_eval: Dict[str, Any]):
        """Accumulate judicial evaluation scores."""
        js = config_dict["judge_scores"]
        for dim in JUDGE_DIMENSIONS:
            js[dim].append(judge_eval.get(dim))

    def _print_comparison_table(self, summary_table: Dict[str, Dict[str, Any]]) -> None:
        """Print comparative Markdown / ASCII table across the 4 configurations."""
        configs = list(summary_table.keys())
        metrics = list(summary_table[configs[0]].keys())

        col_w_metric = 30
        col_w_val = 18

        header_border = f"+{'-' * (col_w_metric + 2)}" + f"+{'-' * (col_w_val + 2)}" * len(configs) + "+"
        print("\n" + header_border)

        # Header row
        header = f"| {'Metric'.ljust(col_w_metric)} "
        for cfg in configs:
            short_cfg = cfg.split("(")[-1].replace(")", "").strip()
            header += f"| {short_cfg.center(col_w_val)} "
        header += "|"
        print(header)
        print(header_border)

        for m in metrics:
            row = f"| {m.ljust(col_w_metric)} "
            for cfg in configs:
                val = summary_table[cfg][m]
                if val is None:
                    val_str = "N/A (no judge)"
                elif m == "Overall Judicial Score":
                    status = "[PASS]" if val >= 7.0 else "[FAIL]"
                    val_str = f"{val:.2f} {status}"
                elif isinstance(val, float):
                    val_str = f"{val:.4f}" if val < 1.0 and val > 0.0 else f"{val:.2f}"
                else:
                    val_str = str(val)
                row += f"| {val_str.center(col_w_val)} "
            row += "|"
            print(row)

        print(header_border + "\n")

    def _print_query_breakdown(self, query_details: List[Dict[str, Any]]) -> None:
        """Print detailed per-query breakdown for Full Legal Funnel."""
        print("=" * 105)
        print(" PER-QUERY RETRIEVAL BREAKDOWN (CONFIG 4: FULL LEGAL FUNNEL) ".center(105))
        print("=" * 105)
        header = f"| {'Query ID'.ljust(9)} | {'Target Case'.ljust(33)} | {'Hit@1'.center(7)} | {'Hit@5'.center(7)} | {'Top-1 Retrieved Case'.ljust(32)} | {'Rank'.center(8)} |"
        border = f"+{'-'*11}+{'-'*35}+{'-'*9}+{'-'*9}+{'-'*34}+{'-'*10}+"
        print(border)
        print(header)
        print(border)
        for q in query_details:
            qid = str(q.get("query_id", "")).ljust(9)
            target = str(q.get("target_case", ""))[:33].ljust(33)
            h1 = "[HIT]" if q.get("hit_1") == 1 else "[MISS]"
            h5 = "[HIT]" if q.get("hit_5") == 1 else "[MISS]"
            top1 = str(q.get("top_1_case", ""))[:32].ljust(32)
            rank = f"Rank {q['rank']}" if q.get("rank") is not None else "Not Found"
            print(f"| {qid} | {target} | {h1.center(7)} | {h5.center(7)} | {top1} | {rank.center(8)} |")
        print(border + "\n")


def main():
    parser = argparse.ArgumentParser(description="Indian Legal RAG Automated Benchmarking Suite")
    parser.add_argument("--queries", type=str, default=None, help="Path to golden queries JSON")
    parser.add_argument("--collection", type=str, default=None, help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved candidates")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM Judge API scoring for quick retrieval latency check")
    parser.add_argument(
        "--champion-only",
        action="store_true",
        help="Evaluate ONLY the single champion combination (Config 4: Full Legal Funnel), skipping baseline ablations.",
    )
    parser.add_argument(
        "--judge-configs",
        type=int,
        nargs="+",
        default=[4],
        help="Configs (1-4) whose answers are synthesized and LLM-judged (2 Gemini calls per query each). Default: 4",
    )
    parser.add_argument(
        "--judge-limit",
        type=int,
        default=15,
        help="Only the first N queries are LLM-judged (free-tier budget). Retrieval metrics always use all queries.",
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        collection_name=args.collection,
        skip_llm_judge=args.skip_llm_judge,
        judge_configs=args.judge_configs,
        judge_limit=args.judge_limit,
    )
    if not args.skip_llm_judge:
        planned = len(args.judge_configs) * args.judge_limit * 2
        logger.info(
            f"LLM judge: up to {planned} calls planned via {evaluator.budget.llm_label(evaluator.client)} "
            f"(cached answers are free)."
        )
    evaluator.run_benchmark(
        queries_path=Path(args.queries) if args.queries else None,
        top_k=args.top_k,
        champion_only=args.champion_only,
    )


if __name__ == "__main__":
    main()
