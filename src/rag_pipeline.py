"""End-to-end production RAG Pipeline for Indian Legal Q&A and Precedent Search.

Phase 3: Multi-Stage Hybrid Retrieval Funnel (Metadata Slicing + Dense Cosine +
BM25Okapi + Reciprocal Rank Fusion + BGE Cross-Encoder Reranker + Parent Expansion)
and Zero-Hallucination Grounded Legal Synthesis.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import Settings, get_settings
from src.concordance import StatutoryConcordance
from src.llm_budget import get_budget
from src.resource_guard import ResourceGuard
from src.sqlite_store import SQLiteParentStore
from src.vector_store import VectorStoreManager

# Configure logger
logger = logging.getLogger("IndianLegalRAG.Pipeline")


def get_stare_decisis_weight(court_name: str, bench_type: Optional[str] = None) -> float:
    """Compute Stare Decisis authoritative weight multiplier based on court tier and bench strength."""
    c = (court_name or "").lower()
    b = (bench_type or "").lower()

    # Court Tier Multiplier
    court_weight = 1.0
    if "supreme court" in c or "insc" in c:
        court_weight = 1.25  # Apex authority binding across all jurisdictions (Art. 141)
    elif "high court" in c:
        court_weight = 1.10  # State-level constitutional precedent
    elif "tribunal" in c or "central administrative" in c:
        court_weight = 1.05

    # Bench Strength Multiplier
    bench_weight = 1.0
    if "constitution bench" in b:
        bench_weight = 1.25  # 5+ judges
    elif "full bench" in b or "full bench" in c:
        bench_weight = 1.15  # 3-4 judges
    elif "division bench" in b:
        bench_weight = 1.05  # 2 judges

    return round(court_weight * bench_weight, 4)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Grounded Legal System Prompt for Gemini
GROUNDED_LEGAL_SYSTEM_INSTRUCTION = """You are an expert Indian Legal Research Assistant aiding a Senior Advocate in High Court and Supreme Court litigation.
Your analysis must strictly adhere to the following judicial standards:
1. Zero-Hallucination Mandate: Do not fabricate citations, case law, statutory sections, or judge names. Answer relying strictly and exclusively on the provided retrieved precedent contexts.
2. Binding Precedent Distinction: Differentiate explicitly between binding Ratio Decidendi (the legal principle on which the case was decided) and non-binding Obiter Dicta (incidental judicial observations).
3. Statutory Concordance: Where relevant, provide statutory concordance between the Indian Penal Code (IPC) and the Bharatiya Nyaya Sanhita (BNS) (e.g., Section 302 IPC -> Section 103 BNS, Section 420 IPC -> Section 318 BNS, Section 304B IPC -> Section 80 BNS).
4. Precise Judicial Citations: Always cite the exact case title, court, judgment year, and relevant sections from the context.
5. Insufficient Context Handling: If the provided precedents do not contain sufficient authoritative legal backing to answer the query, clearly and concisely state: "Insufficient authoritative precedent found in the indexed corpus." """


class IndianLegalRAGPipeline:
    """Orchestrates the 5-Stage Retrieval Funnel, Cross-Encoder Reranking, and Grounded Synthesis."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[VectorStoreManager] = None,
        gemini_client: Optional[genai.Client] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStoreManager(settings=self.settings)
        self.gemini_client = gemini_client or (
            genai.Client(api_key=self.settings.gemini_api_key)
            if self.settings.gemini_api_key and self.settings.gemini_api_key != "your_gemini_api_key_here"
            else None
        )
        self.sqlite_store = SQLiteParentStore(self.settings.sqlite_db_path)
        self.guard = ResourceGuard(
            max_ram_gb=self.settings.max_ram_gb,
            warn_ram_gb=self.settings.warn_ram_gb,
            max_storage_gb=self.settings.max_storage_gb,
            num_cpu_threads=self.settings.num_threads,
        )
        self._reranker: Optional[Any] = None

    def _get_reranker(self) -> Any:
        """Lazy loader for sentence_transformers.CrossEncoder with hardware-throttled profile."""
        if self._reranker is None:
            import torch

            from src.resource_guard import configure_torch_backends

            configure_torch_backends()
            torch.set_num_threads(self.settings.num_threads)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                # Apply configured VRAM safety cap (e.g. 0.85 on B200 or 0.45 on RTX 2050)
                try:
                    torch.cuda.set_per_process_memory_fraction(self.settings.vram_safety_fraction, 0)
                    logger.info(
                        f"Configured PyTorch CUDA memory fraction: {self.settings.vram_safety_fraction} on '{torch.cuda.get_device_name(0)}'"
                    )
                except Exception as e:
                    logger.warning(f"Could not set VRAM memory fraction: {e}")

            logger.info(
                f"Lazy-initializing CrossEncoder ('{self.settings.reranker_model}') on device '{device}' (threads: {self.settings.num_threads})..."
            )
            from sentence_transformers import CrossEncoder

            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            self._reranker = CrossEncoder(
                self.settings.reranker_model,
                device=device,
                model_kwargs=model_kwargs,
            )
            if device == "cuda":
                logger.info(f"Enabled FP16 half-precision on CrossEncoder model '{self.settings.reranker_model}'.")

        return self._reranker

    def _authority_factor(self, authority_mult: float) -> float:
        """Convert a raw authority multiplier (~0.8x to ~2.25x) into a bounded tie-breaking factor.

        factor = 1 + clamp(strength * ln(mult), -max_bonus, +max_bonus)

        Raw multipliers of that size swamp both RRF scores (rank 1 vs rank 30 differ by
        only ~1.5x) and cross-encoder relevance, letting a weakly relevant Constitution
        Bench chunk outrank the actually relevant one. The bounded form keeps the legal
        hierarchy as a prior that only decides between near-equally relevant candidates.
        """
        import math

        strength = self.settings.authority_strength
        if strength <= 0 or authority_mult <= 0:
            return 1.0
        max_bonus = self.settings.authority_max_bonus
        bonus = max(-max_bonus, min(max_bonus, strength * math.log(authority_mult)))
        return 1.0 + bonus

    def retrieve_parent_context(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Fetch original parent chunk context from SQLite key-value store, falling back to JSON."""
        # 1. First attempt: Fast O(1) SQLite query
        try:
            parent_doc = self.sqlite_store.get_parent(parent_id)
            if parent_doc:
                return parent_doc
        except Exception as err:
            logger.debug(f"SQLite lookup skipped ({err}). Checking disk JSON files...")

        # 2. Second attempt: Local disk parent store
        parent_file = self.settings.parent_store_dir / f"{parent_id}.json"
        if parent_file.exists():
            try:
                with open(parent_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as err:
                logger.warning(f"Error reading parent document {parent_file}: {err}")
        return None

    def compute_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform Multi-Tier Weighted Reciprocal Rank Fusion (RRF).

        Formula:
            Final Score = (1.0 / (k + dense_rank) + 1.0 / (k + bm25_rank)) * Authority_Weight
            where Authority_Weight = Stare_Decisis_Weight * Role_Weight * Judge_Affinity
        """
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Dict[str, Any]] = {}

        # Extract target judge from query if present
        target_judge = None
        if query:
            j_match = re.search(r"\b(?:justice|judge|hon'ble\s+justice)\s+([A-Za-z\s\.]+)", query, re.IGNORECASE)
            if j_match:
                target_judge = j_match.group(1).strip()

        # 1. Process Dense Ranking
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = dict(item)
            candidate_map[chunk_id]["dense_rank"] = rank

        # 2. Process BM25 Ranking
        for rank, item in enumerate(bm25_results, start=1):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = dict(item)
            candidate_map[chunk_id]["bm25_rank"] = rank

        # 3. Build Fused List with Stare Decisis Multiplier
        fused_candidates: List[Dict[str, Any]] = []
        for chunk_id, score in rrf_scores.items():
            candidate = candidate_map[chunk_id]
            meta = candidate.get("metadata") or {}
            court = candidate.get("court") or meta.get("court", "")
            bench = candidate.get("bench_type") or meta.get("bench_type", "")
            role = candidate.get("legal_role") or meta.get("legal_role") or meta.get("role", "")
            author = candidate.get("author_judge") or meta.get("author_judge", "")
            coram = candidate.get("coram") or meta.get("coram", "")

            # 1. Stare Decisis Precedent Multiplier (Court Tier + Bench Strength)
            stare_weight = get_stare_decisis_weight(court, bench)

            # 2. Binding Ratio Decidendi Authority Multiplier
            role_l = str(role).lower()
            role_weight = 1.0
            if "ratio" in role_l or "holding" in role_l or "order" in role_l:
                role_weight = 1.15  # Binding judicial holding gets +15% boost
            elif "precedent" in role_l:
                role_weight = 1.05  # Precedent citations get +5% boost
            elif "submission" in role_l or "contention" in role_l:
                role_weight = 0.85  # Non-binding counsel arguments penalized -15%
            elif "obiter" in role_l or "dissent" in role_l:
                role_weight = 0.80  # Non-binding dicta / dissent penalized -20%

            # 3. Judicial Entity Affinity Multiplier
            judge_affinity = 1.0
            if target_judge:
                clean_target = re.sub(r"\b(justice|hon'ble|dr\.|mr\.|mrs\.|ms\.)\b", "", target_judge, flags=re.IGNORECASE).strip().lower()
                if clean_target and (clean_target in author.lower() or clean_target in coram.lower()):
                    judge_affinity = 1.25  # +25% boost for target judge opinions

            final_authority_weight = round(stare_weight * role_weight * judge_affinity, 4)

            candidate["raw_rrf_score"] = float(score)
            candidate["court_tier_weight"] = stare_weight
            candidate["role_weight"] = role_weight
            candidate["judge_affinity"] = judge_affinity
            candidate["rrf_score"] = float(score) * self._authority_factor(final_authority_weight)
            fused_candidates.append(candidate)

        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_candidates

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 60,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the Five-Stage Retrieval Funnel with granular timing instrumentation.

        Stages:
            1. Metadata Slicing (allowed IDs via ChromaDB)
            2. Candidate Generation (Dense vector + BM25Okapi, top 60 each)
            3. Reciprocal Rank Fusion (RRF k=60, top 30 candidates)
            4. Cross-Encoder Reranking with Stare Decisis Weighting (BGE reranker)
            5. Parent Document Expansion (local disk parent store)
        """
        timings: Dict[str, float] = {}
        funnel_start_t = time.perf_counter()

        # ---------------------------------------------------------
        # Stage 0: Statutory Concordance Query Expansion (IPC/CrPC <-> BNS/BNSS)
        # ---------------------------------------------------------
        search_query = StatutoryConcordance.expand_query(query)

        # ---------------------------------------------------------
        # Stage 1: Metadata Slicing
        # ---------------------------------------------------------
        s1_start = time.perf_counter()
        allowed_ids: Optional[List[str]] = None
        if filters:
            allowed_ids = self.vector_store.filter_ids_by_metadata(
                filters, collection_name=collection_name
            )
            logger.debug(f"Stage 1 Slicing: {len(allowed_ids)} candidate IDs match filters.")
        timings["metadata_filtering_sec"] = round(time.perf_counter() - s1_start, 4)

        # ---------------------------------------------------------
        # Stage 2: Parallel Candidate Generation (Dense + Sparse)
        # ---------------------------------------------------------
        s2_dense_start = time.perf_counter()
        try:
            dense_candidates = self.vector_store.query_dense(
                query=search_query,
                top_k=candidate_pool_size,
                filters=filters,
                collection_name=collection_name,
            )
        except Exception as dense_err:
            logger.warning(
                f"Dense retrieval error ({dense_err}). Gracefully falling back to sparse BM25 candidates."
            )
            dense_candidates = []
        timings["dense_retrieval_sec"] = round(time.perf_counter() - s2_dense_start, 4)

        s2_bm25_start = time.perf_counter()
        bm25_candidates = self.vector_store.query_bm25(
            query=search_query,
            top_k=candidate_pool_size,
            allowed_ids=allowed_ids,
            collection_name=collection_name,
        )
        timings["bm25_retrieval_sec"] = round(time.perf_counter() - s2_bm25_start, 4)

        # ---------------------------------------------------------
        # Stage 3: Reciprocal Rank Fusion
        # ---------------------------------------------------------
        s3_start = time.perf_counter()
        fused_candidates = self.compute_rrf(dense_candidates, bm25_candidates, k=60, query=query)
        # Select top 30 fused candidates for cross-encoder reranking
        top_fused = fused_candidates[:30]
        timings["rrf_fusion_sec"] = round(time.perf_counter() - s3_start, 4)

        # ---------------------------------------------------------
        # Stage 4: Cross-Encoder Reranking (Authority & Stare Decisis Weighted)
        # ---------------------------------------------------------
        s4_start = time.perf_counter()
        reranked_candidates: List[Dict[str, Any]] = []

        if top_fused:
            reranker = self._get_reranker()
            pairs = [[query, c["text"]] for c in top_fused]
            raw_scores = reranker.predict(pairs)

            for c, raw_score in zip(top_fused, raw_scores):
                c_copy = dict(c)
                raw_s = float(raw_score)
                # Apply numerically stable sigmoid
                try:
                    import math
                    norm_score = 1.0 / (1.0 + math.exp(-max(min(raw_s, 20.0), -20.0)))
                except Exception:
                    norm_score = 1.0 if raw_s > 0 else 0.0

                # Factor in Stare Decisis precedent multiplier, binding role boost, and judge affinity
                stare_weight = float(c.get("court_tier_weight", 1.0))
                role_weight = float(c.get("role_weight", 1.0))
                judge_affinity = float(c.get("judge_affinity", 1.0))
                final_authority_mult = stare_weight * role_weight * judge_affinity

                # Composite score: Cross-Encoder semantic match with bounded legal-authority prior
                composite_score = round(norm_score * self._authority_factor(final_authority_mult), 5)

                c_copy["raw_cross_encoder_score"] = raw_s
                c_copy["rerank_score"] = composite_score
                reranked_candidates.append(c_copy)

            reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            reranked_candidates = reranked_candidates[:top_k]
        timings["reranking_sec"] = round(time.perf_counter() - s4_start, 4)

        # ---------------------------------------------------------
        # Stage 5: Parent Document Expansion & Deduplication
        # ---------------------------------------------------------
        s5_start = time.perf_counter()
        seen_parent_ids: Set[str] = set()
        retrieved_parents: List[Dict[str, Any]] = []
        child_citations: List[Dict[str, Any]] = []

        for child in reranked_candidates:
            # Metadata resolution
            meta = child.get("metadata") or {}
            parent_id = child.get("parent_id") or meta.get("parent_id")
            case_title = child.get("case_title") or meta.get("case_title", "Unknown Case")
            court = child.get("court") or meta.get("court", "Unknown Court")
            year = child.get("year") or meta.get("year", "")
            role = child.get("role") or meta.get("role", "Obiter Dicta")

            # Collect child citation snippet
            child_citations.append(
                {
                    "chunk_id": child["chunk_id"],
                    "parent_id": parent_id,
                    "case_title": case_title,
                    "court": court,
                    "year": year,
                    "role": role,
                    "snippet": child["text"][:300],
                    "rerank_score": child.get("rerank_score", 0.0),
                }
            )

            # Load full parent context without duplicates
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_doc = self.retrieve_parent_context(parent_id)
                if parent_doc:
                    retrieved_parents.append(parent_doc)

        timings["parent_expansion_sec"] = round(time.perf_counter() - s5_start, 4)
        timings["funnel_latency_sec"] = round(time.perf_counter() - funnel_start_t, 4)

        return {
            "retrieved_parents": retrieved_parents,
            "child_citations": child_citations,
            "reranked_candidates": reranked_candidates,
            "timings": timings,
            "expanded_query": search_query,
            "funnel_counts": {
                "dense": len(dense_candidates),
                "bm25": len(bm25_candidates),
                "fused": len(fused_candidates),
                "reranked": len(top_fused),
                "returned": len(reranked_candidates),
                "parents": len(retrieved_parents),
            },
        }

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 25,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Full Pipeline: Hybrid Funnel + Grounded Legal Generation + Telemetry."""
        total_start_t = time.perf_counter()

        # Intent Routing: Detect Judge Profile / Catalog / Listing Queries
        is_catalog_query = bool(
            re.search(
                r"\b(list|all|cases\s+of|cases\s+by|profile|track\s+record|rulings\s+by|tell\s+me\s+about\s+(?:the\s+)?cases)\b",
                query,
                re.IGNORECASE,
            )
        )
        judge_match = re.search(r"\b(?:justice|judge|hon'ble\s+justice)\s+([A-Za-z\s\.]+)", query, re.IGNORECASE)
        if is_catalog_query and judge_match:
            raw_judge = judge_match.group(1).strip()
            target_judge = re.split(r"\b(how|what|and|list|all|in|the)\b", raw_judge, flags=re.IGNORECASE)[0].strip()
            if len(target_judge) >= 3:
                profile = self.get_judge_profile(target_judge)
                if profile["total_cases"] > 0:
                    logger.info(f"Routed query to Judicial Profile Registry for Justice '{target_judge}' ({profile['total_cases']} cases found).")
                    return self._synthesize_judge_profile_answer(query, profile, total_start_t)

        # Run 5-stage retrieval funnel
        funnel_output = self.retrieve_hybrid(
            query=query,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            filters=filters,
            collection_name=collection_name,
        )

        timings = dict(funnel_output["timings"])
        retrieved_parents = funnel_output["retrieved_parents"]
        child_citations = funnel_output["child_citations"]

        # ---------------------------------------------------------
        # Grounded Legal Generation
        # ---------------------------------------------------------
        llm_start_t = time.perf_counter()

        # Handle empty/insufficient retrieval
        if not retrieved_parents and not child_citations:
            answer = "Insufficient authoritative precedent found in the indexed corpus."
            timings["llm_generation_sec"] = round(time.perf_counter() - llm_start_t, 4)
            timings["total_latency_sec"] = round(time.perf_counter() - total_start_t, 4)
            return {
                "answer": answer,
                "retrieved_parents": [],
                "child_citations": [],
                "timings": timings,
                "query": query,
            }

        # Build context blocks from expanded parents and child role tags
        context_blocks: List[str] = []
        for idx, parent in enumerate(retrieved_parents, start=1):
            p_title = parent.get("case_title", "Precedent")
            p_meta = parent.get("metadata", {})
            p_court = p_meta.get("court", "Court")
            p_year = p_meta.get("year", "")
            p_heading = p_meta.get("structural_heading", "Legal Section")
            p_role = p_meta.get("role") or p_meta.get("legal_role", "Ratio Decidendi")
            p_bench = p_meta.get("bench_type", "Bench")
            p_author = p_meta.get("author_judge", "Judge")
            p_disp = p_meta.get("disposition", "Not Stated")
            p_text = parent.get("text", "")
            context_blocks.append(
                f"### [Precedent {idx}] {p_title} ({p_court}, {p_year})\n"
                f"- **Authority & Bench**: {p_court} | {p_bench} (Author: {p_author})\n"
                f"- **Binding Status**: [{p_role}] | Outcome: `{p_disp}` | Section: {p_heading}\n"
                f"{p_text}"
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""Legal Query:
{query}

Retrieved Judicial Context:
{context_str}

Analyze the legal query based exclusively on the provided judicial context. Provide authoritative findings, distinguishing Ratio Decidendi from Obiter Dicta, with statutory concordance:"""

        answer = get_budget(self.settings).generate(
            client=self.gemini_client,
            models=[self.settings.gemini_model, "gemini-flash-latest"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GROUNDED_LEGAL_SYSTEM_INSTRUCTION,
                temperature=0.1,
            ),
        ) or (
            "Answer synthesis unavailable (no LLM provider reachable: check LLM_API_KEY / GEMINI_API_KEY "
            "quota). The ranked precedents below were retrieved successfully."
        )

        timings["llm_generation_sec"] = round(time.perf_counter() - llm_start_t, 4)
        timings["total_latency_sec"] = round(time.perf_counter() - total_start_t, 4)

        return {
            "answer": answer,
            "retrieved_parents": retrieved_parents,
            "child_citations": child_citations,
            "reranked_candidates": funnel_output["reranked_candidates"],
            "timings": timings,
            "query": query,
        }

    def get_judge_profile(self, judge_query: str) -> Dict[str, Any]:
        """Aggregate career statistics, behavioral metrics, and full case registry for a judge."""
        clean_name = re.sub(
            r"\b(justice|hon'ble|honourable|dr\.|mr\.|mrs\.|ms\.)\b",
            "",
            judge_query,
            flags=re.IGNORECASE,
        ).strip()
        cases = self.sqlite_store.get_cases_by_judge(clean_name)

        total_cases = len(cases)
        dispositions = {"ALLOWED": 0, "DISMISSED": 0, "DISPOSED OF": 0, "PENDING/NOT_STATED": 0}
        categories: Dict[str, int] = {}
        courts: Dict[str, int] = {}
        bench_types: Dict[str, int] = {}

        for c in cases:
            disp = str(c.get("disposition", "PENDING/NOT_STATED")).upper()
            if disp in dispositions:
                dispositions[disp] += 1
            else:
                dispositions["PENDING/NOT_STATED"] += 1

            cat = c.get("case_category", "General")
            categories[cat] = categories.get(cat, 0) + 1

            court = c.get("court", "Court")
            courts[court] = courts.get(court, 0) + 1

            b_type = c.get("bench_type", "Single Bench")
            bench_types[b_type] = bench_types.get(b_type, 0) + 1

        decided = dispositions["ALLOWED"] + dispositions["DISMISSED"]
        allowance_rate = round((dispositions["ALLOWED"] / decided) * 100.0, 1) if decided > 0 else 0.0

        return {
            "judge_query": judge_query,
            "total_cases": total_cases,
            "allowance_rate_pct": allowance_rate,
            "dispositions": dispositions,
            "categories": categories,
            "courts": courts,
            "bench_types": bench_types,
            "cases": cases,
        }

    def _synthesize_judge_profile_answer(
        self, query: str, profile: Dict[str, Any], total_start_t: float
    ) -> Dict[str, Any]:
        """Synthesize an exhaustive case registry and judicial behavioral profile."""
        j_name = profile["judge_query"].strip().title()
        total = profile["total_cases"]
        rate = profile["allowance_rate_pct"]
        disp = profile["dispositions"]
        cases = profile["cases"]

        lines = [
            f"## Judicial Profile & Case Registry: Justice {j_name}\n",
            f"### 1. Empirical Track Record & Behavioral Metrics",
            f"- **Total Indexed Judgments in Corpus**: {total}",
            f"- **Appeal / Petition Allowance Rate**: {rate}% ({disp.get('ALLOWED', 0)} Allowed vs {disp.get('DISMISSED', 0)} Dismissed)",
            f"- **Case Dispositions**: Allowed: `{disp.get('ALLOWED', 0)}` | Dismissed: `{disp.get('DISMISSED', 0)}` | Disposed/Other: `{disp.get('DISPOSED OF', 0) + disp.get('PENDING/NOT_STATED', 0)}`",
            f"- **Primary Benches**: " + ", ".join(f"{k} ({v})" for k, v in profile["bench_types"].items()),
            f"- **Courts Presided**: " + ", ".join(f"{k} ({v})" for k, v in profile["courts"].items()),
            "\n### 2. Complete Case Registry",
            "| # | Case Title | Year | Court | Bench Strength | Disposition | Category |",
            "|---|---|---|---|---|---|---|",
        ]

        for idx, c in enumerate(cases, start=1):
            title = c.get("case_title", "Unknown Case")
            year = c.get("year", "N/A")
            court = c.get("court", "High Court / Supreme Court")
            bench = c.get("bench_type", "Single Bench")
            disposition = c.get("disposition", "Not Stated")
            cat = c.get("case_category", "General")
            lines.append(f"| {idx} | **{title}** | {year} | {court} | {bench} | `{disposition}` | {cat} |")

        answer_text = "\n".join(lines)
        total_latency = round(time.perf_counter() - total_start_t, 4)

        return {
            "answer": answer_text,
            "retrieved_parents": [],
            "child_citations": [],
            "timings": {
                "metadata_slicing_sec": total_latency,
                "total_latency_sec": total_latency,
            },
            "query": query,
            "judicial_profile": profile,
        }

    # Backward compatibility alias
    run = generate_answer
