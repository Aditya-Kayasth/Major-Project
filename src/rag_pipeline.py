"""End-to-end production RAG Pipeline for Indian Legal Q&A and Precedent Search.

Phase 3: Multi-Stage Hybrid Retrieval Funnel (Metadata Slicing + Dense Cosine +
BM25Okapi + Reciprocal Rank Fusion + BGE Cross-Encoder Reranker + Parent Expansion)
and Zero-Hallucination Grounded Legal Synthesis.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import Settings, get_settings
from src.vector_store import VectorStoreManager

# Configure logger
logger = logging.getLogger("IndianLegalRAG.Pipeline")
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
        self._reranker: Optional[Any] = None

    def _get_reranker(self) -> Any:
        """Lazy loader for sentence_transformers.CrossEncoder with hardware-throttled low-power profile."""
        if self._reranker is None:
            import torch
            torch.set_num_threads(2)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                # Hard cap at ~1.8 GB VRAM (0.45 on 4 GB RTX 2050)
                torch.cuda.set_per_process_memory_fraction(0.45, 0)
                logger.info(
                    f"Configured PyTorch CUDA memory fraction cap: 0.45 (~1.8GB) on '{torch.cuda.get_device_name(0)}'"
                )

            logger.info(
                f"Lazy-initializing CrossEncoder ('{self.settings.reranker_model}') on device '{device}' (CPU threads: 2)..."
            )
            from sentence_transformers import CrossEncoder

            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            self._reranker = CrossEncoder(
                self.settings.reranker_model,
                device=device,
                model_kwargs=model_kwargs,
            )
            if device == "cuda":
                logger.info("Enabled FP16 half-precision on CrossEncoder model (VRAM ~1.06 GB).")

        return self._reranker

    def retrieve_parent_context(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the original parent chunk context from local disk parent store."""
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
    ) -> List[Dict[str, Any]]:
        """Perform Reciprocal Rank Fusion (RRF) on dense and sparse candidate rankings.

        RRF Score Formula:
            score = 1.0 / (k + dense_rank) + 1.0 / (k + bm25_rank)
        """
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Dict[str, Any]] = {}

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

        # 3. Build Fused List
        fused_candidates: List[Dict[str, Any]] = []
        for chunk_id, score in rrf_scores.items():
            candidate = candidate_map[chunk_id]
            candidate["rrf_score"] = float(score)
            fused_candidates.append(candidate)

        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_candidates

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 25,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the Five-Stage Retrieval Funnel with granular timing instrumentation.

        Stages:
            1. Metadata Slicing (allowed IDs via ChromaDB)
            2. Candidate Generation (Dense vector + BM25Okapi)
            3. Reciprocal Rank Fusion (RRF k=60, top 15 candidates)
            4. Cross-Encoder Reranking (BGE reranker)
            5. Parent Document Expansion (local disk parent store)
        """
        timings: Dict[str, float] = {}
        funnel_start_t = time.perf_counter()

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
                query=query,
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
            query=query,
            top_k=candidate_pool_size,
            allowed_ids=allowed_ids,
            collection_name=collection_name,
        )
        timings["bm25_retrieval_sec"] = round(time.perf_counter() - s2_bm25_start, 4)

        # ---------------------------------------------------------
        # Stage 3: Reciprocal Rank Fusion
        # ---------------------------------------------------------
        s3_start = time.perf_counter()
        fused_candidates = self.compute_rrf(dense_candidates, bm25_candidates, k=60)
        # Select top 15 fused candidates for cross-encoder reranking
        top_fused = fused_candidates[:15]
        timings["rrf_fusion_sec"] = round(time.perf_counter() - s3_start, 4)

        # ---------------------------------------------------------
        # Stage 4: Cross-Encoder Reranking
        # ---------------------------------------------------------
        s4_start = time.perf_counter()
        reranked_candidates: List[Dict[str, Any]] = []

        if top_fused:
            reranker = self._get_reranker()
            pairs = [[query, c["text"]] for c in top_fused]
            scores = reranker.predict(pairs)

            for c, score in zip(top_fused, scores):
                c_copy = dict(c)
                c_copy["rerank_score"] = float(score)
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
            p_text = parent.get("text", "")
            context_blocks.append(
                f"### [Precedent {idx}] {p_title} ({p_court}, {p_year}) - Section: {p_heading}\n{p_text}"
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""Legal Query:
{query}

Retrieved Judicial Context:
{context_str}

Analyze the legal query based exclusively on the provided judicial context. Provide authoritative findings, distinguishing Ratio Decidendi from Obiter Dicta, with statutory concordance:"""

        if not self.gemini_client:
            answer = "GEMINI_API_KEY is not configured in environment. Retrieval completed successfully."
        else:
            model_candidates = [
                self.settings.gemini_model,
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest",
            ]
            models_to_try = list(dict.fromkeys([m for m in model_candidates if m]))
            answer = ""
            last_err = None

            for model_name in models_to_try:
                for attempt in range(3):
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=GROUNDED_LEGAL_SYSTEM_INSTRUCTION,
                                temperature=0.1,
                            ),
                        )
                        if response and response.text:
                            answer = response.text
                            break
                    except APIError as api_err:
                        last_err = api_err
                        if "503" in str(api_err) or "429" in str(api_err):
                            backoff = 2 ** (attempt + 1)
                            time.sleep(backoff)
                            continue
                        elif "404" in str(api_err):
                            break
                        raise
                    except Exception as err:
                        last_err = err
                        break
                if answer:
                    break

            if not answer:
                answer = (
                    f"Synthesis failed due to upstream API connectivity: {last_err}. "
                    "Precedent contexts were successfully retrieved."
                )

        timings["llm_generation_sec"] = round(time.perf_counter() - llm_start_t, 4)
        timings["total_latency_sec"] = round(time.perf_counter() - total_start_t, 4)

        return {
            "answer": answer,
            "retrieved_parents": retrieved_parents,
            "child_citations": child_citations,
            "timings": timings,
            "query": query,
        }

    # Backward compatibility alias
    run = generate_answer
