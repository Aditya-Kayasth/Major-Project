"""Retrieval funnel and grounded synthesis for SC criminal precedents (Phase 4, minimal).

query -> statutory concordance -> optional filters (SQLite documents -> doc_ids -> Chroma $in +
BM25 allowed_ids) -> dense top-K + BM25 top-K -> Reciprocal Rank Fusion -> cross-encoder rerank
of the top RERANK_TOP_N x bounded bench-strength authority prior -> parent expansion (SQLite)
-> grounded answer with [Precedent N] headers -> citation check (no LLM).

Authority prior (CLAUDE.md D6): Constitution Bench (>=5) > Full (3-4) > Division (2) > Single/unknown,
applied only at the rerank stage as factor = 1 + clamp(AUTHORITY_STRENGTH * ln(multiplier), +/-
AUTHORITY_MAX_BONUS). AUTHORITY_STRENGTH=0 (or use_authority=False) switches it off for the ablation.
The old court-tier x role x judge-affinity weights are gone.
"""

import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.citation_check import REFUSAL_PHRASE, check_citations
from src.config import Settings, get_settings
from src.concordance import StatutoryConcordance
from src.llm_budget import gen_config, get_budget, make_gemini_client
from src.resource_guard import ResourceGuard
from src.sqlite_store import SQLiteParentStore
from src.vector_store import VectorStoreManager, normalize_chroma_filters

logger = logging.getLogger("IndianLegalRAG.Pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Bench strength -> raw authority multiplier (bounded later by _authority_factor)
BENCH_AUTHORITY = ((5, 1.25), (3, 1.15), (2, 1.05), (1, 1.0))
FILTER_KEYS = ("year_from", "year_to", "min_bench")
INSUFFICIENT_ANSWER = "Insufficient precedent in the indexed corpus."

GROUNDED_LEGAL_SYSTEM_INSTRUCTION = """You are a legal research assistant for Indian criminal law. You answer ONLY from the numbered precedents provided to you ([Precedent 1] ... [Precedent N]). Each precedent is one judgment of the Supreme Court of India, with one or two passages from it.

Rules:
1. Cite only as [Precedent N], using the numbers given. Never mention a case, citation, statutory section or judge that does not appear in the provided precedents.
2. Separate the ratio decidendi (the holding on which the case was decided) from obiter dicta (passing remarks), and say which is which. Passages with role "Headnote" are editorial summaries: rely on the judgment text where both are given.
3. A larger bench outweighs a smaller one: a Constitution Bench (5 or more judges) outweighs a Four-Judge or Three-Judge Bench, which outweighs a Division Bench (2 judges), which outweighs a Single Judge. Say so when it matters.
4. If the precedents conflict, say so explicitly, naming the precedents with their dates and bench sizes.
5. IPC and BNS: offences committed before 1 July 2024 are governed by the IPC (and the CrPC / Evidence Act); offences on or after 1 July 2024 by the BNS (and the BNSS / BSA). When a precedent applies an old-code section, mention that the date of the offence decides which code applies.
6. If the precedents do not answer the question, reply exactly "Insufficient precedent in the indexed corpus." and nothing else except rule 7.
7. End every answer with the line: "Verify on the official source."
"""


def reranker_relevances(scores: List[float]) -> List[float]:
    """Cross-encoder outputs as relevances in [0, 1], without a second sigmoid.

    sentence-transformers 6 CrossEncoder.predict() already applies Sigmoid for 1-label models
    (verified on the server: activation_fn=Sigmoid, outputs in [0,1]), so such a batch is used
    as-is. Only a batch of raw logits (any value outside [0,1]) gets one sigmoid, applied to the
    whole batch so the ordering stays consistent.
    """
    vals = [float(s) for s in scores]
    if all(0.0 <= v <= 1.0 for v in vals):
        return vals
    return [1.0 / (1.0 + math.exp(-max(min(v, 20.0), -20.0))) for v in vals]


def bench_authority_multiplier(bench_strength: Any) -> float:
    """CLAUDE.md D6: Constitution (>=5) 1.25 > Full (3-4) 1.15 > Division (2) 1.05 > Single / unknown 1.0."""
    try:
        n = int(bench_strength)
    except (TypeError, ValueError):
        return 1.0
    for minimum, mult in BENCH_AUTHORITY:
        if n >= minimum:
            return mult
    return 1.0


def _meta(c: Dict[str, Any]) -> Dict[str, Any]:
    return c.get("metadata") or {}


def _field(c: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = c.get(key)
    return v if v not in (None, "") else _meta(c).get(key, default)


def bench_label(bench_strength: Any) -> str:
    """Display label from bench_strength (never 'Full Bench')."""
    try:
        n = int(bench_strength)
    except (TypeError, ValueError):
        return "Bench not recorded"
    if n >= 5:
        return f"Constitution Bench ({n} judges)"
    return {1: "Single Judge", 2: "Division Bench (2 judges)", 3: "Three-Judge Bench", 4: "Four-Judge Bench"}.get(
        n, "Bench not recorded")


def precedent_header(n: int, meta: Dict[str, Any]) -> str:
    """[Precedent N] title | citation | date | bench label: judges | author: ... | disposal: ... | role: ..."""
    def val(key: str, default: str = "not recorded") -> str:
        v = meta.get(key)
        return str(v).strip() if v not in (None, "") else default

    date = val("decision_date", str(meta.get("year") or "date not recorded"))
    return (f"[Precedent {n}] {val('case_title', 'untitled')} | {val('citation')} | {date} | "
            f"{bench_label(meta.get('bench_strength'))}: {val('judges', 'judges not recorded')} | "
            f"author: {val('author_judge')} | disposal: {val('disposal_nature')} | role: {val('role', 'Unclassified')}")


def judgment_key(c: Dict[str, Any]) -> str:
    """Group key for judgment-level results: doc_id, else source_path, else parent_id, else chunk_id."""
    return str(_field(c, "doc_id") or _field(c, "source_path") or _field(c, "parent_id") or c.get("chunk_id"))


def dedupe_by_judgment(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """First (best) hit per judgment, order preserved: turns any chunk ranking into a judgment ranking."""
    seen: Set[str] = set()
    out = []
    for h in hits:
        k = judgment_key(h)
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def passage_window(parent_text: str, child_raw: str, size: int = 1500) -> str:
    """~size chars of the parent around the matched child passage (keeps the prompt small)."""
    if not parent_text:
        return child_raw or ""
    if len(parent_text) <= size:
        return parent_text
    pos = parent_text.find((child_raw or "")[:120]) if child_raw else -1
    start = max(0, (pos if pos >= 0 else 0) - 300)
    return parent_text[start:start + size].strip()


class IndianLegalRAGPipeline:
    """Hybrid retrieval funnel + grounded synthesis over one Chroma collection and its SQLite store."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[VectorStoreManager] = None,
        gemini_client: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStoreManager(settings=self.settings, collection_name=collection_name)
        self.collection_name = collection_name or self.vector_store.default_collection_name
        self.gemini_client = gemini_client or make_gemini_client(self.settings.gemini_api_key)
        # One SQLite file per collection (pilot and full index never share parents)
        self.sqlite_store = SQLiteParentStore(self.settings.sqlite_path_for(self.collection_name))
        self._stores: Dict[str, SQLiteParentStore] = {self.collection_name: self.sqlite_store}
        self.guard = ResourceGuard(
            max_ram_gb=self.settings.max_ram_gb,
            warn_ram_gb=self.settings.warn_ram_gb,
            max_storage_gb=self.settings.max_storage_gb,
            num_cpu_threads=self.settings.num_threads,
        )
        self._reranker: Optional[Any] = None

    # ------------------------------------------------------------------ helpers
    def _store(self, collection_name: Optional[str]) -> SQLiteParentStore:
        name = collection_name or self.collection_name
        if name not in self._stores:
            self._stores[name] = SQLiteParentStore(self.settings.sqlite_path_for(name))
        return self._stores[name]

    def _get_reranker(self) -> Any:
        """Lazy CrossEncoder (auto device; fp32 on CPU)."""
        if self._reranker is None:
            import torch

            from src.resource_guard import configure_torch_backends

            configure_torch_backends()
            torch.set_num_threads(self.settings.num_threads)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                try:
                    torch.cuda.set_per_process_memory_fraction(self.settings.vram_safety_fraction, 0)
                except Exception as e:
                    logger.warning(f"Could not set VRAM memory fraction: {e}")
            from sentence_transformers import CrossEncoder

            # transformers 5 / sentence-transformers 6: `dtype` (was `torch_dtype`); fp32 on CPU
            model_kwargs = {"dtype": torch.float16} if device == "cuda" else {}
            logger.info(f"Loading CrossEncoder '{self.settings.reranker_model}' on {device} "
                        f"(threads {self.settings.num_threads})")
            self._reranker = CrossEncoder(self.settings.reranker_model, device=device, model_kwargs=model_kwargs)
        return self._reranker

    def _authority_factor(self, authority_mult: float, use_authority: Optional[bool] = None) -> float:
        """factor = 1 + clamp(strength * ln(mult), -max_bonus, +max_bonus): a bounded tie-breaker (D6)."""
        strength = self.settings.authority_strength
        if use_authority is False or strength <= 0 or authority_mult <= 0:
            return 1.0
        max_bonus = self.settings.authority_max_bonus
        return 1.0 + max(-max_bonus, min(max_bonus, strength * math.log(authority_mult)))

    def retrieve_parent_context(self, parent_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parent section from the collection's SQLite store (legacy JSON parent files as a fallback)."""
        try:
            parent = self._store(collection_name).get_parent(parent_id)
            if parent:
                return parent
        except Exception as err:
            logger.debug(f"SQLite lookup failed ({err})")
        parent_file = self.settings.parent_store_dir / f"{parent_id}.json"
        if parent_file.exists():
            try:
                return json.loads(parent_file.read_text(encoding="utf-8"))
            except Exception as err:
                logger.warning(f"Error reading parent document {parent_file}: {err}")
        return None

    def resolve_filters(self, filters: Optional[Dict[str, Any]], collection_name: Optional[str] = None
                        ) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], Dict[str, Any]]:
        """Filters -> (Chroma where, BM25 allowed chunk ids, info).

        year_from / year_to / min_bench go through the SQLite documents table -> doc_ids -> Chroma
        {"doc_id": {"$in": [...]}}; any other key is an exact-match Chroma metadata filter.
        info["empty"] is True when nothing can match.
        """
        info: Dict[str, Any] = {"applied": {}, "matching_docs": None, "empty": False}
        if not filters:
            return None, None, info
        f = {k: v for k, v in filters.items() if v not in (None, "")}
        doc_filters = {k: f.pop(k) for k in list(f) if k in FILTER_KEYS}
        info["applied"] = {**doc_filters, **f}
        clauses: List[Dict[str, Any]] = []
        if doc_filters:
            doc_ids = self._store(collection_name).doc_ids_where(**doc_filters)
            info["matching_docs"] = len(doc_ids)
            if not doc_ids:
                info["empty"] = True
                return None, [], info
            clauses.append({"doc_id": {"$in": doc_ids}})
        if f:
            clauses.append(normalize_chroma_filters(f))
        if not clauses:
            return None, None, info
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        allowed = self.vector_store.filter_ids_by_metadata(where, collection_name=collection_name)
        info["empty"] = not allowed
        return where, allowed, info

    # ------------------------------------------------------------------ fusion
    def compute_rrf(self, dense_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], k: int = 60,
                    query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Plain Reciprocal Rank Fusion: score = sum 1/(k + rank). The authority prior is NOT applied here."""
        scores: Dict[str, float] = {}
        cands: Dict[str, Dict[str, Any]] = {}
        for label, results in (("dense_rank", dense_results), ("bm25_rank", bm25_results)):
            for rank, item in enumerate(results, start=1):
                cid = item["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                cands.setdefault(cid, dict(item))[label] = rank
        fused = []
        for cid, score in scores.items():
            c = cands[cid]
            c["raw_rrf_score"] = float(score)
            c["rrf_score"] = float(score)
            c["bench_strength"] = _field(c, "bench_strength")
            c["authority_multiplier"] = bench_authority_multiplier(c["bench_strength"])
            fused.append(c)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    # ------------------------------------------------------------------ retrieval
    def retrieve_hybrid(self, query: str, top_k: int = 5, candidate_pool_size: int = 60,
                        filters: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None,
                        use_authority: Optional[bool] = None) -> Dict[str, Any]:
        """Concordance -> filters -> dense + BM25 -> RRF -> rerank (x bounded bench prior) -> parents."""
        timings: Dict[str, float] = {}
        t_all = time.perf_counter()
        search_query = StatutoryConcordance.expand_query(query)

        t = time.perf_counter()
        where, allowed_ids, finfo = self.resolve_filters(filters, collection_name)
        timings["metadata_filtering_sec"] = round(time.perf_counter() - t, 4)
        empty = {"judgments": [], "retrieved_parents": [], "child_citations": [], "reranked_candidates": [], "expanded_query": search_query,
                 "filters": finfo, "funnel_counts": {"dense": 0, "bm25": 0, "fused": 0, "reranked": 0, "returned": 0, "parents": 0}}
        if finfo["empty"]:
            timings["funnel_latency_sec"] = round(time.perf_counter() - t_all, 4)
            return {**empty, "timings": timings}

        t = time.perf_counter()
        try:
            dense = self.vector_store.query_dense(query=search_query, top_k=candidate_pool_size, filters=where,
                                                  collection_name=collection_name)
        except Exception as err:
            logger.warning(f"Dense retrieval failed ({err}); continuing with BM25 only.")
            dense = []
        timings["dense_retrieval_sec"] = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        bm25 = self.vector_store.query_bm25(query=search_query, top_k=candidate_pool_size, allowed_ids=allowed_ids,
                                            collection_name=collection_name)
        timings["bm25_retrieval_sec"] = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        fused = self.compute_rrf(dense, bm25, k=60, query=query)
        top_fused = fused[: self.settings.rerank_top_n]
        timings["rrf_fusion_sec"] = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        scored: List[Dict[str, Any]] = []
        if top_fused:
            raw = self._get_reranker().predict([[query, c["text"]] for c in top_fused])
            for c, raw_s, rel in zip(top_fused, raw, reranker_relevances(raw)):
                factor = self._authority_factor(c["authority_multiplier"], use_authority)
                scored.append({**c, "raw_cross_encoder_score": float(raw_s), "relevance": rel,
                               "authority_factor": factor, "rerank_score": round(rel * factor, 6)})
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Judgment-level ranking: score = best chunk; up to PASSAGES_PER_JUDGMENT best passages each
        groups: Dict[str, Dict[str, Any]] = {}
        for c in scored:
            key = judgment_key(c)
            g = groups.get(key)
            if g is None:
                groups[key] = {"doc_id": _field(c, "doc_id") or key, "best": c, "score": c["rerank_score"],
                               "relevance": c["relevance"], "authority_factor": c["authority_factor"], "passages": [c]}
            elif len(g["passages"]) < self.settings.passages_per_judgment:
                g["passages"].append(c)
        judgments = sorted(groups.values(), key=lambda g: g["score"], reverse=True)[:top_k]
        for rank, g in enumerate(judgments, start=1):
            g["rank"] = rank
        reranked = [g["best"] for g in judgments]  # one (best) chunk per judgment, in judgment order
        timings["reranking_sec"] = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        parents: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        for g in judgments:
            seen: Set[str] = set()
            g["parents"] = []
            for c in g["passages"]:
                parent_id = _field(c, "parent_id")
                citations.append({
                    "chunk_id": c["chunk_id"], "parent_id": parent_id, "doc_id": g["doc_id"], "judgment_rank": g["rank"],
                    "case_title": _field(c, "case_title", "Unknown case"), "citation": _field(c, "citation"),
                    "court": _field(c, "court", ""), "year": _field(c, "year", ""), "role": _field(c, "role", "Unclassified"),
                    "snippet": (_field(c, "raw_text") or c.get("text", ""))[:300], "rerank_score": c.get("rerank_score", 0.0),
                })
                if parent_id and parent_id not in seen:
                    seen.add(parent_id)
                    parent = self.retrieve_parent_context(parent_id, collection_name)
                    if parent:
                        g["parents"].append(parent)
                        parents.append(parent)
        timings["parent_expansion_sec"] = round(time.perf_counter() - t, 4)
        timings["funnel_latency_sec"] = round(time.perf_counter() - t_all, 4)
        return {
            "judgments": judgments,
            "retrieved_parents": parents,
            "child_citations": citations,
            "reranked_candidates": reranked,
            "timings": timings,
            "expanded_query": search_query,
            "filters": finfo,
            "funnel_counts": {"dense": len(dense), "bm25": len(bm25), "fused": len(fused), "reranked": len(top_fused),
                              "judgments": len(groups), "returned": len(judgments), "parents": len(parents)},
        }

    # ------------------------------------------------------------------ synthesis
    @staticmethod
    def judgment_meta(g: Dict[str, Any]) -> Dict[str, Any]:
        """Doc-level metadata of a judgment (parent metadata if loaded, else the best chunk's)."""
        meta = dict(_meta(g["best"]))
        if g.get("parents"):
            pm = g["parents"][0]
            meta.update({k: v for k, v in (pm.get("metadata") or {}).items() if v not in (None, "")})
            meta["case_title"] = pm.get("case_title") or meta.get("case_title")
        roles = list(dict.fromkeys(str(_field(c, "role", "Unclassified")) for c in g["passages"]))
        meta["role"] = ", ".join(roles)
        return meta

    def build_context(self, judgments: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """ONE [Precedent N] block per judgment with its passages under it, + the precedent list."""
        blocks, precedents = [], []
        parent_by_id = {p.get("parent_id"): p for g in judgments for p in g.get("parents", [])}
        for n, g in enumerate(judgments, start=1):
            meta = self.judgment_meta(g)
            lines = [precedent_header(n, meta)]
            used: Set[str] = set()
            for i, c in enumerate(g["passages"], start=1):
                pid = _field(c, "parent_id")
                raw = _field(c, "raw_text") or c.get("text", "")
                text = passage_window(parent_by_id[pid]["text"], raw) if (pid in parent_by_id and pid not in used) else raw
                used.add(pid)
                lines.append(f"Passage {i} (role: {_field(c, 'role', 'Unclassified')}):\n{text}")
            blocks.append("\n".join(lines))
            precedents.append({
                "n": n, "doc_id": g["doc_id"], "case_title": meta.get("case_title"), "citation": meta.get("citation"),
                "case_id": meta.get("case_id"), "decision_date": meta.get("decision_date"),
                "bench_strength": meta.get("bench_strength"), "bench_label": bench_label(meta.get("bench_strength")),
                "judges": meta.get("judges"), "author_judge": meta.get("author_judge"),
                "disposal_nature": meta.get("disposal_nature"), "roles": meta.get("role"),
                "n_passages": len(g["passages"]), "score": g["score"], "relevance": g["relevance"],
                "header": precedent_header(n, meta),
            })
        return "\n\n".join(blocks), precedents

    def generate_answer(self, query: str, top_k: int = 5, candidate_pool_size: int = 60,
                        filters: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None,
                        use_authority: Optional[bool] = None) -> Dict[str, Any]:
        """Funnel + grounded answer + citation check.

        No LLM call (answer = exactly INSUFFICIENT_ANSWER) when nothing was retrieved, or when the best
        judgment's relevance is below MIN_RELEVANCE (low_confidence=True; the ranked list is still returned).
        """
        t_all = time.perf_counter()
        funnel = self.retrieve_hybrid(query=query, top_k=top_k, candidate_pool_size=candidate_pool_size,
                                      filters=filters, collection_name=collection_name, use_authority=use_authority)
        timings = dict(funnel["timings"])
        judgments = funnel["judgments"]
        parents = funnel["retrieved_parents"]
        best_relevance = max(g["relevance"] for g in judgments) if judgments else None
        low_confidence = bool(judgments) and best_relevance < self.settings.min_relevance

        t = time.perf_counter()
        if not judgments or low_confidence:
            answer, precedents = INSUFFICIENT_ANSWER, []
        else:
            context, precedents = self.build_context(judgments)
            prompt = (f"Legal question:\n{query}\n\nPrecedents (the only sources you may use):\n\n{context}\n\n"
                      f"Answer the question following the rules. Cite as [Precedent N].")
            answer = get_budget(self.settings).generate(
                client=self.gemini_client,
                models=[self.settings.gemini_model, "gemini-flash-latest"],
                contents=prompt,
                config=gen_config(system_instruction=GROUNDED_LEGAL_SYSTEM_INSTRUCTION, temperature=0.1),
            ) or ("Answer synthesis unavailable (no LLM provider reachable: check LLM_API_KEY / GEMINI_API_KEY). "
                  "The ranked precedents were retrieved successfully.")
        timings["llm_generation_sec"] = round(time.perf_counter() - t, 4)
        timings["total_latency_sec"] = round(time.perf_counter() - t_all, 4)
        return {
            "answer": answer,
            "low_confidence": low_confidence,
            "best_relevance": best_relevance,
            "min_relevance": self.settings.min_relevance,
            "precedents": precedents,
            "judgments": [{"rank": g["rank"], "doc_id": g["doc_id"], "score": g["score"], "relevance": g["relevance"],
                           "case_title": self.judgment_meta(g).get("case_title"),
                           "citation": self.judgment_meta(g).get("citation"), "n_passages": len(g["passages"])}
                          for g in judgments],
            "citation_check": check_citations(answer, precedents),
            "retrieved_parents": parents,
            "child_citations": funnel["child_citations"],
            "reranked_candidates": funnel["reranked_candidates"],
            "expanded_query": funnel["expanded_query"],
            "filters": funnel["filters"],
            "timings": timings,
            "query": query,
        }

    run = generate_answer
