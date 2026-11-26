#!/usr/bin/env python3
"""Pre-flight smoke test on THIS machine (CPU-only plan; CUDA is reported, never required).

Checks (each prints PASS/FAIL, one line):
  1. Library versions and device
  2. Effective settings vs CLAUDE.md targets (OK / DIFFERS; values are not printed)
  3. GPU attention kernel (skipped without CUDA)
  4. Embedding model: BGE query prefix applied, relevant > irrelevant, throughput (chunks/sec)
  5. Cross-encoder reranker: ordering, latency per 20 pairs, output range (Phase 4 double-sigmoid check)
  6. ChromaDB: create/upsert/get/query/delete, and that the collection really uses cosine distance
  7. BM25 registry scoring
  8. (optional, --llm) two LLM calls through the provider chain

Usage:
    python scripts/smoke_test.py            # no API calls
    python scripts/smoke_test.py --llm      # + 2 LLM requests
"""

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []

# CLAUDE.md targets (§3, D4, D5)
TARGETS = {
    "max_ram_gb": 18.0,
    "warn_ram_gb": 15.0,
    "num_threads": 14,
    "parent_chunk_size": 2000,
    "child_chunk_size": 800,
    "chunk_overlap": 120,
    "embed_batch_size": 64,
    "local_embedding_model": "BAAI/bge-small-en-v1.5",
    "reranker_model": "BAAI/bge-reranker-base",
    "collection_name": "sc_criminal_cpu",
}


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True))
                print(f"[PASS] {name} ({time.perf_counter() - t0:.1f}s) {detail or ''}")
            except Exception as err:
                RESULTS.append((name, False))
                print(f"[FAIL] {name}: {err}")
                traceback.print_exc(limit=3)
        return run
    return wrap


@check("Library versions & device")
def check_versions():
    import chromadb
    import pypdf
    import sentence_transformers
    import torch
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"sentence-transformers {sentence_transformers.__version__} | chromadb {chromadb.__version__} | "
        f"pypdf {pypdf.__version__} | device={device} | torch threads={torch.get_num_threads()}"
    )


@check("Settings vs CLAUDE.md targets")
def check_settings(settings):
    diffs = [k for k, v in TARGETS.items() if getattr(settings, k) != v]
    if diffs:
        print("       DIFFERS (check .env): " + ", ".join(diffs))
    return "all match" if not diffs else f"{len(diffs)} differ (listed above; not a failure)"


@check("GPU attention kernel")
def check_sdpa():
    import torch

    if not torch.cuda.is_available():
        return "skipped (no CUDA; CPU plan)"
    import torch.nn.functional as F

    from src.resource_guard import configure_torch_backends

    configure_torch_backends()
    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    out = F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    assert out.shape == q.shape
    return f"cudnn_enabled={torch.backends.cudnn.enabled}"


@check("Embedding model (query prefix, relevance, throughput)")
def check_embeddings(settings):
    import numpy as np

    from src.local_embeddings import LocalEmbeddingEngine

    engine = LocalEmbeddingEngine(
        model_name=settings.local_embedding_model,
        batch_size=settings.embed_batch_size,
        num_threads=settings.num_threads,
        query_instruction=settings.query_instruction,
    )
    query = "Is anticipatory bail available for offences under Section 438 CrPC?"
    q = engine.embed_texts([query], is_query=True)[0]
    docs = engine.embed_texts([
        "The court held that anticipatory bail under Section 438 CrPC may be granted where the accused apprehends arrest.",
        "The appellant challenged the land acquisition award on the ground of inadequate compensation.",
    ])
    sims = [float(np.dot(q, d)) for d in docs]
    q_prompt, d_prompt = engine.effective_prompt(True), engine.effective_prompt(False)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-2, "query vector not normalised"
    assert sims[0] > sims[1], f"relevant doc not closer to query: {sims}"
    is_bge_en = "bge" in engine.model_name.lower() and "-en" in engine.model_name.lower()
    if is_bge_en:
        assert q_prompt.startswith("Represent this sentence"), f"BGE query prompt missing: {q_prompt!r}"
        assert d_prompt == "", f"documents must not get a prefix, got {d_prompt!r}"
    # Behavioural check: the query vector must differ from the same text encoded with no prompt
    q_plain = engine._st_model.encode([query], prompt="", normalize_embeddings=True, convert_to_numpy=True)[0]
    cos_plain = float(np.dot(q, q_plain))
    if q_prompt:
        assert cos_plain < 0.9999, "query prompt has no effect on the vector (not applied)"
    print(f"       query prompt source={engine.prompt_source(True)} prompt={q_prompt[:58]!r} | "
          f"doc prompt={d_prompt!r} | cos(with, without prompt)={cos_plain:.4f}")
    prefix_applied = bool(q_prompt) and cos_plain < 0.9999

    chunk = ("The prosecution relied on the dying declaration recorded by the Magistrate and the testimony "
             "of the eye-witnesses to establish the guilt of the accused under Section 302 IPC. ") * 5
    texts = [chunk[:800]] * 256
    t0 = time.perf_counter()
    engine.embed_texts(texts)
    rate = len(texts) / (time.perf_counter() - t0)
    return (f"model={engine.model_name} device={engine.device} dim={len(q)} prefix_applied={prefix_applied} "
            f"sim {sims[0]:.3f} > {sims[1]:.3f} | {rate:.0f} chunks/sec (800 chars, batch {engine.batch_size}; "
            f"CLAUDE.md: 170)")


@check("Cross-encoder reranker (ordering, latency, output range)")
def check_reranker(settings):
    from src.rag_pipeline import IndianLegalRAGPipeline

    pipe = IndianLegalRAGPipeline.__new__(IndianLegalRAGPipeline)  # avoid touching the real stores
    pipe.settings = settings
    pipe._reranker = None
    reranker = pipe._get_reranker()
    q = "Is anticipatory bail available under Section 438 CrPC?"
    scores = reranker.predict([
        [q, "Anticipatory bail under Section 438 CrPC may be granted by the Sessions Court or High Court."],
        [q, "The tenant failed to pay rent for three consecutive months."],
    ])
    assert float(scores[0]) > float(scores[1]), f"reranker ordering wrong: {scores}"
    passage = ("The High Court granted anticipatory bail subject to conditions, observing that personal liberty "
               "under Article 21 must be balanced against the needs of the investigation. ") * 4
    pairs = [[q, passage[:800]] for _ in range(20)]
    reranker.predict(pairs)  # warm-up
    t0 = time.perf_counter()
    out = reranker.predict(pairs)
    per20 = time.perf_counter() - t0
    lo, hi = float(min(out)), float(max(out))
    in_unit = 0.0 <= min(float(s) for s in scores) and max(float(s) for s in scores) <= 1.0
    activation = type(getattr(reranker, "activation_fn", None)).__name__
    return (f"model={settings.reranker_model} scores={[round(float(s), 3) for s in scores]} | "
            f"{per20:.2f}s per 20 pairs (CLAUDE.md: 0.42) | predict() in [0,1]: {in_unit} "
            f"(activation_fn={activation}) -> rag_pipeline.reranker_relevances uses in-range predict() output as-is (no second sigmoid)")


@check("ChromaDB create/upsert/get/query/delete + cosine space")
def check_chroma():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = chromadb.PersistentClient(path=tmp, settings=ChromaSettings(anonymized_telemetry=False))
        col = client.get_or_create_collection(name="smoke", metadata={"hnsw:space": "cosine"})
        col.upsert(
            ids=["a", "b"],
            documents=["section 302 murder", "land acquisition"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[{"year": 2025, "doc_id": "x"}, {"year": 2024, "doc_id": "y"}],
        )
        got = col.get(ids=["a", "zzz"], include=[])
        assert got["ids"] == ["a"], f"get by ids returned {got['ids']}"
        res = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["distances"])
        assert res["ids"][0][0] == "a", f"nearest neighbour wrong: {res['ids']}"
        d_orth = float(res["distances"][0][1])  # cosine: 1.0 for orthogonal unit vectors; squared L2: 2.0
        space = "cosine" if abs(d_orth - 1.0) < 1e-3 else ("l2" if abs(d_orth - 2.0) < 1e-3 else f"? ({d_orth})")
        assert space == "cosine", f"collection distance space is {space}, not cosine"
        filt = col.get(where={"doc_id": {"$in": ["y"]}}, include=[])
        assert filt["ids"] == ["b"], f"$in filter returned {filt['ids']}"
        page = col.get(include=["documents", "metadatas"], limit=1, offset=1)
        assert len(page["ids"]) == 1, "pagination failed"
        client.delete_collection("smoke")
    return f"space={space} | $in filter ok"


@check("BM25 registry")
def check_bm25():
    from src.vector_store import BM25Registry

    reg = BM25Registry()
    reg.add_documents(
        ids=["x", "y", "z"],
        texts=["Section 138 Negotiable Instruments Act cheque dishonour", "Article 21 right to life", "Order XLI Rule 1 CPC appeal"],
        metadatas=[{}, {}, {}],
    )
    top = reg.query("cheque dishonour Section 138", top_k=1)
    assert top and top[0]["chunk_id"] == "x", f"unexpected BM25 result {top}"
    return "ok"


@check("LLM provider chain (answer + JSON mode)")
def check_llm(settings):
    import json

    from src.llm_budget import gen_config, get_budget, make_gemini_client

    client = make_gemini_client(settings.gemini_api_key)
    budget = get_budget(settings)
    route = budget.llm_label(client)
    assert route != "none configured", "no LLM configured (set LLM_API_KEY or GEMINI_API_KEY)"
    text = budget.generate(client=client, models=[settings.gemini_model],
                           contents="In one sentence: what does Section 438 CrPC provide?", use_cache=False)
    assert text, f"no response from any provider in route: {route}"
    raw = budget.generate(
        client=client, models=[settings.gemini_model], use_cache=False,
        contents='Return JSON {"query": "<a legal research question about bail>", "answer": "<one sentence>"}',
        config=gen_config(response_mime_type="application/json", temperature=0.2),
    )
    data = json.loads(raw)
    assert data.get("query") and data.get("answer"), f"JSON mode returned unexpected keys: {list(data)}"
    return f"route={route}"


def main():
    parser = argparse.ArgumentParser(description="Pre-flight smoke test (CPU)")
    parser.add_argument("--llm", "--gemini", dest="llm", action="store_true", help="Also make 2 LLM requests")
    args = parser.parse_args()

    import torch

    from src.config import get_settings
    from src.resource_guard import ResourceGuard

    settings = get_settings()
    torch.set_num_threads(settings.num_threads)

    check_versions()
    check_settings(settings)
    check_sdpa()
    check_embeddings(settings)
    check_reranker(settings)
    check_chroma()
    check_bm25()
    if args.llm:
        check_llm(settings)

    tree_gb = ResourceGuard.process_tree_rss_bytes() / 1024 ** 3
    print(f"\nProcess-tree RAM: {tree_gb:.2f} GB (limit {settings.max_ram_gb:.0f} GB)")
    failed = [n for n, ok in RESULTS if not ok]
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
