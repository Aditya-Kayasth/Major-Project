#!/usr/bin/env python3
"""Pre-flight smoke test: verifies every component on THIS machine's library versions
before committing to a long indexing run.

Checks (each prints PASS/FAIL):
  1. Library versions and CUDA
  2. Embedding model loads, query/document prompts are applied, vectors are sane
  3. Cross-encoder reranker loads and ranks a relevant passage above an irrelevant one
  4. ChromaDB create / upsert / get-by-ids / query / delete in a temporary directory
  5. BM25 registry scoring
  6. (optional, --llm) two LLM calls through the provider chain (plain + JSON mode)

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
                traceback.print_exc()
        return run
    return wrap


@check("Library versions & CUDA")
def check_versions():
    import chromadb
    import sentence_transformers
    import torch
    import transformers

    assert torch.cuda.is_available(), "CUDA not available to PyTorch"
    return (
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"sentence-transformers {sentence_transformers.__version__} | chromadb {chromadb.__version__} | "
        f"GPU {torch.cuda.get_device_name(0)}"
    )


@check("GPU attention kernel (scaled_dot_product_attention)")
def check_sdpa():
    import torch
    import torch.nn.functional as F

    from src.resource_guard import configure_torch_backends

    configure_torch_backends()
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception as err:
        cudnn_version = f"unavailable ({err})"
    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    out = F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    assert out.shape == q.shape
    return f"cudnn_enabled={torch.backends.cudnn.enabled} cudnn_version={cudnn_version}"


@check("Embedding model (query vs document prompts)")
def check_embeddings(settings):
    import numpy as np

    from src.local_embeddings import LocalEmbeddingEngine

    engine = LocalEmbeddingEngine(model_name=settings.local_embedding_model, batch_size=8)
    q = engine.embed_texts(["Is anticipatory bail available for offences under Section 438 CrPC?"], is_query=True)[0]
    docs = engine.embed_texts(
        [
            "The court held that anticipatory bail under Section 438 CrPC may be granted where the accused apprehends arrest.",
            "The appellant challenged the land acquisition award on the ground of inadequate compensation.",
        ]
    )
    sims = [float(np.dot(q, d)) for d in docs]
    prompts = list((getattr(engine._st_model, "prompts", None) or {}).keys()) if engine._st_model else []
    assert len(q) > 0 and abs(float(np.linalg.norm(q)) - 1.0) < 1e-2, "query vector not normalised"
    assert sims[0] > sims[1], f"relevant doc not closer to query: {sims}"
    return f"model={engine.model_name} dim={len(q)} prompts={prompts} sim(relevant)={sims[0]:.3f} > sim(irrelevant)={sims[1]:.3f}"


@check("Cross-encoder reranker")
def check_reranker(settings):
    from src.rag_pipeline import IndianLegalRAGPipeline

    class _NoStore:  # avoid touching the real vector store
        pass

    pipe = IndianLegalRAGPipeline.__new__(IndianLegalRAGPipeline)
    pipe.settings = settings
    pipe._reranker = None
    reranker = pipe._get_reranker()
    scores = reranker.predict(
        [
            ["Is anticipatory bail available under Section 438 CrPC?", "Anticipatory bail under Section 438 CrPC may be granted by the Sessions Court or High Court."],
            ["Is anticipatory bail available under Section 438 CrPC?", "The tenant failed to pay rent for three consecutive months."],
        ]
    )
    assert float(scores[0]) > float(scores[1]), f"reranker ordering wrong: {scores}"
    return f"scores={[round(float(s), 3) for s in scores]}"


@check("ChromaDB create/upsert/get/query/delete")
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
            metadatas=[{"year": 2025, "court": "SC"}, {"year": 2024, "court": "HC"}],
        )
        got = col.get(ids=["a", "zzz"], include=[])
        assert got["ids"] == ["a"], f"get by ids returned {got['ids']}"
        res = col.query(query_embeddings=[[0.9, 0.1, 0.0]], n_results=2, include=["documents", "metadatas", "distances"])
        assert res["ids"][0][0] == "a", f"nearest neighbour wrong: {res['ids']}"
        page = col.get(include=["documents", "metadatas"], limit=1, offset=1)
        assert len(page["ids"]) == 1, "pagination failed"
        client.delete_collection("smoke")
    return "ok"


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

    from google import genai
    from google.genai import types

    from src.llm_budget import get_budget

    has_gemini = bool(settings.gemini_api_key) and settings.gemini_api_key != "your_gemini_api_key_here"
    client = genai.Client(api_key=settings.gemini_api_key) if has_gemini else None
    budget = get_budget(settings)
    route = budget.llm_label(client)
    assert route != "none configured", "no LLM configured (set LLM_API_KEY or GEMINI_API_KEY)"
    text = budget.generate(
        client=client, models=[settings.gemini_model],
        contents="In one sentence: what does Section 438 CrPC provide?", use_cache=False,
    )
    assert text, f"no response from any provider in route: {route}"
    raw = budget.generate(
        client=client, models=[settings.gemini_model], use_cache=False,
        contents='Return JSON {"query": "<a legal research question about bail>", "answer": "<one sentence>"}',
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    data = json.loads(raw)
    assert data.get("query") and data.get("answer"), f"JSON mode returned unexpected keys: {list(data)}"
    return f"route={route} | answer={text.strip()[:70]!r}"


def main():
    parser = argparse.ArgumentParser(description="Pre-flight smoke test")
    parser.add_argument("--llm", "--gemini", dest="llm", action="store_true", help="Also make 2 LLM requests through the provider chain")
    args = parser.parse_args()

    from src.config import get_settings

    settings = get_settings()
    print(f"Embedding provider: {settings.embedding_provider} ({settings.local_embedding_model}) | reranker: {settings.reranker_model}\n")

    check_versions()
    check_sdpa()
    check_embeddings(settings)
    check_reranker(settings)
    check_chroma()
    check_bm25()
    if args.llm:
        check_llm(settings)

    try:
        import psutil
        import torch

        rss = psutil.Process().memory_info().rss / 1024 ** 3
        vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        print(f"\nPeak VRAM: {vram:.1f} GB | Process RAM: {rss:.1f} GB")
    except Exception:
        pass

    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + ("ALL CHECKS PASSED - safe to start indexing." if not failed else f"FAILED: {failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
