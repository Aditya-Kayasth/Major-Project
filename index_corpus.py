#!/usr/bin/env python3
"""Streamed, resumable indexing of the SC criminal corpus from the manifest (Phase 3).

Inputs   : $METADATA_DIR/manifest.parquet rows with a downloaded PDF (status ok/exists).
Workers  : a ProcessPoolExecutor with the *spawn* start method (EXTRACT_WORKERS, default 4).
           Workers import only src/judgment_text.py + pypdf (no torch); they extract, clean,
           split the headnote and chunk each PDF, returning plain dicts.
Main     : SQLite parents + documents (one <collection>.sqlite per collection), embedding
           (bge-small, batch EMBED_BATCH_SIZE, torch threads INDEX_TORCH_THREADS), Chroma upsert,
           the process-tree RAM guard before every micro-batch, and one BM25 rebuild at the end.
Resume   : checkpoint_<collection>.json keyed by doc_id. Pilot and full collections never share
           a checkpoint or a SQLite file; --reset only touches the named collection.

Usage (server):
    python index_corpus.py --collection sc_criminal_cpu_test --max-docs 20 --reset
    nohup python index_corpus.py --collection sc_criminal_cpu > ~/Pratik/workspace/logs/index_full.log 2>&1 &

IMPORTANT: this module's top level must stay free of torch / chromadb / pydantic imports:
with the spawn start method every worker re-imports it (as __mp_main__).
"""

import argparse
import datetime
import json
import logging
import math
import multiprocessing as mp
import statistics
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import judgment_text as jt

logger = logging.getLogger("IndianLegalRAG.Indexer")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def setup_file_logging(logs_dir: Path, collection: str) -> Path:
    """Detailed log in LOGS_DIR/index_<collection>.log (never inside the git folder). stdout gets the summary."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"index_{collection}.log"
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return log_file


class CheckpointManager:
    """Per-collection indexing state keyed by doc_id, saved atomically after every micro-batch."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {"indexed": {}, "failed": {}, "last_updated": None}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded.get("indexed"), dict):
                    self.data = loaded
            except Exception as err:
                logger.warning(f"[Checkpoint] unreadable ({err}); starting fresh.")

    def reset(self) -> None:
        self.data = {"indexed": {}, "failed": {}, "last_updated": None}
        self.save()

    def save(self) -> None:
        self.data["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def is_indexed(self, doc_id: str) -> bool:
        return doc_id in self.data["indexed"]

    def mark_indexed(self, doc_id: str, n_chunks: int, n_parents: int) -> None:
        self.data["indexed"][doc_id] = {"chunks": n_chunks, "parents": n_parents}
        self.data["failed"].pop(doc_id, None)

    def mark_failed(self, doc_id: str, reason: str) -> None:
        self.data["failed"][doc_id] = reason


def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Manifest row -> plain picklable dict (NaN -> None, numpy scalars -> Python)."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                v = v.item()
            except Exception:
                pass
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    return out


def select_rows(rows: List[Dict[str, Any]], max_docs: Optional[int]) -> List[Dict[str, Any]]:
    """Available rows sorted by (year, path). With max_docs: an even spread across the sorted list,
    so a 20-PDF test covers every year and all three PDF layouts."""
    avail = sorted((r for r in rows if r.get("status") in ("ok", "exists")),
                   key=lambda r: (int(r.get("year") or 0), str(r.get("path"))))
    if not max_docs or max_docs >= len(avail):
        return avail
    if max_docs <= 0:
        return []
    step = len(avail) / max_docs
    return [avail[int(i * step)] for i in range(max_docs)]


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index the SC criminal corpus from the manifest")
    p.add_argument("--collection", default=None, help="Chroma collection (default COLLECTION_NAME)")
    p.add_argument("--max-docs", type=int, default=None, help="Index an even spread of N docs (tests / pilot)")
    p.add_argument("--reset", action="store_true", help="Wipe THIS collection (Chroma, BM25, SQLite, checkpoint)")
    p.add_argument("--doc-batch-size", type=int, default=16, help="Documents per embed/commit micro-batch")
    p.add_argument("--workers", type=int, default=None, help="Extraction workers (default EXTRACT_WORKERS)")
    p.add_argument("--torch-threads", type=int, default=None, help="torch threads (default INDEX_TORCH_THREADS)")
    p.add_argument("--manifest", default=None, help="Default: $METADATA_DIR/manifest.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_arguments()

    # Heavy imports only here (never at module level: spawned workers re-import this file)
    import pandas as pd

    from src.config import get_settings
    from src.resource_guard import ResourceGuard, ResourceGuardrailException

    settings = get_settings()
    settings.validate(settings)
    collection = args.collection or settings.collection_name
    workers = args.workers or settings.extract_workers
    torch_threads = args.torch_threads or settings.index_torch_threads
    log_file = setup_file_logging(settings.logs_dir, collection)
    t_start = time.perf_counter()

    manifest_path = Path(args.manifest) if args.manifest else settings.manifest_path
    if not manifest_path.exists():
        sys.exit("[ERROR] manifest.parquet not found (run scripts/download_aws_sc.py first).")
    all_rows = [clean_row(r) for r in pd.read_parquet(manifest_path).to_dict("records")]
    rows = [r for r in select_rows(all_rows, args.max_docs) if Path(str(r["local_path"])).exists()]

    checkpoint = CheckpointManager(settings.checkpoint_path_for(collection))
    if args.reset:
        checkpoint.reset()
    pending = [r for r in rows if not checkpoint.is_indexed(r["doc_id"])]
    logger.info(f"collection={collection} selected={len(rows)} pending={len(pending)} workers={workers} "
                f"torch_threads={torch_threads}")

    # Pool first (spawn; workers never see torch), then the models in the main process
    pool = ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn"))

    import torch

    from src.sqlite_store import SQLiteParentStore
    from src.vector_store import VectorStoreManager

    torch.set_num_threads(torch_threads)
    guard = ResourceGuard(max_ram_gb=settings.max_ram_gb, warn_ram_gb=settings.warn_ram_gb,
                          max_storage_gb=settings.max_storage_gb, storage_check_paths=[settings.chroma_db_dir],
                          vram_safety_fraction=settings.vram_safety_fraction, num_cpu_threads=torch_threads)
    guard.initialize_hardware()
    vstore = VectorStoreManager(settings=settings, collection_name=collection, embed_threads=torch_threads)
    sqlite_store = SQLiteParentStore(settings.sqlite_path_for(collection))
    if args.reset:
        vstore.get_or_create_collection(collection, overwrite=True)
        sqlite_store.clear()

    agg: Dict[str, Counter] = {k: Counter() for k in
                               ("layout", "split", "roles", "kind", "removed", "author", "bench", "statutes", "reportable")}
    chunks_per_doc: List[int] = []
    author_agree = Counter()
    disp_agree = Counter()
    disp_found = 0
    docs_with_statute = 0
    failures: List[str] = []
    n_indexed = n_chunks = n_parents = 0
    embed_sec = 0.0
    halted = None

    batch: List[Dict[str, Any]] = []

    def commit(batch_results: List[Dict[str, Any]]) -> None:
        nonlocal n_indexed, n_chunks, n_parents, embed_sec
        if not batch_results:
            return
        guard.enforce_guardrails("Index.commit")
        children = [c for r in batch_results for c in r["children"]]
        sqlite_store.insert_parents_batch([p for r in batch_results for p in r["parents"]])
        sqlite_store.upsert_documents([r["doc"] for r in batch_results])
        t0 = time.perf_counter()
        vstore.index_documents(children, collection_name=collection, batch_size=256, rebuild_bm25=False)
        embed_sec += time.perf_counter() - t0
        for r in batch_results:
            checkpoint.mark_indexed(r["doc_id"], len(r["children"]), len(r["parents"]))
        checkpoint.save()
        n_indexed += len(batch_results)
        n_chunks += len(children)
        n_parents += sum(len(r["parents"]) for r in batch_results)
        logger.info(f"[batch] +{len(batch_results)} docs (+{len(children)} chunks) | total {n_indexed} docs | "
                    f"{guard.format_telemetry()}")
        if n_indexed % 100 < len(batch_results):  # coarse progress on stdout as well
            print(f"[progress] {n_indexed}/{len(pending)} docs | {n_chunks} chunks | "
                  f"peak RAM {guard.peak_ram_gb:.1f} GB | {time.perf_counter() - t_start:.0f}s", flush=True)

    def record(res: Dict[str, Any]) -> None:
        nonlocal disp_found, docs_with_statute
        st = res["stats"]
        agg["layout"][st["layout"]] += 1
        agg["split"][st["headnote_split"]] += 1
        agg["roles"].update(st["roles"])
        agg["kind"].update(st["section_kind"])
        agg["removed"].update(st["removed"])
        agg["author"][st["author_source"]] += 1
        agg["bench"][st["bench_source"] or "none"] += 1
        agg["reportable"][str(st["reportable"])] += 1
        for s in [x for x in st["statutes"].split("; ") if x]:
            agg["statutes"][s] += 1
        docs_with_statute += st["n_statutes"] > 0
        disp_found += st["disposition_found"]
        author_agree[str(st["author_pdf_agrees"])] += 1
        disp_agree[str(st["disposition_agrees"])] += 1
        chunks_per_doc.append(len(res["children"]))

    s = settings
    inflight: deque = deque()
    it = iter(pending)
    max_inflight = max(2, workers * 3)
    try:
        while True:
            while len(inflight) < max_inflight:
                row = next(it, None)
                if row is None:
                    break
                inflight.append(pool.submit(jt.build_document, str(row["local_path"]), row,
                                            s.parent_chunk_size, s.child_chunk_size, s.chunk_overlap))
            if not inflight:
                break
            res = inflight.popleft().result()
            if "error" in res:
                checkpoint.mark_failed(res["doc_id"], res["error"])
                failures.append(f"{res['doc_id']}: {res['error']}")
                logger.warning(f"[fail] {res['doc_id']}: {res['error']}")
                continue
            record(res)
            batch.append(res)
            if len(batch) >= args.doc_batch_size:
                commit(batch)
                batch = []
        commit(batch)
        batch = []
    except ResourceGuardrailException as err:
        halted = f"RAM guard: {err}"
        logger.critical(f"[HALT] {halted}")
    except KeyboardInterrupt:
        halted = "interrupted (Ctrl+C)"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        checkpoint.save()
        t_bm = time.perf_counter()
        vstore.build_and_save_bm25_from_collection(collection)
        bm25_sec = time.perf_counter() - t_bm

    total_sec = time.perf_counter() - t_start
    col_count = vstore.get_or_create_collection(collection).count()
    S: List[str] = []
    S.append(f"INDEX | collection {collection} | {datetime.datetime.now():%Y-%m-%d %H:%M} | "
             f"workers {workers} (spawn) | torch threads {torch_threads} | embed batch {s.embed_batch_size}")
    S.append(f"chunking parent/child/overlap = {s.parent_chunk_size}/{s.child_chunk_size}/{s.chunk_overlap} | "
             f"model {s.local_embedding_model}")
    S.append(f"selected {len(rows)} | already in checkpoint {len(rows) - len(pending)} | indexed now {n_indexed} | "
             f"failed {len(failures)}" + (f" | HALTED: {halted}" if halted else ""))
    rate = n_indexed / (total_sec / 60) if total_sec else 0
    S.append(f"time {total_sec:.0f}s ({rate:.1f} docs/min) | embed+upsert {embed_sec:.0f}s "
             f"({n_chunks / embed_sec if embed_sec else 0:.0f} chunks/s) | BM25 rebuild {bm25_sec:.1f}s")
    if chunks_per_doc:
        S.append(f"chunks {n_chunks} | parents {n_parents} | chunks/doc mean {statistics.mean(chunks_per_doc):.1f} "
                 f"median {statistics.median(chunks_per_doc):.0f} max {max(chunks_per_doc)}")
    S.append(f"RAM peak (process tree) {guard.peak_ram_gb:.2f} GB / limit {s.max_ram_gb:.0f} GB")
    S.append(f"stores: Chroma rows {col_count} | SQLite parents {sqlite_store.count_parents()} | "
             f"documents {sqlite_store.count_documents()}")
    fmt = lambda c, n=12: "  ".join(f"{k}:{v}" for k, v in c.most_common(n))  # noqa: E731
    S.append("layout: " + fmt(agg["layout"]))
    S.append("headnote split: " + fmt(agg["split"]))
    S.append("section_kind (chunks): " + fmt(agg["kind"]))
    S.append("roles (chunks): " + fmt(agg["roles"]))
    S.append("author_source: " + fmt(agg["author"]) + " | PDF author vs manifest agree: " + fmt(author_agree))
    S.append("bench_source: " + fmt(agg["bench"]))
    S.append(f"disposition from PDF found {disp_found}/{n_indexed} | agrees with disposal_nature: " + fmt(disp_agree))
    S.append(f"docs with >=1 statute {docs_with_statute}/{n_indexed} | top: " + fmt(agg["statutes"], 10))
    S.append("removed lines: " + fmt(agg["removed"]))
    S.append(f"failures: {len(failures)}" + ("" if not failures else " (first 5)"))
    S.extend(f"  {f}"[:200] for f in failures[:5])
    S.append(f"detail log: $LOGS_DIR/{log_file.name}")
    logger.info("\n".join(S))
    print("\n".join(S))


if __name__ == "__main__":
    main()
