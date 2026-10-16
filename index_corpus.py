#!/usr/bin/env python3
"""Production Resumable Stream Ingestion & Vector Indexing Engine.

Engineered for large-scale legal corpora (100–400+ judgments):
- Strict Resource Guardrails: Caps RAM at 20 GB and monitored storage at 30 GB
- True Streaming Pipeline: Ingests documents in micro-batches (5-10 PDFs at a time),
  preventing memory ballooning and avoiding container cgroup OOM kills
- Persistent Checkpointing: Records progress in data/processed/indexing_checkpoint.json,
  enabling instant seamless resumption after interruptions or restarts
- 3-Tier Section-Bounded Chunking with Rhetorical Flush & Context Prepending
- O(1) SQLite Parent Store persistence
- High-Throughput Neural Embeddings (zeroentropy/zembed-1 BF16 or Gemini API)
- Fault Tolerance: Gracefully handles scanned/corrupted PDFs, logs to failed_files.json
"""

import argparse
import datetime
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from src.config import Settings, get_settings
from src.document_loader import LegalDocumentLoader
from src.resource_guard import ResourceGuard, ResourceGuardrailException
from src.sqlite_store import SQLiteParentStore
from src.vector_store import VectorStoreManager

# Setup rotating file and console logging
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "indexing.log"

logger = logging.getLogger("IndianLegalRAG.Indexer")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console Handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File Handler
fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)


class CheckpointManager:
    """Manages persistent indexing state on disk to allow resilient stop-and-resume execution."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {
            "indexed_files": [],
            "failed_files": [],
            "total_chunks_indexed": 0,
            "total_parents_indexed": 0,
            "last_updated": None,
        }
        self.load()

    def load(self) -> None:
        """Load checkpoint state if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(
                    f"[Checkpoint] Loaded existing checkpoint: {len(self.data.get('indexed_files', []))} files indexed."
                )
            except Exception as err:
                logger.warning(f"[Checkpoint] Could not load checkpoint file ({err}). Starting fresh.")

    def save(self) -> None:
        """Atomically persist checkpoint state to disk."""
        self.data["last_updated"] = datetime.datetime.now().isoformat()
        temp_file = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.checkpoint_path)
        except Exception as err:
            logger.warning(f"[Checkpoint] Failed to save checkpoint: {err}")

    def is_file_indexed(self, file_path: Path) -> bool:
        """Check if file has already been committed to the index."""
        rel_or_name = file_path.name
        full_str = str(file_path.resolve())
        return full_str in self.data.get("indexed_files", []) or rel_or_name in self.data.get("indexed_files", [])

    def mark_file_indexed(self, file_path: Path, num_chunks: int, num_parents: int) -> None:
        """Record completed file."""
        self.data["indexed_files"].append(str(file_path.resolve()))
        self.data["total_chunks_indexed"] = self.data.get("total_chunks_indexed", 0) + num_chunks
        self.data["total_parents_indexed"] = self.data.get("total_parents_indexed", 0) + num_parents
        self.save()

    def mark_file_failed(self, file_path: Path, reason: str) -> None:
        """Record unprocessable or corrupted file."""
        self.data["failed_files"].append({
            "path": str(file_path.resolve()),
            "filename": file_path.name,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        self.save()


def parse_arguments() -> argparse.Namespace:
    """Parse CLI configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Streamed, Hardware-Safe Legal Corpus Indexing with Checkpoints"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories (scanned recursively) or individual PDF files (defaults to data/raw_judgments)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (defaults to indian_legal_precedents)",
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=8,
        help="Number of documents to extract and embed per micro-batch (prevents RAM ballooning)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for neural embedding inference",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the checkpoint, the ChromaDB collection, its BM25 index and the SQLite parent store, then re-index from scratch",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    settings = get_settings()
    settings.validate()

    target_collection = args.collection or settings.collection_name
    doc_batch_size = args.doc_batch_size
    embed_batch_size = args.embedding_batch_size

    # 1. Initialize Guardrails Monitor
    guard = ResourceGuard(
        max_ram_gb=settings.max_ram_gb,
        warn_ram_gb=settings.warn_ram_gb,
        max_storage_gb=settings.max_storage_gb,
        storage_check_paths=[
            settings.chroma_db_dir,
            settings.sqlite_db_path.parent,
            settings.parent_store_dir,
        ],
        vram_safety_fraction=settings.vram_safety_fraction,
        num_cpu_threads=settings.num_threads,
    )
    hw_telemetry = guard.initialize_hardware()

    logger.info("=" * 75)
    logger.info("  INDIAN LEGAL RAG - PRODUCTION STREAM INDEXING ENGINE")
    logger.info("=" * 75)
    logger.info(f"  * Device: {hw_telemetry.get('device_name')} (Threads: {hw_telemetry.get('threads_set')})")
    if hw_telemetry.get("cuda_available"):
        logger.info(f"  * VRAM Limit: ~{hw_telemetry.get('vram_allocated_cap_gb')} GB / {hw_telemetry.get('total_vram_gb')} GB")
    logger.info(f"  * RAM Ceiling: {settings.max_ram_gb:.1f} GB | Storage Ceiling: {settings.max_storage_gb:.1f} GB")
    logger.info(f"  * Embedding Provider: {settings.embedding_provider} ({settings.local_embedding_model if settings.embedding_provider=='local' else settings.gemini_embedding_model})")
    logger.info(f"  * SQLite Store: {settings.sqlite_db_path}")
    logger.info(f"  * Target Collection: {target_collection}")
    logger.info("=" * 75)

    # 2. Checkpoint Management
    checkpoint_mgr = CheckpointManager(settings.checkpoint_path)
    if args.reset:
        logger.warning("[RESET] --reset specified. Wiping checkpoint and initializing clean run.")
        checkpoint_mgr.data = {"indexed_files": [], "failed_files": [], "total_chunks_indexed": 0, "total_parents_indexed": 0}
        checkpoint_mgr.save()

    # 3. Locate Data Directories
    requested_dirs = args.data_dir or [str(settings.raw_data_dir)]
    data_dirs = [Path(d).resolve() for d in requested_dirs if Path(d).exists()]
    missing_dirs = [d for d in requested_dirs if not Path(d).exists()]
    if missing_dirs:
        logger.error(f"Data directories not found: {missing_dirs}")
        sys.exit(1)

    # Case-insensitive suffix match: Linux globbing treats '*.pdf' and '*.PDF' as different
    seen_files: Set[Path] = set()
    for data_dir in data_dirs:
        if data_dir.is_file():
            if data_dir.suffix.lower() == ".pdf":
                seen_files.add(data_dir)
            continue
        logger.info(f"Scanning for PDF judgments in: {data_dir}")
        seen_files.update(
            p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
        )
    all_pdf_files = sorted(seen_files)
    total_found = len(all_pdf_files)
    logger.info(f"Discovered {total_found} PDF judgments across directory tree.")

    if total_found == 0:
        logger.warning("No PDF files found to index. Exiting.")
        sys.exit(0)

    # Filter out already indexed files
    pending_files = [f for f in all_pdf_files if not checkpoint_mgr.is_file_indexed(f)]
    logger.info(
        f"Corpus Status: {total_found - len(pending_files)} already indexed, {len(pending_files)} pending ingestion."
    )

    if not pending_files:
        logger.info("[COMPLETE] All files in the directory are already indexed. Nothing to do!")
        sys.exit(0)

    # 4. Initialize Core Pipeline Components
    loader = LegalDocumentLoader(settings=settings, enable_ocr=False)
    vstore = VectorStoreManager(settings=settings, collection_name=target_collection)
    if args.reset:
        logger.warning(f"[RESET] Dropping collection '{target_collection}', its BM25 index and all SQLite parents.")
        vstore.get_or_create_collection(target_collection, overwrite=True)
        loader.sqlite_store.clear()

    # 5. Streaming Execution Loop (Micro-Batches of Files)
    overall_pbar = tqdm(
        total=len(pending_files),
        desc="Corpus Progress (PDFs)",
        unit="pdf",
        dynamic_ncols=True,
    )

    start_total_t = time.perf_counter()
    session_chunks_indexed = 0
    session_parents_indexed = 0

    try:
        for file_idx in range(0, len(pending_files), doc_batch_size):
            file_micro_batch = pending_files[file_idx : file_idx + doc_batch_size]
            batch_children: List[Dict[str, Any]] = []
            batch_parents: List[Dict[str, Any]] = []
            files_processed_in_step: List[Tuple[Path, int, int]] = []

            # Step A: Parse documents in current micro-batch
            for file_path in file_micro_batch:
                try:
                    # Enforce guardrails before extracting
                    guard.enforce_guardrails("PDFExtraction")

                    text, _, _ = loader.load_document(file_path, allow_ocr=False)
                    if not text or len(text.strip()) < 50:
                        logger.warning(f"File '{file_path.name}' yielded insufficient or empty text. Recording failure.")
                        checkpoint_mgr.mark_file_failed(file_path, "Empty text or scanned PDF requiring OCR")
                        overall_pbar.update(1)
                        continue

                    # 3-Tier Chunking with Rhetorical Flushing & Prepending
                    parents, children = loader.create_parent_child_chunks(text, str(file_path))

                    if not children:
                        logger.warning(f"No child chunks created for '{file_path.name}'.")
                        checkpoint_mgr.mark_file_failed(file_path, "Zero chunks generated")
                        overall_pbar.update(1)
                        continue

                    batch_parents.extend(parents)
                    batch_children.extend(children)
                    files_processed_in_step.append((file_path, len(children), len(parents)))

                except Exception as doc_err:
                    logger.error(f"Error processing document {file_path.name}: {doc_err}")
                    checkpoint_mgr.mark_file_failed(file_path, str(doc_err))

                overall_pbar.update(1)

            # Step B: Ingest & Embed Micro-Batch Chunks
            if batch_children:
                logger.info(
                    f"[StreamBatch] Embedding & committing {len(batch_children)} chunks from {len(files_processed_in_step)} documents..."
                )
                guard.enforce_guardrails("VectorEmbedding")

                vstore.index_documents(
                    chunks=batch_children,
                    collection_name=target_collection,
                    batch_size=embed_batch_size,
                    rebuild_bm25=False,  # rebuilt once over the full collection in the finally block
                )

                session_chunks_indexed += len(batch_children)
                session_parents_indexed += len(batch_parents)

                # Step C: Commit checkpoint for this batch
                for f_path, c_count, p_count in files_processed_in_step:
                    checkpoint_mgr.mark_file_indexed(f_path, c_count, p_count)

            # Step D: Memory Compaction & Telemetry
            gc.collect()
            if hw_telemetry.get("cuda_available"):
                import torch
                torch.cuda.empty_cache()

            telemetry_str = guard.format_telemetry("Health")
            overall_pbar.set_postfix({"stats": telemetry_str})

    except ResourceGuardrailException as guard_err:
        logger.critical(f"\n[HALT] Indexing paused safely by Resource Guardrail: {guard_err}")
        logger.info("[INFO] Checkpoint saved. Re-run script to resume from exact halting point.")
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPTED] User cancelled indexing via SIGINT (Ctrl+C).")
        logger.info("[INFO] Saving current progress and building BM25 index before exit...")
    except Exception as fatal_err:
        logger.critical(f"\n[FATAL ERROR] Unexpected exception during indexing: {fatal_err}", exc_info=True)
    finally:
        overall_pbar.close()

        # Step E: Synchronize BM25 Index Across Entire Collection
        logger.info("\n[Finalizing] Synchronizing global BM25 keyword index...")
        bm25_path = vstore.build_and_save_bm25_from_collection(target_collection)
        checkpoint_mgr.save()

        total_elapsed = time.perf_counter() - start_total_t
        total_in_db = vstore.get_or_create_collection(target_collection).count()
        parents_in_sqlite = loader.sqlite_store.count_parents()

        logger.info("=" * 75)
        logger.info("  STREAM INGESTION SESSION COMPLETED")
        logger.info("=" * 75)
        logger.info(f"  * Session Chunks Indexed     : {session_chunks_indexed:,}")
        logger.info(f"  * Total Chunks in ChromaDB   : {total_in_db:,}")
        logger.info(f"  * Total Parents in SQLite    : {parents_in_sqlite:,}")
        logger.info(f"  * Global BM25 Index Path     : {bm25_path}")
        logger.info(f"  * Elapsed Time               : {total_elapsed:.2f}s")
        logger.info(f"  * System Telemetry           : {guard.format_telemetry('Final')}")
        logger.info("=" * 75)


if __name__ == "__main__":
    main()
