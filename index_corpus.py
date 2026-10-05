import os
import sys
import time
from pathlib import Path
from src.config import get_settings
from src.document_loader import LegalDocumentLoader
from src.vector_store import VectorStoreManager

DATA_DIR = r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data"
COLLECTION_NAME = "indian_legal_precedents"
BATCH_SIZE = 50


def main():
    print("=" * 70)
    print("  INDIAN LEGAL RAG - CORPUS INGESTION & VECTOR INDEXING")
    print("=" * 70)
    sys.stdout.flush()

    settings = get_settings()
    settings.validate()

    print("[1/3] Initializing LegalDocumentLoader and VectorStoreManager...")
    sys.stdout.flush()
    loader = LegalDocumentLoader(settings=settings)
    vstore = VectorStoreManager(settings=settings, collection_name=COLLECTION_NAME)

    print(f"\n[2/3] Processing judgment corpus from: {DATA_DIR}")
    sys.stdout.flush()
    start_extraction = time.perf_counter()
    result = loader.process_directory(DATA_DIR, recursive=True, allow_ocr=False)
    extract_time = time.perf_counter() - start_extraction

    # Safely unpack child chunks supporting both tuple and dict formats
    if isinstance(result, tuple):
        parents, children = result[0], result[1]
    elif isinstance(result, dict):
        children = result.get("children") or result.get("child_chunks") or []
        parents = result.get("parents") or result.get("parent_chunks") or []
    else:
        raise ValueError(f"Unexpected return type from process_directory: {type(result)}")

    total_children = len(children)
    total_parents = len(parents)

    print(f"\n  Extraction Summary:")
    print(f"  * Parent Chunks Generated : {total_parents:,}")
    print(f"  * Child Chunks Generated  : {total_children:,}")
    print(f"  * Extraction Time         : {extract_time:.2f}s")
    sys.stdout.flush()

    if total_children == 0:
        print("[ERROR] No child chunks were found or extracted. Aborting indexing.")
        sys.stdout.flush()
        sys.exit(1)

    # 1. Resumable Chunk Indexing:
    # Query ChromaDB collection to get the set of existing IDs and indexed file paths
    collection = vstore.get_or_create_collection(COLLECTION_NAME)
    existing_records = collection.get(include=["metadatas"])
    existing_ids = set(existing_records["ids"]) if existing_records and existing_records.get("ids") else set()
    indexed_paths = {
        m.get("source_path")
        for m in existing_records.get("metadatas", [])
        if m and m.get("source_path")
    }

    if len(existing_ids) > 0:
        # Filter out chunks from files already indexed in ChromaDB
        children = [
            c for c in children
            if c.get("source_path") not in indexed_paths and c.get("chunk_id") not in existing_ids
        ]
        print(f"[INFO] Found {len(existing_ids)} chunks from {len(indexed_paths)} files already in ChromaDB.")
        print(f"[INFO] Indexing remaining {len(children)} chunks from new files...")
        sys.stdout.flush()

    print(f"\n[3/3] Ingesting {len(children):,} child chunks into ChromaDB & BM25 index...")
    print(f"  * Target Collection: {COLLECTION_NAME}")
    print(f"  * Embedding Batch Size: {BATCH_SIZE}")
    print(f"  * Embedding Model: {settings.gemini_embedding_model}")
    sys.stdout.flush()

    start_index = time.perf_counter()
    if len(children) > 0:
        summary = vstore.index_documents(
            chunks=children,
            collection_name=COLLECTION_NAME,
            batch_size=BATCH_SIZE,
        )
    else:
        print(f"[INFO] All chunks are already indexed in ChromaDB. Skipping embedding step.")
        sys.stdout.flush()
        summary = {
            "collection_name": COLLECTION_NAME,
            "indexed_count": collection.count(),
            "duration_sec": 0.0,
            "throughput_chunks_per_sec": 0.0,
            "bm25_index_path": str(vstore._get_bm25_path(COLLECTION_NAME)),
            "chroma_db_dir": str(vstore.persist_dir),
        }
    index_time = time.perf_counter() - start_index

    # Ensure the BM25 model is built across ALL chunks (existing + new) and saved to disk
    print(f"\n[INFO] Building synchronized BM25 index across ALL chunks in '{COLLECTION_NAME}'...")
    sys.stdout.flush()
    bm25_path = vstore.build_and_save_bm25_from_collection(COLLECTION_NAME)

    total_in_collection = collection.count()

    print("\n" + "=" * 70)
    print("  INDEXING COMPLETE - SUMMARY TELEMETRY")
    print("=" * 70)
    print(f"  * Collection Name        : {summary.get('collection_name')}")
    print(f"  * Total Chunks in DB     : {total_in_collection:,}")
    print(f"  * New Chunks Indexed     : {len(children):,}")
    print(f"  * Indexing Time          : {summary.get('duration_sec')}s")
    print(f"  * Indexing Throughput    : {summary.get('throughput_chunks_per_sec')} chunks/s")
    print(f"  * BM25 Model Saved To    : {bm25_path}")
    print(f"  * ChromaDB Persist Dir   : {summary.get('chroma_db_dir')}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

