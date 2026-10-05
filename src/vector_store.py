"""Multi-Index Vector and Keyword Store for Indian Legal RAG.

Phase 2 Store Manager combining persistent ChromaDB (dense cosine similarity),
BM25Okapi (sparse keyword search with statutory precision), and metadata slicing.
"""

import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from google import genai
from google.genai import types
from google.genai.errors import APIError
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import Settings, get_settings

# Configure logger
logger = logging.getLogger("IndianLegalRAG.VectorStore")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def tokenize_legal_text(text: str) -> List[str]:
    """Tokenize legal text extracting words, section numbers, and statutory abbreviations."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def sanitize_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """Ensure all metadata values are primitive scalar types supported by ChromaDB."""
    clean: Dict[str, Union[str, int, float, bool]] = {}
    for k, v in metadata.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ", ".join(str(item) for item in v)
        elif isinstance(v, dict):
            clean[k] = str(v)
        else:
            clean[k] = str(v)
    return clean


def normalize_chroma_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize user filter dictionary to ChromaDB's where format with $and/$eq support."""
    if not filters:
        return None

    # If already using Chroma operator at root ($and, $or)
    if any(k.startswith("$") for k in filters.keys()):
        return filters

    clauses: List[Dict[str, Any]] = []
    for k, v in filters.items():
        if isinstance(v, dict):
            clauses.append({k: v})
        else:
            clauses.append({k: {"$eq": v}})

    if len(clauses) == 0:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class BM25Registry:
    """Synchronized disk-persisted document registry and BM25 index."""

    def __init__(self):
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self.corpus_metadatas: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.model: Optional[BM25Okapi] = None
        self._id_to_index: Dict[str, int] = {}

    def add_documents(
        self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add new documents, updating internal corpus and re-fitting BM25Okapi."""
        for chunk_id, text, meta in zip(ids, texts, metadatas):
            if chunk_id in self._id_to_index:
                # Update existing document
                idx = self._id_to_index[chunk_id]
                self.corpus_texts[idx] = text
                self.corpus_metadatas[idx] = meta
                self.tokenized_corpus[idx] = tokenize_legal_text(text)
            else:
                self._id_to_index[chunk_id] = len(self.corpus_ids)
                self.corpus_ids.append(chunk_id)
                self.corpus_texts.append(text)
                self.corpus_metadatas.append(meta)
                self.tokenized_corpus.append(tokenize_legal_text(text))

        if self.tokenized_corpus:
            self.model = BM25Okapi(self.tokenized_corpus)
            # Apply positive floor on IDF values to prevent terms from zeroing out in small corpora
            for word, val in self.model.idf.items():
                if val <= 0.0:
                    self.model.idf[word] = 0.2
        else:
            self.model = None

    def query(
        self, query: str, top_k: int = 20, allowed_ids: Optional[Union[List[str], Set[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Compute BM25 scores for tokenized query, optionally constrained to allowed IDs."""
        if not self.model or not self.corpus_ids:
            return []

        tokens = tokenize_legal_text(query)
        if not tokens:
            return []

        scores = self.model.get_scores(tokens)
        allowed_set = set(allowed_ids) if allowed_ids is not None else None

        results: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            chunk_id = self.corpus_ids[idx]
            if allowed_set is not None and chunk_id not in allowed_set:
                continue
            if score > 0.0:
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "text": self.corpus_texts[idx],
                        "metadata": self.corpus_metadatas[idx],
                        "bm25_score": float(score),
                    }
                )

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    def save(self, file_path: Path) -> None:
        """Serialize BM25 state to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "corpus_ids": self.corpus_ids,
            "corpus_texts": self.corpus_texts,
            "corpus_metadatas": self.corpus_metadatas,
            "tokenized_corpus": self.tokenized_corpus,
            "id_to_index": self._id_to_index,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: Path) -> "BM25Registry":
        """Load BM25 state from disk."""
        registry = cls()
        if not file_path.exists():
            return registry

        try:
            with open(file_path, "rb") as f:
                state = pickle.load(f)
            registry.corpus_ids = state.get("corpus_ids", [])
            registry.corpus_texts = state.get("corpus_texts", [])
            registry.corpus_metadatas = state.get("corpus_metadatas", [])
            registry.tokenized_corpus = state.get("tokenized_corpus", [])
            registry._id_to_index = state.get("id_to_index", {})
            if registry.tokenized_corpus:
                registry.model = BM25Okapi(registry.tokenized_corpus)
                for word, val in registry.model.idf.items():
                    if val <= 0.0:
                        registry.model.idf[word] = 0.2
        except Exception as err:
            logger.warning(f"Failed to load BM25 index from {file_path}: {err}. Initializing fresh registry.")
        return registry


class VectorStoreManager:
    """Production Multi-Index Store manager integrating ChromaDB, Gemini Embeddings, and BM25."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: Optional[str] = None,
        gemini_client: Optional[genai.Client] = None,
    ):
        self.settings = settings or get_settings()
        self.persist_dir = Path(self.settings.chroma_db_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.default_collection_name = collection_name or self.settings.collection_name
        self.client = gemini_client or (
            genai.Client(api_key=self.settings.gemini_api_key)
            if self.settings.gemini_api_key and self.settings.gemini_api_key != "your_gemini_api_key_here"
            else None
        )

        self._active_collections: Dict[str, Collection] = {}
        self._bm25_registries: Dict[str, BM25Registry] = {}
        self._query_embedding_cache: Dict[str, List[float]] = {}

        # Initialize default collection
        self.get_or_create_collection(self.default_collection_name)

    def _get_bm25_path(self, collection_name: str) -> Path:
        """Resolve path to BM25 pickle file alongside the Chroma store."""
        return self.persist_dir / f"{collection_name}_bm25.pkl"

    def get_or_create_collection(
        self, collection_name: Optional[str] = None, overwrite: bool = False
    ) -> Collection:
        """Get or initialize isolated ChromaDB collection and its accompanying BM25 registry."""
        name = collection_name or self.default_collection_name
        bm25_path = self._get_bm25_path(name)

        if overwrite:
            logger.info(f"Overwriting collection '{name}' and removing cached BM25 index...")
            try:
                self.chroma_client.delete_collection(name=name)
            except Exception:
                pass
            if bm25_path.exists():
                try:
                    bm25_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove old BM25 file {bm25_path}: {e}")
            self._active_collections.pop(name, None)
            self._bm25_registries.pop(name, None)

        if name not in self._active_collections:
            collection = self.chroma_client.get_or_create_collection(
                name=name,
                metadata={
                    "description": "Indian Legal Precedents and Judgment Chunks",
                    "hnsw:space": "cosine",
                },
            )
            self._active_collections[name] = collection

        if name not in self._bm25_registries:
            registry = BM25Registry.load(bm25_path)
            if not registry.corpus_ids and collection.count() > 0:
                logger.info(f"Synchronizing BM25 index from {collection.count()} persisted ChromaDB records...")
                all_records = collection.get(include=["documents", "metadatas"])
                if all_records and all_records.get("ids"):
                    registry.add_documents(
                        ids=all_records["ids"],
                        texts=all_records["documents"],
                        metadatas=all_records["metadatas"],
                    )
                    registry.save(bm25_path)
            self._bm25_registries[name] = registry

        return self._active_collections[name]

    def _embed_texts_batch(
        self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Embed text batch with Gemini embedding model, handling retries and rate limit backoff."""
        if not self.client:
            raise ValueError("GEMINI_API_KEY is not configured for embedding generation.")

        model_candidates = [
            self.settings.gemini_embedding_model,
            "gemini-embedding-2",
            "gemini-embedding-2-preview",
            "gemini-embedding-001",
        ]
        models_to_try = list(dict.fromkeys(model_candidates))

        last_error = None
        for model_name in models_to_try:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    contents_payload = [
                        types.Content(parts=[types.Part.from_text(text=t)])
                        for t in texts
                    ]
                    response = self.client.models.embed_content(
                        model=model_name,
                        contents=contents_payload,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=768,
                        ),
                    )
                    embeddings = [list(emb.values) for emb in response.embeddings]
                    if len(embeddings) != len(texts):
                        raise ValueError(
                            f"Model '{model_name}' returned {len(embeddings)} embeddings for {len(texts)} texts."
                        )
                    return embeddings
                except APIError as api_err:
                    last_error = api_err
                    if "404" in str(api_err):
                        logger.info(f"Embedding model '{model_name}' not supported (404). Trying next...")
                        break
                    elif "503" in str(api_err) or "429" in str(api_err):
                        err_str = str(api_err)
                        delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                        if not delay_match:
                            delay_match = re.search(r"retryDelay': '(\d+)s", err_str, re.IGNORECASE)
                        if delay_match:
                            backoff = float(delay_match.group(1)) + 1.0
                        else:
                            backoff = 22.0

                        if backoff > 120.0:
                            logger.error(
                                f"API quota exhausted for {model_name} with long reset time ({backoff:.0f}s). "
                                f"Daily limit reached. Aborting retries."
                            )
                            break

                        logger.warning(
                            f"Encountered HTTP 429/503 on {model_name}: {api_err}. "
                            f"Sleeping for {backoff:.1f} seconds and retrying batch (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(backoff)
                        continue
                    raise
                except Exception as err:
                    last_error = err
                    logger.warning(f"Embedding generation error on {model_name}: {err}")
                    break

        raise RuntimeError(f"Failed to generate embeddings across all candidate models. Error: {last_error}")

    def build_and_save_bm25_from_collection(self, collection_name: Optional[str] = None) -> Path:
        """Build BM25 index across ALL chunks (existing + new) in ChromaDB collection and persist to disk."""
        target_name = collection_name or self.default_collection_name
        collection = self.get_or_create_collection(target_name)
        bm25_path = self._get_bm25_path(target_name)

        all_records = collection.get(include=["documents", "metadatas"])
        all_ids = all_records.get("ids", []) if all_records else []
        all_texts = all_records.get("documents", []) if all_records else []
        all_metas = all_records.get("metadatas", []) if all_records else []

        registry = BM25Registry()
        if all_ids:
            registry.add_documents(ids=all_ids, texts=all_texts, metadatas=all_metas)
        registry.save(bm25_path)
        self._bm25_registries[target_name] = registry
        logger.info(f"[INFO] Built BM25 model across all {len(all_ids)} chunks and saved to {bm25_path}")
        return bm25_path

    def index_documents(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Ingest child chunks into ChromaDB and synchronized BM25 index with throughput telemetry.

        Args:
            chunks: List of child chunk dictionaries (from LegalDocumentLoader).
            collection_name: Target collection name (defaults to active collection).
            batch_size: Embedding and ingestion batch size (default: 50).

        Returns:
            Dictionary containing indexing summary metrics.
        """
        if not chunks:
            logger.warning("No chunks provided to index_documents.")
            return {"indexed_count": 0, "duration_sec": 0.0, "throughput_chunks_per_sec": 0.0}

        target_name = collection_name or self.default_collection_name
        collection = self.get_or_create_collection(target_name)
        bm25_path = self._get_bm25_path(target_name)

        # 1. Resumable indexing: query collection to find already indexed IDs
        existing_ids: Set[str] = set()
        if collection.count() > 0:
            existing_data = collection.get(include=[])
            if existing_data and existing_data.get("ids"):
                existing_ids = set(existing_data["ids"])

        chunks_to_index = [c for c in chunks if c.get("chunk_id") not in existing_ids]

        if existing_ids:
            logger.info(
                f"[INFO] Found {len(existing_ids)} chunks already in ChromaDB. "
                f"Indexing remaining {len(chunks_to_index)} chunks..."
            )

        if not chunks_to_index:
            logger.info(
                f"All {len(chunks)} chunks already exist in collection '{target_name}'. Skipping embedding."
            )
            self.build_and_save_bm25_from_collection(target_name)
            return {
                "collection_name": target_name,
                "indexed_count": collection.count(),
                "duration_sec": 0.0,
                "throughput_chunks_per_sec": 0.0,
                "bm25_index_path": str(bm25_path),
                "chroma_db_dir": str(self.persist_dir),
            }

        total_chunks = len(chunks_to_index)
        start_time = time.perf_counter()

        indexed_count = 0
        pbar = tqdm(
            total=total_chunks,
            desc=f"Indexing ({target_name})",
            unit="chunk",
            dynamic_ncols=True,
        )

        try:
            for i in range(0, total_chunks, batch_size):
                batch = chunks_to_index[i : i + batch_size]
                batch_texts = [c.get("text", "") for c in batch]
                batch_ids = [c.get("chunk_id", "") for c in batch]
                batch_metas = []

                for c in batch:
                    meta = {k: v for k, v in c.items() if k not in ["chunk_id", "text"]}
                    batch_metas.append(sanitize_metadata_for_chroma(meta))

                # Step 1: Generate Dense Embeddings
                batch_embeddings = self._embed_texts_batch(
                    batch_texts, task_type="RETRIEVAL_DOCUMENT"
                )

                # Step 2: Ingest into ChromaDB
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas,
                )

                indexed_count += len(batch)
                elapsed = time.perf_counter() - start_time
                throughput = indexed_count / elapsed if elapsed > 0 else 0.0

                pbar.set_postfix({"throughput": f"{throughput:.1f} ch/s"})
                pbar.update(len(batch))

                # Rate-Limit Pacing: sleep 3.5s between batches to respect rolling RPM limits and laptop thermals
                time.sleep(3.5)

        finally:
            pbar.close()
            # Ensure the BM25 model is built across ALL chunks (existing + new) and saved to disk
            self.build_and_save_bm25_from_collection(target_name)

        total_duration = time.perf_counter() - start_time
        avg_throughput = total_chunks / total_duration if total_duration > 0 else 0.0

        summary = {
            "collection_name": target_name,
            "indexed_count": collection.count(),
            "duration_sec": round(total_duration, 2),
            "throughput_chunks_per_sec": round(avg_throughput, 2),
            "bm25_index_path": str(bm25_path),
            "chroma_db_dir": str(self.persist_dir),
        }
        logger.info(f"Indexing complete: {summary}")
        return summary

    def query_dense(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute dense cosine semantic search with ChromaDB metadata slicing.

        Args:
            query: User legal query or precedent description.
            top_k: Maximum candidate chunks to retrieve.
            filters: Metadata filter dictionary supporting $and, $eq, $gte on court, year, role.
            collection_name: Optional target collection name.

        Returns:
            List of dicts: chunk_id, text, metadata, similarity_score (1 - cosine distance).
        """
        target_name = collection_name or self.default_collection_name
        collection = self.get_or_create_collection(target_name)

        if collection.count() == 0:
            return []

        # Embed query with caching
        if hasattr(self, "_query_embedding_cache") and query in self._query_embedding_cache:
            query_embeddings = [self._query_embedding_cache[query]]
        else:
            query_embeddings = self._embed_texts_batch(
                [query], task_type="RETRIEVAL_QUERY"
            )
            if hasattr(self, "_query_embedding_cache") and query_embeddings:
                self._query_embedding_cache[query] = query_embeddings[0]

        chroma_where = normalize_chroma_filters(filters)

        # Retrieve nearest vectors
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, collection.count()),
            where=chroma_where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
                # In cosine space, similarity = 1.0 - distance
                similarity_score = max(0.0, min(1.0, 1.0 - dist))
                formatted_results.append(
                    {
                        "chunk_id": chunk_id,
                        "text": doc,
                        "metadata": meta,
                        "similarity_score": round(float(similarity_score), 4),
                    }
                )

        return formatted_results

    def query_bm25(
        self,
        query: str,
        top_k: int = 20,
        allowed_ids: Optional[Union[List[str], Set[str]]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute sparse lexical keyword search via BM25Okapi with optional ID slicing.

        Args:
            query: User legal query (e.g. 'Section 302 IPC', 'Article 21').
            top_k: Maximum candidate chunks to retrieve.
            allowed_ids: Optional list/set of chunk IDs to restrict BM25 scoring.
            collection_name: Optional target collection name.

        Returns:
            List of dicts: chunk_id, text, metadata, bm25_score.
        """
        target_name = collection_name or self.default_collection_name
        self.get_or_create_collection(target_name)
        registry = self._bm25_registries.get(target_name)

        if not registry:
            return []

        return registry.query(query=query, top_k=top_k, allowed_ids=allowed_ids)

    def filter_ids_by_metadata(
        self, filters: Dict[str, Any], collection_name: Optional[str] = None
    ) -> List[str]:
        """Fetch chunk IDs from Chroma matching metadata filters (for BM25 slicing)."""
        target_name = collection_name or self.default_collection_name
        collection = self.get_or_create_collection(target_name)
        chroma_where = normalize_chroma_filters(filters)

        if not chroma_where:
            get_res = collection.get(include=[])
            return get_res.get("ids", [])

        get_res = collection.get(where=chroma_where, include=[])
        return get_res.get("ids", [])


# Backward compatibility alias
ChromaVectorStore = VectorStoreManager
