"""Unit tests for Multi-Index VectorStoreManager (ChromaDB + BM25 + Metadata Slicing)."""

import os
import shutil
import unittest
import uuid
from pathlib import Path

from src.config import Settings, get_settings
from src.vector_store import BM25Registry, VectorStoreManager, normalize_chroma_filters


class TestVectorStoreManager(unittest.TestCase):
    """Test suite for Phase 2 multi-index retrieval and slicing."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.test_collection = f"test_legal_col_{uuid.uuid4().hex[:6]}"
        cls.manager = VectorStoreManager(
            settings=cls.settings, collection_name=cls.test_collection
        )

        # Sample legal child chunks with distinct statutory concepts & courts
        cls.mock_chunks = [
            {
                "chunk_id": "chunk-302-ipc",
                "text": "Section 302 IPC prescribes punishment for murder where death is caused with intention or knowledge.",
                "court": "Supreme Court of India",
                "year": 2024,
                "role": "Ratio Decidendi",
                "case_title": "State of Maharashtra vs. Sharma",
                "char_length": 105,
            },
            {
                "chunk_id": "chunk-304b-ipc",
                "text": "Under Section 304B IPC, dowry death requires proof of cruelty or harassment shortly before death.",
                "court": "High Court of Delhi",
                "year": 2022,
                "role": "Facts",
                "case_title": "Rajesh Kumar vs. State",
                "char_length": 98,
            },
            {
                "chunk_id": "chunk-art21-const",
                "text": "Article 21 guarantees that no person shall be deprived of life or personal liberty except by procedure established by law.",
                "court": "Supreme Court of India",
                "year": 2023,
                "role": "Ratio Decidendi",
                "case_title": "Puttaswamy vs. Union of India",
                "char_length": 123,
            },
            {
                "chunk_id": "chunk-420-ipc",
                "text": "Section 420 IPC penalizes cheating and dishonestly inducing delivery of property with fraudulent intent.",
                "court": "High Court of Bombay",
                "year": 2020,
                "role": "Obiter Dicta",
                "case_title": "Mehta vs. State of Maharashtra",
                "char_length": 104,
            },
        ]

    @classmethod
    def tearDownClass(cls):
        # Cleanup isolated test collection and bm25 pkl
        try:
            cls.manager.chroma_client.delete_collection(name=cls.test_collection)
        except Exception:
            pass
        bm25_file = cls.manager._get_bm25_path(cls.test_collection)
        if bm25_file.exists():
            bm25_file.unlink(missing_ok=True)

    def test_01_index_documents_and_persistence(self):
        """Test batch indexing into ChromaDB and disk persistence of BM25."""
        summary = self.manager.index_documents(
            self.mock_chunks,
            collection_name=self.test_collection,
            batch_size=2,
        )

        self.assertEqual(summary["indexed_count"], 4)
        self.assertGreater(summary["throughput_chunks_per_sec"], 0)

        # Verify ChromaDB storage
        col = self.manager.get_or_create_collection(self.test_collection)
        self.assertEqual(col.count(), 4)

        # Verify BM25 file on disk
        bm25_path = self.manager._get_bm25_path(self.test_collection)
        self.assertTrue(bm25_path.exists())

    def test_02_reloading_from_disk(self):
        """Verify that a newly instantiated manager reloads ChromaDB and BM25 without re-indexing."""
        new_manager = VectorStoreManager(
            settings=self.settings, collection_name=self.test_collection
        )
        col = new_manager.get_or_create_collection(self.test_collection)
        self.assertEqual(col.count(), 4)

        registry = new_manager._bm25_registries[self.test_collection]
        self.assertEqual(len(registry.corpus_ids), 4)
        self.assertIn("chunk-302-ipc", registry.corpus_ids)

    def test_03_bm25_statutory_exact_retrieval(self):
        """Verify BM25 accurately matches exact statutory numbers (e.g. 'Section 302 IPC')."""
        results_302 = self.manager.query_bm25(
            "Section 302 IPC", top_k=2, collection_name=self.test_collection
        )
        self.assertGreater(len(results_302), 0)
        self.assertEqual(results_302[0]["chunk_id"], "chunk-302-ipc")
        self.assertIn("Section 302", results_302[0]["text"])

        results_304b = self.manager.query_bm25(
            "Section 304B dowry death", top_k=2, collection_name=self.test_collection
        )
        self.assertGreater(len(results_304b), 0)
        self.assertEqual(results_304b[0]["chunk_id"], "chunk-304b-ipc")

    def test_04_dense_semantic_retrieval(self):
        """Verify dense retrieval computes similarity scores and semantic relevance."""
        results = self.manager.query_dense(
            "punishment for murder and intentional death",
            top_k=2,
            collection_name=self.test_collection,
        )
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["chunk_id"], "chunk-302-ipc")
        self.assertGreater(top_match["similarity_score"], 0.4)

    def test_05_metadata_slicing_dense(self):
        """Verify ChromaDB metadata slicing properly restricts results by court and role."""
        # Filter for Supreme Court + Ratio Decidendi
        filters = {
            "court": "Supreme Court of India",
            "role": "Ratio Decidendi",
        }
        results = self.manager.query_dense(
            "law and punishment",
            top_k=10,
            filters=filters,
            collection_name=self.test_collection,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["metadata"]["court"], "Supreme Court of India")
            self.assertEqual(r["metadata"]["role"], "Ratio Decidendi")

        # Ensure High Court chunks are excluded
        returned_ids = [r["chunk_id"] for r in results]
        self.assertNotIn("chunk-304b-ipc", returned_ids)
        self.assertNotIn("chunk-420-ipc", returned_ids)

    def test_06_metadata_slicing_with_bm25(self):
        """Verify metadata pre-filtering applied to BM25 using allowed_ids."""
        # Find all chunks from High Court of Delhi
        delhi_ids = self.manager.filter_ids_by_metadata(
            {"court": "High Court of Delhi"}, collection_name=self.test_collection
        )
        self.assertEqual(delhi_ids, ["chunk-304b-ipc"])

        # Query across death/punishment restricted to Delhi High Court
        bm25_results = self.manager.query_bm25(
            "death punishment",
            top_k=5,
            allowed_ids=delhi_ids,
            collection_name=self.test_collection,
        )
        self.assertEqual(len(bm25_results), 1)
        self.assertEqual(bm25_results[0]["chunk_id"], "chunk-304b-ipc")


if __name__ == "__main__":
    unittest.main()
