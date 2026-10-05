"""Unit tests for Phase 3 Indian Legal RAG Pipeline.

Verifies Reciprocal Rank Fusion, Cross-Encoder rescoring, Parent Expansion,
Grounded Legal Synthesis, and granular timing telemetry.
"""

import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from src.config import Settings, get_settings
from src.rag_pipeline import IndianLegalRAGPipeline
from src.vector_store import VectorStoreManager


class MockCrossEncoder:
    """Mock CrossEncoder that deterministically scores pairs based on query keyword matching."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            overlap = len(q_words.intersection(t_words))
            scores.append(float(overlap) + 0.1)
        return scores


class TestIndianLegalRAGPipeline(unittest.TestCase):
    """Test suite for Hybrid Retrieval Funnel and Pipeline Orchestration."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.test_col = f"test_rag_pipeline_{uuid.uuid4().hex[:6]}"
        cls.vector_store = VectorStoreManager(
            settings=cls.settings, collection_name=cls.test_col
        )
        cls.pipeline = IndianLegalRAGPipeline(
            settings=cls.settings, vector_store=cls.vector_store
        )
        cls.pipeline._reranker = MockCrossEncoder()

        # Create persistent mock parent JSON documents in parent store
        cls.parent_id_1 = str(uuid.uuid4())
        cls.parent_id_2 = str(uuid.uuid4())

        cls.parent_doc_1 = {
            "parent_id": cls.parent_id_1,
            "case_title": "State of Maharashtra vs. Sharma",
            "text": (
                "FULL JUDGMENT CONTEXT:\n"
                "The Supreme Court observed that under Section 302 of the Indian Penal Code, "
                "the prosecution must establish the bodily injury was intended and sufficient in "
                "the ordinary course of nature to cause death. This constitutes binding Ratio Decidendi."
            ),
            "metadata": {
                "court": "Supreme Court of India",
                "year": 2024,
                "structural_heading": "Ratio Decidendi",
            },
        }

        cls.parent_doc_2 = {
            "parent_id": cls.parent_id_2,
            "case_title": "Rajesh vs. State of Delhi",
            "text": (
                "FULL JUDGMENT CONTEXT:\n"
                "The Delhi High Court examined Section 304B regarding dowry death and cruelty. "
                "The bench noted circumstantial aspects as Obiter Dicta."
            ),
            "metadata": {
                "court": "High Court of Delhi",
                "year": 2022,
                "structural_heading": "Facts",
            },
        }

        # Write to parent store
        cls.settings.parent_store_dir.mkdir(parents=True, exist_ok=True)
        with open(cls.settings.parent_store_dir / f"{cls.parent_id_1}.json", "w", encoding="utf-8") as f:
            json.dump(cls.parent_doc_1, f)
        with open(cls.settings.parent_store_dir / f"{cls.parent_id_2}.json", "w", encoding="utf-8") as f:
            json.dump(cls.parent_doc_2, f)

        # Index child chunks referencing these parents
        cls.mock_children = [
            {
                "chunk_id": f"child-{cls.parent_id_1}-1",
                "parent_id": cls.parent_id_1,
                "text": "Section 302 IPC punishment for murder and bodily injury intention.",
                "court": "Supreme Court of India",
                "year": 2024,
                "role": "Ratio Decidendi",
                "case_title": "State of Maharashtra vs. Sharma",
                "char_length": 68,
            },
            {
                "chunk_id": f"child-{cls.parent_id_2}-1",
                "parent_id": cls.parent_id_2,
                "text": "Section 304B IPC dowry death and harassment shortly before death.",
                "court": "High Court of Delhi",
                "year": 2022,
                "role": "Facts",
                "case_title": "Rajesh vs. State of Delhi",
                "char_length": 65,
            },
        ]
        cls.vector_store.index_documents(
            cls.mock_children, collection_name=cls.test_col, batch_size=2
        )

    @classmethod
    def tearDownClass(cls):
        # Cleanup Chroma collection and parent JSONs
        try:
            cls.vector_store.chroma_client.delete_collection(name=cls.test_col)
        except Exception:
            pass
        bm25_file = cls.vector_store._get_bm25_path(cls.test_col)
        if bm25_file.exists():
            bm25_file.unlink(missing_ok=True)

        # Remove parent test files
        (cls.settings.parent_store_dir / f"{cls.parent_id_1}.json").unlink(missing_ok=True)
        (cls.settings.parent_store_dir / f"{cls.parent_id_2}.json").unlink(missing_ok=True)

    def test_01_rrf_fusion_ordering(self):
        """Verify Reciprocal Rank Fusion formula order on mock dense and sparse rankings."""
        # Chunk B is rank 2 in dense, rank 1 in BM25 -> Highest combined score
        dense_results = [
            {"chunk_id": "chunk-A", "text": "Doc A"},
            {"chunk_id": "chunk-B", "text": "Doc B"},
        ]
        bm25_results = [
            {"chunk_id": "chunk-B", "text": "Doc B"},
            {"chunk_id": "chunk-C", "text": "Doc C"},
        ]

        fused = self.pipeline.compute_rrf(dense_results, bm25_results, k=60)

        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["chunk_id"], "chunk-B")  # rank 2 + rank 1
        self.assertEqual(fused[1]["chunk_id"], "chunk-A")  # rank 1 only
        self.assertEqual(fused[2]["chunk_id"], "chunk-C")  # rank 2 only

        # Score assertions
        expected_score_b = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
        expected_score_a = 1.0 / (60 + 1)
        self.assertAlmostEqual(fused[0]["rrf_score"], expected_score_b, places=5)
        self.assertAlmostEqual(fused[1]["rrf_score"], expected_score_a, places=5)

    def test_02_cross_encoder_reranker_rescores(self):
        """Verify that Cross-Encoder rescores and prioritizes the most relevant clause."""
        output = self.pipeline.retrieve_hybrid(
            query="murder bodily injury intention Section 302 IPC",
            top_k=2,
            collection_name=self.test_col,
        )

        reranked = output["reranked_candidates"]
        self.assertGreater(len(reranked), 0)
        self.assertEqual(reranked[0]["chunk_id"], f"child-{self.parent_id_1}-1")
        self.assertIn("rerank_score", reranked[0])

    def test_03_parent_document_expansion(self):
        """Verify Parent Expansion loads full JSON context from parent store without duplicate parents."""
        output = self.pipeline.retrieve_hybrid(
            query="Section 302 IPC murder",
            top_k=5,
            collection_name=self.test_col,
        )

        parents = output["retrieved_parents"]
        self.assertGreater(len(parents), 0)
        self.assertEqual(parents[0]["parent_id"], self.parent_id_1)
        self.assertIn("FULL JUDGMENT CONTEXT", parents[0]["text"])

        # Check citations
        citations = output["child_citations"]
        self.assertGreater(len(citations), 0)
        self.assertEqual(citations[0]["case_title"], "State of Maharashtra vs. Sharma")
        self.assertEqual(citations[0]["court"], "Supreme Court of India")

    def test_04_empty_context_fallback(self):
        """Verify fallback when no precedents match the query or filter."""
        result = self.pipeline.generate_answer(
            query="Nonexistent legal doctrine XYZ",
            filters={"court": "Nonexistent Supreme Tribunal of Atlantis"},
            collection_name=self.test_col,
        )

        self.assertIn("Insufficient authoritative precedent", result["answer"])
        self.assertEqual(len(result["retrieved_parents"]), 0)
        self.assertIn("total_latency_sec", result["timings"])

    def test_05_full_pipeline_execution_and_telemetry(self):
        """Verify full execution pipeline generates structured response and timing breakdown."""
        result = self.pipeline.generate_answer(
            query="What are the essential ingredients of murder under Section 302 IPC?",
            top_k=2,
            collection_name=self.test_col,
        )

        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 10)
        self.assertEqual(result["query"], "What are the essential ingredients of murder under Section 302 IPC?")
        self.assertGreater(len(result["retrieved_parents"]), 0)

        # Check timing telemetry breakdown
        timings = result["timings"]
        expected_timing_keys = [
            "metadata_filtering_sec",
            "dense_retrieval_sec",
            "bm25_retrieval_sec",
            "rrf_fusion_sec",
            "reranking_sec",
            "parent_expansion_sec",
            "llm_generation_sec",
            "total_latency_sec",
        ]
        for key in expected_timing_keys:
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0.0)


if __name__ == "__main__":
    unittest.main()
