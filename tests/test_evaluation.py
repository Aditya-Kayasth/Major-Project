"""Unit tests for Phase 4 Evaluation and Benchmarking Suite.

Verifies Hit@k, MRR computation, latency statistics, case matching heuristics,
and schema integrity of golden_queries.json.
"""

import json
import unittest
from pathlib import Path

from evaluate_pipeline import (
    compute_latency_metrics,
    compute_retrieval_metrics,
    is_case_match,
)
from src.config import get_settings


class TestEvaluationSuite(unittest.TestCase):
    """Test suite for evaluation metric formulas and benchmark components."""

    def test_golden_queries_schema(self):
        """Validate structure and minimum size of golden_queries.json."""
        settings = get_settings()
        queries_file = settings.benchmarks_dir / "golden_queries.json"
        self.assertTrue(queries_file.exists(), "golden_queries.json must exist")

        with open(queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        self.assertGreaterEqual(len(queries), 10, "Must contain at least 10 queries")

        required_keys = {
            "query_id",
            "query",
            "target_case_title",
            "expected_statutes",
            "target_role",
            "ground_truth_answer",
        }
        for q in queries:
            self.assertTrue(required_keys.issubset(set(q.keys())))
            self.assertIsInstance(q["expected_statutes"], list)
            self.assertGreater(len(q["query"]), 10)
            self.assertGreater(len(q["target_case_title"]), 3)
            self.assertGreater(len(q["ground_truth_answer"]), 20)

    def test_retrieval_metrics_perfect_ranking(self):
        """Verify Hit@k and MRR when all queries hit rank 1."""
        rankings = [1, 1, 1, 1]
        metrics = compute_retrieval_metrics(rankings)

        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertEqual(metrics["hit@5"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_retrieval_metrics_mixed_ranking(self):
        """Verify metric calculation math on a known distribution of ranks."""
        # 4 queries: rank 1, rank 2, rank 3, not in top 5 (None)
        rankings = [1, 2, 3, None]
        metrics = compute_retrieval_metrics(rankings)

        # Hit@1: 1 / 4 = 0.25
        self.assertEqual(metrics["hit@1"], 0.25)
        # Hit@3: 3 / 4 = 0.75
        self.assertEqual(metrics["hit@3"], 0.75)
        # Hit@5: 3 / 4 = 0.75
        self.assertEqual(metrics["hit@5"], 0.75)
        # MRR: (1/1 + 1/2 + 1/3 + 0) / 4 = (1.0 + 0.5 + 0.33333 + 0) / 4 = 1.83333 / 4 = 0.4583
        expected_mrr = round((1.0 + 0.5 + (1.0 / 3.0) + 0.0) / 4.0, 4)
        self.assertEqual(metrics["mrr"], expected_mrr)

    def test_retrieval_metrics_out_of_bounds_ranks(self):
        """Verify that ranks beyond top 5 (rank 6, 7, None) yield zero hits and zero MRR."""
        rankings = [6, 7, None, None]
        metrics = compute_retrieval_metrics(rankings)

        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["hit@3"], 0.0)
        self.assertEqual(metrics["hit@5"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_latency_metrics_percentiles(self):
        """Verify mean, p50, and p95 calculations."""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        metrics = compute_latency_metrics(latencies)

        self.assertEqual(metrics["mean_ms"], 30.0)
        self.assertEqual(metrics["p50_ms"], 30.0)
        self.assertAlmostEqual(metrics["p95_ms"], 48.0, delta=1.0)

    def test_is_case_match_heuristics(self):
        """Verify case title matching logic against various candidate representations."""
        target = "Government of India vs. ISRO Drivers Association (2020 INSC 484)"

        # Chunk with exact match in metadata
        hit_1 = {
            "case_title": "Government of India vs. ISRO Drivers Association",
            "metadata": {"court": "Supreme Court of India"},
            "text": "Civil Appeal No. 7138 of 2010 reported in 2020 INSC 484",
        }
        self.assertTrue(is_case_match(hit_1, target))

        # Chunk with citation in text
        hit_2 = {
            "case_title": "2020 INSC 484",
            "metadata": {"source_path": "data/SPACE/2020 INSC 484.pdf"},
            "text": "Judicial order regarding ISRO Drivers Association",
        }
        self.assertTrue(is_case_match(hit_2, target))

        # Irrelevant chunk
        hit_3 = {
            "case_title": "State of Maharashtra vs. Sharma",
            "metadata": {"court": "High Court of Bombay"},
            "text": "Trial under Section 302 IPC.",
        }
        self.assertFalse(is_case_match(hit_3, target))


if __name__ == "__main__":
    unittest.main()
