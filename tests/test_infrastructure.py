"""Tests for Phase 3 Step 2 infrastructure: config defaults/paths, process-tree RAM guard,
BGE query prefix, lazy LLM config. No model downloads, no network, temp dirs only."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.local_embeddings import BGE_QUERY_INSTRUCTION, LocalEmbeddingEngine, resolve_query_instruction
from src.llm_budget import gen_config
from src.resource_guard import ResourceGuard
from src.vector_store import sanitize_metadata_for_chroma
from tests._isolated import isolated_settings


def _proc(rss_bytes, children=()):
    p = MagicMock()
    p.memory_info.return_value = SimpleNamespace(rss=rss_bytes)
    p.children.return_value = list(children)
    return p


class TestConfig(unittest.TestCase):
    def test_per_collection_paths_are_separate(self):
        tmp, s = isolated_settings()
        try:
            self.assertNotEqual(s.sqlite_path_for("sc_criminal_cpu_pilot"), s.sqlite_path_for("sc_criminal_cpu"))
            self.assertNotEqual(s.checkpoint_path_for("sc_criminal_cpu_pilot"), s.checkpoint_path_for("sc_criminal_cpu"))
            self.assertEqual(s.sqlite_path_for("x").name, "x.sqlite")
            self.assertEqual(s.checkpoint_path_for("x").name, "checkpoint_x.json")
            self.assertEqual(s.manifest_path.name, "manifest.parquet")
            self.assertTrue(s.logs_dir.exists())
        finally:
            tmp.cleanup()

    def test_isolated_settings_have_no_llm_keys(self):
        tmp, s = isolated_settings()
        try:
            self.assertEqual(s.llm_api_key, "")
            self.assertEqual(s.gemini_api_key, "")
            self.assertTrue(str(s.chroma_db_dir).startswith(tmp.name))
        finally:
            tmp.cleanup()


class TestResourceGuard(unittest.TestCase):
    def test_process_tree_rss_includes_children(self):
        gb = 1024 ** 3
        parent = _proc(2 * gb, children=[_proc(1 * gb), _proc(int(0.5 * gb))])
        self.assertEqual(ResourceGuard.process_tree_rss_bytes(parent), int(3.5 * gb))

    def test_vanished_child_is_skipped(self):
        import psutil

        dead = MagicMock()
        dead.memory_info.side_effect = psutil.NoSuchProcess(pid=1)
        parent = _proc(100, children=[dead, _proc(50)])
        self.assertEqual(ResourceGuard.process_tree_rss_bytes(parent), 150)

    def test_guard_raises_over_limit_and_tracks_peak(self):
        guard = ResourceGuard(max_ram_gb=1.0, warn_ram_gb=0.5)
        with patch.object(ResourceGuard, "process_tree_rss_bytes", return_value=2 * 1024 ** 3):
            with self.assertRaises(Exception):
                guard.enforce_guardrails("test")
        self.assertGreaterEqual(guard.peak_ram_gb, 2.0)

    def test_storage_walk_disabled_and_throttled(self):
        self.assertEqual(ResourceGuard(max_storage_gb=0, storage_check_paths=["/"]).get_monitored_storage_gb(), 0.0)
        tmp, s = isolated_settings()
        try:
            g = ResourceGuard(max_storage_gb=100, storage_check_paths=[s.chroma_db_dir], storage_check_every=3)
            with patch("src.resource_guard.os.walk", return_value=iter([])) as walk:
                for _ in range(7):
                    g.get_monitored_storage_gb()
            self.assertEqual(walk.call_count, 3)  # calls 1, 4, 7
        finally:
            tmp.cleanup()


class TestQueryPrefix(unittest.TestCase):
    def test_auto_prefix_for_bge_english(self):
        self.assertEqual(resolve_query_instruction("BAAI/bge-small-en-v1.5"), BGE_QUERY_INSTRUCTION)
        self.assertEqual(resolve_query_instruction("BAAI/bge-m3"), "")
        self.assertEqual(resolve_query_instruction("zeroentropy/zembed-1"), "")
        self.assertEqual(resolve_query_instruction("BAAI/bge-small-en-v1.5", ""), "")
        self.assertEqual(resolve_query_instruction("any", "query: "), "query: ")

    def test_prefix_only_on_queries(self):
        engine = LocalEmbeddingEngine("BAAI/bge-small-en-v1.5", device="cpu")  # not loaded
        self.assertEqual(engine.prepare_texts(["bail"], is_query=True), [BGE_QUERY_INSTRUCTION + "bail"])
        self.assertEqual(engine.prepare_texts(["bail"], is_query=False), ["bail"])

    def test_model_query_prompt_wins(self):
        # Server finding: the smoke check failed when the model config ships its own 'query' prompt,
        # because prepare_texts() then returned the text unchanged. The prompt is now always explicit.
        engine = LocalEmbeddingEngine("BAAI/bge-small-en-v1.5", device="cpu")
        engine._st_model = SimpleNamespace(prompts={"query": "q: ", "document": ""})
        self.assertEqual(engine.prepare_texts(["bail"], is_query=True), ["q: bail"])
        self.assertEqual(engine.prepare_texts(["bail"], is_query=False), ["bail"])
        self.assertEqual(engine.prompt_source(True), "model_config")

    def test_empty_model_prompt_falls_back_to_instruction(self):
        engine = LocalEmbeddingEngine("BAAI/bge-small-en-v1.5", device="cpu")
        engine._st_model = SimpleNamespace(prompts={"query": "", "document": ""})
        self.assertEqual(engine.effective_prompt(True), BGE_QUERY_INSTRUCTION)
        self.assertEqual(engine.effective_prompt(False), "")
        self.assertEqual(engine.prompt_source(True), "query_instruction")

    def test_encode_gets_explicit_prompt(self):
        class FakeST:
            prompts = {}

            def __init__(self):
                self.calls = []

            def encode(self, texts, **kw):
                import numpy as np

                self.calls.append(kw.get("prompt"))
                return np.ones((len(texts), 3))

        engine = LocalEmbeddingEngine("BAAI/bge-small-en-v1.5", device="cpu")
        engine._st_model, engine._initialized = FakeST(), True
        engine.embed_texts(["bail"], is_query=True)
        engine.embed_texts(["a judgment chunk"], is_query=False)
        self.assertEqual(engine._st_model.calls, [BGE_QUERY_INSTRUCTION, ""])

    def test_cpu_dtype_is_fp32(self):
        import torch

        self.assertEqual(LocalEmbeddingEngine("BAAI/bge-small-en-v1.5", device="cpu")._dtype(), torch.float32)


class TestRerankerRelevance(unittest.TestCase):
    def test_no_double_sigmoid(self):
        from src.rag_pipeline import reranker_relevances

        # predict() output already in [0,1] (activation_fn=Sigmoid): used as-is
        self.assertEqual(reranker_relevances([0.0, 0.12, 0.8, 1.0]), [0.0, 0.12, 0.8, 1.0])
        # a batch of raw logits gets exactly one sigmoid, on every value (consistent ordering)
        out = reranker_relevances([3.0, 0.5, -2.0])
        self.assertAlmostEqual(out[0], 1 / (1 + math.exp(-3.0)))
        self.assertAlmostEqual(out[1], 1 / (1 + math.exp(-0.5)))
        self.assertAlmostEqual(out[2], 1 / (1 + math.exp(2.0)))
        self.assertEqual(out, sorted(out, reverse=True))

    def test_ordering_preserved_with_bounded_prior(self):
        from src.rag_pipeline import reranker_relevances

        # Without the double sigmoid, a 0.10 relevance gap survives the maximum +5% bonus on the weaker chunk
        strong, weak = reranker_relevances([0.90, 0.80])
        self.assertGreater(strong * 1.0, weak * 1.05)


class TestMisc(unittest.TestCase):
    def test_lists_joined_with_semicolon(self):
        self.assertEqual(sanitize_metadata_for_chroma({"judges": ["A", "B"], "x": None})["judges"], "A; B")

    def test_gen_config_exposes_fields(self):
        cfg = gen_config(system_instruction="sys", temperature=0.1, response_mime_type="application/json")
        self.assertEqual(getattr(cfg, "system_instruction"), "sys")
        self.assertEqual(getattr(cfg, "response_mime_type"), "application/json")


if __name__ == "__main__":
    unittest.main()
