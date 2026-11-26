"""Per-collection SQLite stores, the documents table, checkpoint and row selection (temp dirs only)."""

import math
import unittest

import index_corpus
from src.sqlite_store import SQLiteParentStore
from src.vector_store import sanitize_metadata_for_chroma
from tests._isolated import isolated_settings

DOC = {"doc_id": "d1", "case_title": "A v. B", "court": "Supreme Court of India", "decision_date": "2023-08-02",
       "year": 2023, "judges": "X; Y", "bench_strength": 2, "bench_type": "Division (2)", "reportable": True,
       "author_pdf_agrees": False, "statutes_cited": "IPC 302", "headnote": "Issue for consideration ..."}


class TestStores(unittest.TestCase):
    def setUp(self):
        self.tmp, self.s = isolated_settings()

    def tearDown(self):
        self.tmp.cleanup()

    def test_documents_table(self):
        store = SQLiteParentStore(self.s.sqlite_path_for("c1"))
        store.upsert_documents([DOC, dict(DOC, doc_id="d2", year=2019, bench_strength=3)])
        self.assertEqual(store.count_documents(), 2)
        row = store.get_document("d1")
        self.assertEqual((row["year"], row["bench_strength"], row["reportable"], row["author_pdf_agrees"]), (2023, 2, 1, 0))
        self.assertIsNotNone(row["indexed_at"])
        self.assertEqual(sorted(store.doc_ids_where(min_bench=3)), ["d2"])
        self.assertEqual(sorted(store.doc_ids_where(year_from=2020)), ["d1"])

    def test_collections_are_isolated(self):
        pilot = SQLiteParentStore(self.s.sqlite_path_for("sc_criminal_cpu_pilot"))
        full = SQLiteParentStore(self.s.sqlite_path_for("sc_criminal_cpu"))
        parent = {"parent_id": "d1-p00000", "doc_id": "d1", "case_title": "A v. B", "text": "t",
                  "metadata": {"court": "SC", "year": 2023, "role": "Facts"}}
        pilot.insert_parents_batch([parent])
        pilot.upsert_documents([DOC])
        full.insert_parents_batch([parent])
        pilot.clear()  # --reset of the pilot
        self.assertEqual((pilot.count_parents(), pilot.count_documents()), (0, 0))
        self.assertEqual(full.count_parents(), 1)

    def test_checkpoint_per_collection(self):
        a = index_corpus.CheckpointManager(self.s.checkpoint_path_for("a"))
        a.mark_indexed("d1", 10, 3)
        a.save()
        self.assertTrue(index_corpus.CheckpointManager(self.s.checkpoint_path_for("a")).is_indexed("d1"))
        self.assertFalse(index_corpus.CheckpointManager(self.s.checkpoint_path_for("b")).is_indexed("d1"))
        a.reset()
        self.assertFalse(index_corpus.CheckpointManager(self.s.checkpoint_path_for("a")).is_indexed("d1"))


class TestRowSelection(unittest.TestCase):
    ROWS = [{"doc_id": f"d{i}", "year": 2018 + i // 100, "path": f"p{i:04d}", "status": "ok"} for i in range(800)] + \
           [{"doc_id": "missing", "year": 2023, "path": "zz", "status": "not_found"}]

    def test_spread_covers_all_years(self):
        sel = index_corpus.select_rows(self.ROWS, 20)
        self.assertEqual(len(sel), 20)
        self.assertEqual({r["year"] for r in sel}, set(range(2018, 2026)))
        self.assertNotIn("missing", [r["doc_id"] for r in sel])

    def test_all_available_without_limit(self):
        self.assertEqual(len(index_corpus.select_rows(self.ROWS, None)), 800)

    def test_clean_row(self):
        row = index_corpus.clean_row({"a": float("nan"), "b": 2.0, "c": "x"})
        self.assertIsNone(row["a"])
        self.assertEqual(row["b"], 2.0)


class TestChromaMetadata(unittest.TestCase):
    def test_none_values_are_omitted(self):
        meta = sanitize_metadata_for_chroma({"bench_strength": None, "reportable": True, "year": 2023, "judges": ["A", "B"]})
        self.assertNotIn("bench_strength", meta)
        self.assertEqual(meta, {"reportable": True, "year": 2023, "judges": "A; B"})


if __name__ == "__main__":
    unittest.main()
