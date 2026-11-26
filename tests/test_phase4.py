"""Phase 4 (minimal): concordance with required Act names, citation check, bench-strength authority
prior, precedent headers + prompt rules, and SQLite-based filters. Temp dirs only; no LLM calls."""

import unittest
import uuid
from unittest.mock import MagicMock, patch

from src.citation_check import check_citations
from src.concordance import StatutoryConcordance as SC
from src.rag_pipeline import (GROUNDED_LEGAL_SYSTEM_INSTRUCTION, IndianLegalRAGPipeline, bench_authority_multiplier,
                              precedent_header)
from src.sqlite_store import SQLiteParentStore
from src.vector_store import VectorStoreManager
from tests._isolated import isolated_settings


class TestConcordance(unittest.TestCase):
    def test_act_name_required(self):
        # Phase 0 bug: "Section 107 IPC" also added CrPC 107 -> "Section 126 BNSS"
        self.assertEqual(SC.expand_query("abetment under Section 107 IPC"), "abetment under Section 107 IPC")
        self.assertEqual(SC.expand_query("is Section 302 enough"), "is Section 302 enough")

    def test_each_code_maps_within_itself(self):
        self.assertIn("Section 528 BNSS", SC.expand_query("quashing under Section 482 CrPC"))
        self.assertNotIn("BNS ", SC.expand_query("quashing under Section 482 CrPC") + " ")
        self.assertIn("Section 103 BNS", SC.expand_query("murder under Section 302 IPC"))
        self.assertIn("Section 302 IPC", SC.expand_query("murder under Section 103 BNS"))
        self.assertIn("Section 482 CrPC", SC.expand_query("inherent powers, Section 528 BNSS"))
        self.assertIn("Section 307 IPC", SC.expand_query("attempt to murder, Section 109 BNS"))
        self.assertIn("Section 109 BNS", SC.expand_query("attempt to murder, Section 307 IPC"))
        self.assertIn("Section 63 BSA", SC.expand_query("certificate under Section 65B of the Evidence Act"))
        exp = SC.expand_query("FIR under Section 154 CrPC")
        self.assertIn("Section 173 BNSS", exp)
        self.assertNotIn("BSA", exp)  # Evidence Act s.154 must not fire without its Act name

    def test_section_lists_and_act_first(self):
        exp = SC.expand_query("Sections 302 and 34 IPC")
        self.assertIn("Section 103 BNS", exp)
        self.assertIn("Section 3(5) BNS", exp)
        self.assertIn("Section 103 BNS", SC.expand_query("IPC Section 302"))

    def test_no_duplicate_when_already_present(self):
        q = "Section 302 IPC and Section 103 BNS"
        self.assertEqual(SC.expand_query(q).count("Section 103 BNS"), 1)


class TestCitationCheck(unittest.TestCase):
    PRECEDENTS = [{"citation": "2023 INSC 675", "scr_citation": "[2023] 11 S.C.R. 34"}, {"citation": "2019 INSC 386"}]

    def test_all_valid(self):
        cc = check_citations("Held in [Precedent 1] and [Precedent 2].\n\nVerify on the official source.", self.PRECEDENTS)
        self.assertEqual(cc["citation_precision"], 1.0)
        self.assertEqual(cc["valid_precedents"], [1, 2])
        self.assertTrue(cc["ends_with_verify"])

    def test_unknown_number_and_external_citation(self):
        cc = check_citations("See [Precedent 3] and (2014) 8 SCC 273; also 2023 INSC 675 [Precedent 1].", self.PRECEDENTS)
        self.assertEqual(cc["unknown_precedent_numbers"], [3])
        self.assertEqual(cc["unknown_external_citations"], ["(2014) 8 SCC 273"])
        # valid: [Precedent 1] + known external "2023 INSC 675"; all citations: 2 tags + 2 external
        self.assertEqual(cc["citation_precision"], 0.5)
        self.assertFalse(cc["ends_with_verify"])

    def test_grouped_tags_and_refusal(self):
        self.assertEqual(check_citations("[Precedents 1, 2]", self.PRECEDENTS)["valid_precedents"], [1, 2])
        cc = check_citations("Insufficient precedent in the indexed corpus.\n\nVerify on the official source.", [])
        self.assertTrue(cc["refused"])
        self.assertIsNone(cc["citation_precision"])


class TestAuthorityAndPrompt(unittest.TestCase):
    def test_bench_ordering(self):
        m = [bench_authority_multiplier(n) for n in (7, 5, 4, 3, 2, 1)]
        self.assertEqual(m, sorted(m, reverse=True))
        self.assertGreater(bench_authority_multiplier(5), bench_authority_multiplier(3))
        self.assertGreater(bench_authority_multiplier(3), bench_authority_multiplier(2))
        self.assertGreater(bench_authority_multiplier(2), bench_authority_multiplier(1))
        self.assertEqual(bench_authority_multiplier(None), 1.0)

    def test_factor_bounded_and_switchable(self):
        tmp, s = isolated_settings()
        try:
            pipe = IndianLegalRAGPipeline.__new__(IndianLegalRAGPipeline)
            pipe.settings = s.model_copy(update={"authority_strength": 0.1, "authority_max_bonus": 0.05})
            f5 = pipe._authority_factor(bench_authority_multiplier(5))
            self.assertTrue(1.0 < f5 <= 1.05)
            self.assertEqual(pipe._authority_factor(bench_authority_multiplier(1)), 1.0)
            self.assertEqual(pipe._authority_factor(bench_authority_multiplier(5), use_authority=False), 1.0)
            pipe.settings = s.model_copy(update={"authority_strength": 0.0})
            self.assertEqual(pipe._authority_factor(bench_authority_multiplier(5)), 1.0)
        finally:
            tmp.cleanup()

    def test_precedent_header(self):
        h = precedent_header(2, {"case_title": "SUGREEV KUMAR versus STATE OF PUNJAB & ORS.", "citation": "2019 INSC 386",
                                 "decision_date": "2019-03-15", "bench_type": "Division (2)", "bench_strength": 2,
                                 "judges": "ABHAY MANOHAR SAPRE; DINESH MAHESHWARI", "author_judge": "DINESH MAHESHWARI",
                                 "disposal_nature": "Case Partly allowed", "role": "Ratio/Analysis"})
        self.assertEqual(h, "[Precedent 2] SUGREEV KUMAR versus STATE OF PUNJAB & ORS. | 2019 INSC 386 | 2019-03-15 | "
                            "Division (2) [2 judges]: ABHAY MANOHAR SAPRE; DINESH MAHESHWARI | author: DINESH MAHESHWARI | "
                            "disposal: Case Partly allowed | role: Ratio/Analysis")
        self.assertIn("author: not recorded", precedent_header(1, {"case_title": "X"}))

    def test_prompt_rules(self):
        p = GROUNDED_LEGAL_SYSTEM_INSTRUCTION
        for rule in ("[Precedent N]", "ratio decidendi", "obiter dicta", "larger benches", "conflict",
                     "1 July 2024", "Insufficient precedent in the indexed corpus.", "Verify on the official source."):
            self.assertIn(rule, p)


class MockCrossEncoder:
    def predict(self, pairs):
        return [0.5 for _ in pairs]  # equal relevance: only the authority prior decides


class TestFiltersAndPrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.settings = isolated_settings(authority_strength=0.1, authority_max_bonus=0.05)
        cls.col = f"test_phase4_{uuid.uuid4().hex[:6]}"
        cls.vs = VectorStoreManager(settings=cls.settings, collection_name=cls.col)
        cls.pipe = IndianLegalRAGPipeline(settings=cls.settings, vector_store=cls.vs)
        cls.pipe._reranker = MockCrossEncoder()
        store = SQLiteParentStore(cls.settings.sqlite_path_for(cls.col))
        docs = [("dA", 2019, 2), ("dB", 2021, 3), ("dC", 2024, 5)]
        store.upsert_documents([{"doc_id": d, "year": y, "bench_strength": b, "case_title": f"Case {d}"} for d, y, b in docs])
        chunks = []
        for d, y, b in docs:
            pid = f"{d}-p00000"
            store.insert_parents_batch([{"parent_id": pid, "doc_id": d, "case_title": f"Case {d}",
                                         "text": f"Parent text of {d} about bail under Section 439 CrPC.",
                                         "metadata": {"doc_id": d, "year": y, "bench_strength": b, "case_title": f"Case {d}",
                                                      "citation": f"{y} INSC {b}", "role": "Ratio/Analysis"}}])
            chunks.append({"chunk_id": f"{pid}-c000", "parent_id": pid, "doc_id": d, "year": y, "bench_strength": b,
                           "case_title": f"Case {d}", "role": "Ratio/Analysis",
                           "text": "Bail under Section 439 CrPC: personal liberty and the triple test."})
        cls.vs.index_documents(chunks, collection_name=cls.col, batch_size=10)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.vs.chroma_client.delete_collection(name=cls.col)
        except Exception:
            pass
        cls.tmp.cleanup()

    def docs(self, out):
        return [c["metadata"]["doc_id"] for c in out["reranked_candidates"]]

    def test_min_bench_and_year_filters(self):
        out = self.pipe.retrieve_hybrid("bail Section 439 CrPC", top_k=5, filters={"min_bench": 3})
        self.assertEqual(sorted(self.docs(out)), ["dB", "dC"])
        self.assertEqual(out["filters"]["matching_docs"], 2)
        out = self.pipe.retrieve_hybrid("bail Section 439 CrPC", top_k=5, filters={"year_from": 2020, "year_to": 2022})
        self.assertEqual(self.docs(out), ["dB"])

    def test_filter_without_matches_refuses(self):
        out = self.pipe.retrieve_hybrid("bail", top_k=5, filters={"year_from": 2030})
        self.assertEqual(out["reranked_candidates"], [])
        fake = MagicMock()
        with patch("src.rag_pipeline.get_budget", return_value=fake):
            res = self.pipe.generate_answer("bail", filters={"year_from": 2030})
        fake.generate.assert_not_called()
        self.assertTrue(res["citation_check"]["refused"])

    def test_prior_breaks_ties_by_bench_and_can_be_disabled(self):
        out = self.pipe.retrieve_hybrid("bail Section 439 CrPC", top_k=3)
        self.assertEqual(self.docs(out), ["dC", "dB", "dA"])  # 5 > 3 > 2 judges at equal relevance
        off = self.pipe.retrieve_hybrid("bail Section 439 CrPC", top_k=3, use_authority=False)
        self.assertEqual({c["authority_factor"] for c in off["reranked_candidates"]}, {1.0})

    def test_answer_has_headers_and_citation_check(self):
        fake = MagicMock()
        fake.generate.return_value = "Bail [Precedent 1].\n\nVerify on the official source."
        with patch("src.rag_pipeline.get_budget", return_value=fake):
            res = self.pipe.generate_answer("bail Section 439 CrPC", top_k=3)
        prompt = fake.generate.call_args.kwargs["contents"]
        self.assertIn("[Precedent 1] Case dC | 2024 INSC 5 |", prompt)
        self.assertEqual(res["citation_check"]["citation_precision"], 1.0)
        self.assertEqual(len(res["precedents"]), 3)


if __name__ == "__main__":
    unittest.main()
