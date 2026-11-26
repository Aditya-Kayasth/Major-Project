"""Tests for src/benchmark_v2.py (metrics, bootstrap, leakage filter, allocation) and the judge-provider switch."""

import math
import unittest

from src import benchmark_v2 as bv


class TestMetrics(unittest.TestCase):
    def test_single_target(self):
        m = bv.query_metrics(["a", "b", "c"], ["b"])
        self.assertEqual((m["hit@1"], m["hit@3"], m["hit@10"]), (0.0, 1.0, 1.0))
        self.assertAlmostEqual(m["mrr@10"], 0.5)
        self.assertAlmostEqual(m["ndcg@10"], 1 / math.log2(3))
        self.assertEqual(m["recall@10"], 1.0)
        self.assertEqual(m["rank"], 2)

    def test_miss_and_cutoff(self):
        m = bv.query_metrics([f"d{i}" for i in range(20)], ["d12"])
        self.assertEqual((m["hit@10"], m["mrr@10"], m["ndcg@10"], m["rank"]), (0.0, 0.0, 0.0, None))

    def test_multiple_targets(self):
        m = bv.query_metrics(["x", "a", "b"], ["a", "b"])
        idcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(m["ndcg@10"], (1 / math.log2(3) + 1 / math.log2(4)) / idcg)
        self.assertEqual(m["recall@10"], 1.0)

    def test_bootstrap_deterministic_and_bounds(self):
        vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        ci = bv.bootstrap_ci(vals, seed=42)
        self.assertEqual(ci, bv.bootstrap_ci(vals, seed=42))
        self.assertTrue(ci[0] <= 0.7 <= ci[1])
        d = bv.paired_bootstrap_delta([1, 1, 0, 1], [0, 1, 0, 0], seed=42)
        self.assertEqual(d["delta"], 0.5)

    def test_summarize_and_latency(self):
        rows = [bv.query_metrics(["a"], ["a"]), bv.query_metrics(["b"], ["a"])]
        s = bv.summarize(rows)
        self.assertEqual(s["hit@1"]["value"], 0.5)
        self.assertEqual(s["n"], 2)
        self.assertEqual(bv.latency_stats([10, 20, 30])["p50"], 20.0)

    def test_full_ranking_fills_from_fused(self):
        self.assertEqual(bv.full_ranking(["c", "a"], ["a", "b", "c", "d"], k=3), ["c", "a", "b"])


class TestLeakage(unittest.TestCase):
    def test_party_and_judge_names(self):
        parties = bv.party_tokens("SUGREEV KUMAR versus STATE OF PUNJAB & ORS.")
        self.assertEqual(parties, ["kumar", "sugreev"])
        self.assertIn("party_name:sugreev", bv.leakage_reasons("Was Sugreev entitled to summon them?", parties))
        judges = bv.judge_tokens("ABHAY MANOHAR SAPRE; DINESH MAHESHWARI")
        self.assertTrue(bv.leakage_reasons("as held by Justice Maheshwari", [], judges))

    def test_case_numbers_citations_dates(self):
        self.assertIn("case_number", bv.leakage_reasons("In Criminal Appeal No. 509 of 2019 ..."))
        self.assertIn("citation", bv.leakage_reasons("as laid down in 2019 INSC 386"))
        self.assertIn("date", bv.leakage_reasons("the FIR lodged on 15 March 2019"))
        self.assertTrue(any(r.startswith("year") for r in bv.leakage_reasons("an incident in 2016")))

    def test_clean_legal_queries_pass(self):
        for q in ("When can a court summon additional accused under Section 319 CrPC?",
                  "Is a conviction under Sections 302/34 IPC sustainable on circumstantial evidence alone?",
                  "Does Section 27 of the Indian Evidence Act, 1872 apply to a disclosure made in police custody?"):
            self.assertEqual(bv.leakage_reasons(q), [], q)


class TestAllocationAndTemplates(unittest.TestCase):
    def test_proportional_allocation(self):
        counts = {2018: 185, 2019: 264, 2020: 132, 2021: 196, 2022: 244, 2023: 248, 2024: 273, 2025: 347}
        alloc = bv.allocate_by_year(counts, 100)
        self.assertEqual(sum(alloc.values()), 100)
        self.assertEqual(max(alloc, key=alloc.get), 2025)
        self.assertEqual(sum(bv.allocate_by_year(counts, 50).values()), 50)
        self.assertEqual(bv.allocate_by_year({2018: 2}, 10), {2018: 2})

    def test_out_of_scope_and_human_template(self):
        self.assertEqual(len(bv.OUT_OF_SCOPE_QUERIES), 5)
        tpl = bv.human_template()
        self.assertEqual(len(tpl), 20)
        self.assertEqual(tpl[0]["query_id"], "HQ-01")
        self.assertEqual(tpl[0]["query"], "")


class TestJudgeProviderSwitch(unittest.TestCase):
    def test_providers_restricts_chain(self):
        from src.llm_budget import GeminiBudget
        from tests._isolated import isolated_settings

        tmp, s = isolated_settings(llm_api_key="dummy")
        try:
            b = GeminiBudget(s)
            calls = []
            b._generate_compat = lambda *a, **k: calls.append("openai") or None
            b._generate_gemini = lambda *a, **k: calls.append("gemini") or "ok"
            self.assertEqual(b.generate(client=object(), models=["m"], contents="x", use_cache=False,
                                        providers=["gemini"]), "ok")
            self.assertEqual(calls, ["gemini"])
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
