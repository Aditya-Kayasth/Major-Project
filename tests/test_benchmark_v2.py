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

    def test_exclude_self_before_scoring(self):
        ranking = bv.exclude_and_cut(["self", "a", "b"], ["self"], k=2)
        self.assertEqual(ranking, ["a", "b"])
        self.assertEqual(bv.query_metrics(ranking, ["a"])["hit@1"], 1.0)  # self no longer occupies rank 1

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


def _context(n=5, passage_len=3000):
    """Same shape as rag_pipeline.build_context: one block per judgment, header + passages."""
    return "\n\n".join(f"[Precedent {i}] Case {i} | {2019 + i} INSC {i} | 2020 | Division Bench (2 judges): A; B | "
                       f"author: A | disposal: Dismissed | role: Ratio/Analysis\nPassage 1 (role: Ratio/Analysis):\n"
                       + ("text " * (passage_len // 5)) for i in range(1, n + 1))


class TestJudgeContext(unittest.TestCase):
    """Bug seen on the server: the judge got context[:6000] and called [Precedent 4]/[Precedent 5] non-existent."""

    def test_all_precedents_survive_shortening(self):
        ctx = _context(5, 3000)  # ~16,000 chars
        fit = bv.fit_context(ctx, max_chars=6000)
        self.assertEqual(fit["n_precedents"], 5)
        self.assertTrue(fit["shortened"])
        for i in range(1, 6):
            self.assertIn(f"[Precedent {i}] Case {i} |", fit["text"])  # every header kept in full
        self.assertLess(len(fit["text"]), len(ctx))

    def test_short_context_is_unchanged(self):
        ctx = _context(3, 200)
        self.assertEqual(bv.fit_context(ctx, max_chars=24000), {"text": ctx, "n_precedents": 3, "shortened": False})

    def test_judge_prompt_states_valid_range(self):
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("evaluate_v2", root / "scripts" / "evaluate_v2.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        jp = ev.judge_prompt("Q?", _context(5, 3000), "Answer citing [Precedent 5].")
        self.assertEqual(jp["n_precedents"], 5)
        self.assertIn("Precedents 1..5", jp["prompt"])
        self.assertIn("[Precedent 5] Case 5 |", jp["prompt"])


class TestGroqJudgeRoute(unittest.TestCase):
    """Gemini free tier exhausted: the judge runs on Groq with ONLY --judge-model (never the generator)."""

    def test_only_the_judge_model_is_called(self):
        from unittest.mock import MagicMock, patch

        from src.llm_budget import GeminiBudget
        from tests._isolated import isolated_settings

        tmp, s = isolated_settings(llm_api_key="dummy", llm_model="openai/gpt-oss-120b",
                                   llm_fallback_models=["openai/gpt-oss-20b"], llm_rpm=1000.0)
        try:
            b = GeminiBudget(s)
            ok = MagicMock(status_code=200, headers={})
            ok.json.return_value = {"choices": [{"message": {"content": '{"groundedness": 9}'}}]}
            with patch("httpx.post", return_value=ok) as post:
                out = b.generate(client=object(), models=[], contents="judge this", use_cache=False,
                                 providers=["openai"], compat_models=["qwen/qwen3.8-27b"])
            self.assertEqual(out, '{"groundedness": 9}')
            self.assertEqual([c.kwargs["json"]["model"] for c in post.call_args_list], ["qwen/qwen3.8-27b"])
        finally:
            tmp.cleanup()

    def test_rate_limited_judge_does_not_fall_back_to_generator(self):
        from unittest.mock import MagicMock, patch

        from src.llm_budget import GeminiBudget
        from tests._isolated import isolated_settings

        tmp, s = isolated_settings(llm_api_key="dummy", llm_model="openai/gpt-oss-120b",
                                   llm_fallback_models=["openai/gpt-oss-20b"], llm_rpm=1000.0)
        try:
            b = GeminiBudget(s)
            bad = MagicMock(status_code=400, headers={}, text="bad request")
            with patch("httpx.post", return_value=bad) as post:
                out = b.generate(client=None, models=[], contents="judge this", use_cache=False,
                                 providers=["openai"], compat_models=["qwen/qwen3.8-27b"], max_retries=0)
            self.assertIsNone(out)
            self.assertTrue(all(c.kwargs["json"]["model"] == "qwen/qwen3.8-27b" for c in post.call_args_list))
        finally:
            tmp.cleanup()


class TestGeneratorAttribution(unittest.TestCase):
    CHAIN = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

    def test_log_trace(self):
        log = [
            "[gen] 20 queries (5 out-of-scope, 10 legal, 5 facts); 0 already done; route ...",
            "[gen] OOS-01 out_of_scope | low_conf True | precision None | refused True | llm 0.0s | wall 1.2s",
            "[gen] LQ-001 legal_issue | low_conf False | precision 1.0 | refused False | llm 3.1s | wall 5.0s",
            "'openai/gpt-oss-120b' rate-limited (429); trying the next model.",
            "[gen] LQ-002 legal_issue | low_conf False | precision 1.0 | refused False | llm 9.0s | wall 11.0s",
            "'openai/gpt-oss-120b' rate-limited (429); trying the next model.",
            "LLM API error 400 on 'qwen/qwen3.8-27b': bad request",
            "[gen] LQ-003 legal_issue | low_conf False | precision 1.0 | refused False | llm 9.0s | wall 11.0s",
            "LLM API asked to wait 300s (quota likely exhausted): ...",
            "[gen] LQ-004 legal_issue | low_conf False | precision 1.0 | refused False | llm 9.0s | wall 11.0s",
        ]
        out = bv.attribute_generators_from_log(log, self.CHAIN)
        self.assertEqual(out["OOS-01"]["model"], None)
        self.assertEqual(out["LQ-001"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(out["LQ-002"]["model"], "qwen/qwen3.8-27b")
        self.assertEqual(out["LQ-003"]["model"], "openai/gpt-oss-20b")
        self.assertEqual(out["LQ-004"]["model"], "unknown")  # old message without a model name

    def test_judge_exclusion(self):
        answers = {"A": {"generator_model": "openai/gpt-oss-120b"}, "B": {"generator_model": "qwen/qwen3.8-27b"},
                   "C": {"generator_model": "unknown"}}
        rows = [{"query_id": q, "judge_model": "qwen/qwen3.8-27b", "scores": {"groundedness": 8}} for q in "ABC"] + \
               [{"query_id": "D", "judge_model": "qwen/qwen3.8-27b", "scores": None}]
        split = bv.judge_exclusion(rows, answers)
        self.assertEqual([r["query_id"] for r in split["excluded"]], ["B"])
        self.assertEqual([r["query_id"] for r in split["unknown"]], ["C"])
        self.assertEqual(sorted(r["query_id"] for r in split["included"]), ["A", "C"])

    def test_llm_budget_records_the_fallback_model(self):
        from unittest.mock import MagicMock, patch

        from src.llm_budget import GeminiBudget
        from tests._isolated import isolated_settings

        tmp, s = isolated_settings(llm_api_key="dummy", llm_model="openai/gpt-oss-120b",
                                   llm_fallback_models=["qwen/qwen3.8-27b"], llm_rpm=1000.0)
        try:
            b = GeminiBudget(s)
            limited = MagicMock(status_code=429, headers={"retry-after": "60"}, text="rate limit")
            ok = MagicMock(status_code=200, headers={})
            ok.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
            with patch("httpx.post", side_effect=[limited, ok]):
                self.assertEqual(b.generate(client=None, models=[], contents="q", providers=["openai"]), "answer")
            self.assertEqual(b.last_model, "qwen/qwen3.8-27b")
            self.assertFalse(b.last_from_cache)
            b2 = GeminiBudget(s)  # new process: the cache meta file still knows who answered
            self.assertEqual(b2.generate(client=None, models=[], contents="q", providers=["openai"]), "answer")
            self.assertEqual((b2.last_model, b2.last_from_cache), ("qwen/qwen3.8-27b", True))
        finally:
            tmp.cleanup()


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
