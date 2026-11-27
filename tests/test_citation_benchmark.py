"""Tests for src/citation_benchmark.py (benchmark v3). Citation formats are the real ones seen in the PDFs:
'[2019] 4 S.C.R. 208' (old SCR page header), '[2025] 11 S.C.R. 197 : 2025 INSC 1210' (e-SCR page 1)."""

import unittest

from src import citation_benchmark as cb
from src import judgment_text as jt
from tests.test_judgment_text import P2025_1, P2025_2, P2025_LAST, pages

CORPUS = [
    {"doc_id": "d_sugreev", "year": 2019, "petitioner": "SUGREEV KUMAR", "respondent": "STATE OF PUNJAB & ORS.",
     "scr_citation": "[2019] 4 S.C.R. 208", "neutral_citation": "2019 INSC 386", "citation": "2019 INSC 386"},
    {"doc_id": "d_thapa", "year": 2025, "petitioner": "SANKAR PADAM THAPA", "respondent": "VIJAYKUMAR DINESHCHANDRA AGARWAL",
     "scr_citation": "[2025] 11 S.C.R. 197", "neutral_citation": "2025 INSC 1210", "citation": "2025 INSC 1210"},
]


class TestCitations(unittest.TestCase):
    def setUp(self):
        self.index = cb.build_corpus_index(CORPUS)

    def test_keys_in_both_scr_spellings(self):
        self.assertEqual(cb.scr_key("[2019] 4 S.C.R. 208"), (2019, 4, 208))
        self.assertEqual(cb.scr_key("[2019] 4 SCR 208"), (2019, 4, 208))
        self.assertEqual(cb.insc_key("2025 INSC 1210"), "2025 INSC 1210")
        self.assertEqual(self.index["scr"][(2025, 11, 197)], "d_thapa")
        self.assertEqual(self.index["insc"]["2019 INSC 386"], "d_sugreev")

    def test_extract_and_map(self):
        text = ("Reliance was placed on Sugreev Kumar v. State of Punjab, [2019] 4 SCR 208, and on "
                "2025 INSC 1210. See also (2014) 8 SCC 273 and AIR 1952 SC 196; and [2015] 3 SCR 999.")
        cites = cb.extract_citations(text)
        self.assertEqual(cites["scr"], [(2019, 4, 208), (2015, 3, 999)])
        self.assertEqual(cites["insc"], ["2025 INSC 1210"])
        self.assertEqual((cites["scc"], cites["air"]), (1, 1))
        targets = cb.map_targets(cites, self.index)
        self.assertEqual({t["doc_id"]: t["match_type"] for t in targets}, {"d_sugreev": "scr", "d_thapa": "neutral"})

    def test_conservative_name_match(self):
        cites = cb.extract_citations("In Sugreev Kumar v. State of Punjab (2019), the Court held ...")
        targets = cb.map_targets(cites, self.index)
        self.assertEqual([(t["doc_id"], t["match_type"]) for t in targets], [("d_sugreev", "name")])
        # wrong year or no year: no match
        self.assertEqual(cb.map_targets(cb.extract_citations("Sugreev Kumar v. State of Punjab (2015)"), self.index), [])
        self.assertEqual(cb.map_targets(cb.extract_citations("Sugreev Kumar v. State of Punjab, decided"), self.index), [])
        self.assertEqual(cb.map_targets(cites, self.index, use_names=False), [])

    def test_party_normalisation(self):
        self.assertEqual(cb.normalize_party("MEHBOOB-UR-REHMAN (D) THR. LRS. & ORS."), "MEHBOOB-UR-REHMAN")
        self.assertEqual(cb.normalize_party("Mehboob-ur-Rehman (Dead) through LRs and another"), "Mehboob-ur-Rehman")
        self.assertEqual(cb.normalize_party("Reliance was placed on Sugreev Kumar", cited=True), "Sugreev Kumar")
        self.assertEqual(cb.normalize_party("In Sugreev Kumar", cited=True), "Sugreev Kumar")

    def test_separators_and_suffixes_still_match(self):
        index = cb.build_corpus_index(CORPUS + [
            {"doc_id": "d_mehboob", "year": 2019, "petitioner": "MEHBOOB-UR-REHMAN (D) THR. LRS.",
             "respondent": "SUDHA SEETHARAM & ANR", "scr_citation": "[2019] 2 S.C.R. 185"}])
        for text in ("Sugreev Kumar vs State of Punjab (2019) 4 SCC 1", "Sugreev Kumar versus State of Punjab (2019)",
                     "Sugreev Kumar v State of Punjab (2019)"):
            with self.subTest(text=text):
                self.assertEqual([t["doc_id"] for t in cb.map_targets(cb.extract_citations(text), index)], ["d_sugreev"])
        text = "as held in Mehboob-ur-Rehman (Dead) through LRs v. Sudha Seetharam & Anr. (2019) 13 SCC 234"
        self.assertEqual([t["doc_id"] for t in cb.map_targets(cb.extract_citations(text), index)], ["d_mehboob"])

    def test_unmatched_names_are_reported(self):
        cites = cb.extract_citations("See Ramesh Chand v. Suresh Lal (2020) 5 SCC 1.")
        self.assertEqual(cb.unmatched_names(cites, self.index), ["Ramesh Chand v. Suresh Lal (2020)"])

    def test_excluded_self_citation(self):
        cites = cb.extract_citations("[2019] 4 SCR 208")
        self.assertEqual(cb.map_targets(cites, self.index, exclude=["d_sugreev"]), [])


class TestQueryText(unittest.TestCase):
    def test_issue_for_consideration_from_real_page(self):
        c = jt.clean_pages(pages(P2025_1, P2025_2, P2025_LAST))
        source, issue = cb.issue_text("", "\n".join(c["lines"]))
        self.assertEqual(source, "issue")
        self.assertTrue(issue.startswith("Whether in the absence of a Trust being made an accused"))
        self.assertTrue(issue.endswith("said Trust."))

    def test_scrub_removes_leaks(self):
        q = ("Whether, in view of Sugreev Kumar v. State of Punjab, [2019] 4 SCR 208, decided on 15 March 2019, the "
             "summons issued to Thapa in Criminal Appeal No. 509 of 2019 under the Negotiable Instruments Act, 1881 survives?")
        clean, removed = cb.scrub(q, parties=["thapa"])
        for leak in ("Sugreev", "[2019]", "2019", "Thapa", "No. 509", "March"):
            self.assertNotIn(leak, clean)
        self.assertIn("Negotiable Instruments Act, 1881", clean)
        self.assertGreaterEqual(removed["citation"], 1)
        self.assertGreaterEqual(removed["case_name"], 1)

    def test_headnote_paragraph_fallback(self):
        source, text = cb.issue_text("HEADNOTES\n\nPenal Code, 1860 – s.302 – Murder – Circumstantial evidence – Chain of "
                                     "circumstances must be complete and unbroken.")
        self.assertEqual(source, "headnote_para")
        self.assertTrue(text.startswith("Penal Code, 1860"))


if __name__ == "__main__":
    unittest.main()
