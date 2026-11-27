"""Presentation-layer citation normalisation in server.py: 【 or [, "Precedent(s)", number(s), then optionally
† ‡ : | or "," and any label up to 】 or ] -> "[Precedent N, <label>]" / "[Precedent N]", and the existing
citation check run on it (it counts only the precedent numbers).

server.py builds the full pipeline (models, Chroma, BM25) at import time, so these tests do NOT import it:
only the regex, the dash table and the functions are taken out of its source with `ast` and executed alone."""

import ast
import re
import unittest
from pathlib import Path

from src.citation_check import check_citations

_NAMES = {"_DASHES", "CITATION_FORM_RE", "_rewrite_citations", "normalize_answer_citations", "citations_for_check"}


def _load_from_server() -> dict:
    src = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    nodes = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in _NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _NAMES for t in node.targets):
            nodes.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "server.py", "exec"), ns)
    missing = _NAMES - set(ns)
    if missing:
        raise RuntimeError(f"not found in server.py: {sorted(missing)}")
    return ns


_NS = _load_from_server()
normalize = _NS["normalize_answer_citations"]
for_check = _NS["citations_for_check"]

NBH = "‑"  # non-breaking hyphen, as the model writes it
PRECEDENTS = [{"citation": "2023 INSC 675"}, {"citation": "2019 INSC 386"}, {"citation": "2021 INSC 100"},
              {"citation": "2024 INSC 200"}, {"citation": "2022 INSC 300"}]


def check(answer: str) -> dict:
    """What server.py does: normalise for display, then run the existing check on the label-free form."""
    return check_citations(for_check(normalize(answer)), PRECEDENTS)


class TestNormalizeAnswerCitations(unittest.TestCase):
    def test_exact_strings_from_real_answers(self):
        self.assertEqual(normalize("【Precedent 1†Passage 2】"), "[Precedent 1, Passage 2]")
        self.assertEqual(normalize(f"【Precedent 1†Para 55{NBH}56】"), "[Precedent 1, Para 55-56]")
        self.assertEqual(normalize(f"【Precedent 3†Para 63{NBH}64】"), "[Precedent 3, Para 63-64]")
        self.assertEqual(normalize("【Precedent 1†Headnote】"), "[Precedent 1, Headnote]")

    def test_exact_strings_get_cited(self):
        # before the fix the check saw no tag at all: "precision n/a, cited 0/5"
        raw = f"【Precedent 1†Para 55{NBH}56】 【Precedent 3†Para 63{NBH}64】 【Precedent 1†Headnote】"
        self.assertEqual(check_citations(raw, PRECEDENTS)["valid_precedents"], [])
        cc = check(raw)
        self.assertEqual(cc["valid_precedents"], [1, 3])
        self.assertEqual(cc["unknown_precedent_numbers"], [])
        self.assertEqual(cc["citation_precision"], 1.0)

    def test_every_separator_and_dash(self):
        self.assertEqual(normalize("【Precedent 2‡Facts】"), "[Precedent 2, Facts]")
        self.assertEqual(normalize("[Precedent 2: Ratio/Analysis]"), "[Precedent 2, Ratio/Analysis]")
        self.assertEqual(normalize("[Precedent 2 | Order]"), "[Precedent 2, Order]")
        self.assertEqual(normalize("[Precedent 2†Passage 1]"), "[Precedent 2, Passage 1]")
        for dash in ("‐", "‑", "–", "—"):
            self.assertEqual(normalize(f"【Precedent 4†Para 1{dash}2】"), "[Precedent 4, Para 1-2]")
        self.assertEqual(normalize("【Precedent 2†】"), "[Precedent 2]")

    def test_mixed_forms_in_one_answer(self):
        answer = (f"Bail is the rule 【Precedent 1†Passage 2】 and 【Precedent 1†Para 55{NBH}56】. "
                  f"The bar applies 【Precedent 3】 and 【Precedent 3†Para 63{NBH}64】; see the headnote "
                  "【Precedent 1†Headnote】, also [Precedent 2†Passage 1], [Precedent 4] and "
                  "[Precedent 5: Order]; both lines agree [Precedents 1, 3].")
        expected = ("Bail is the rule [Precedent 1, Passage 2] and [Precedent 1, Para 55-56]. "
                    "The bar applies [Precedent 3] and [Precedent 3, Para 63-64]; see the headnote "
                    "[Precedent 1, Headnote], also [Precedent 2, Passage 1], [Precedent 4] and "
                    "[Precedent 5, Order]; both lines agree [Precedents 1, 3].")
        norm = normalize(answer)
        self.assertEqual(norm, expected)
        for ch in ("【", "】", "†", NBH):
            self.assertNotIn(ch, norm)
        cc = check(answer)
        self.assertEqual(cc["valid_precedents"], [1, 2, 3, 4, 5])
        self.assertEqual(cc["unknown_precedent_numbers"], [])
        self.assertEqual(cc["citation_precision"], 1.0)

    def test_check_counts_only_the_number(self):
        self.assertEqual(for_check("[Precedent 1, Para 55-56] and [Precedent 3, Headnote]"),
                         "[Precedent 1] and [Precedent 3]")
        self.assertEqual(for_check("[Precedents 1, 3]"), "[Precedents 1, 3]")  # a number list is not a label

    def test_unknown_number_still_flagged(self):
        cc = check("See 【Precedent 7†Headnote】 and 【Precedent 1†Para 2】.")
        self.assertEqual(cc["valid_precedents"], [1])
        self.assertEqual(cc["unknown_precedent_numbers"], [7])
        self.assertEqual(cc["citation_precision"], 0.5)

    def test_plain_form_unchanged_and_idempotent(self):
        for text in ("Held in [Precedent 1].", "[Precedents 1 and 3]", "[Precedents 1, 2]",
                     "[Precedent 2, Passage 1]", "[Precedent 1, Para 55-56]", ""):
            self.assertEqual(normalize(text), text)
        once = normalize(f"【Precedent 1†Para 55{NBH}56】")
        self.assertEqual(normalize(once), once)

    def test_none_is_safe(self):
        self.assertEqual(normalize(None), "")


if __name__ == "__main__":
    unittest.main()
