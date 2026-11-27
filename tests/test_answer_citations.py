"""Presentation-layer citation normalisation in server.py: 【Precedent N†Passage M】 / 【Precedent N】 /
[Precedent N†Passage M] -> "[Precedent N, passage M]" / "[Precedent N]", and the existing citation check run on it.

server.py builds the full pipeline (models, Chroma, BM25) at import time, so these tests do NOT import it:
only the two regexes and the two functions are taken out of its source with `ast` and executed alone."""

import ast
import re
import unittest
from pathlib import Path

from src.citation_check import check_citations

_NAMES = {"CITATION_FORM_RE", "PASSAGE_SUFFIX_RE", "normalize_answer_citations", "citations_for_check"}


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

PRECEDENTS = [{"citation": "2023 INSC 675"}, {"citation": "2019 INSC 386"},
              {"citation": "2021 INSC 100"}, {"citation": "2024 INSC 200"}]


class TestNormalizeAnswerCitations(unittest.TestCase):
    def test_exact_bug_string(self):
        self.assertEqual(normalize("【Precedent 1†Passage 2】"), "[Precedent 1, passage 2]")

    def test_bug_string_gets_cited(self):
        # before the fix the check saw no tag at all: "precision n/a, cited 0"
        self.assertEqual(check_citations("【Precedent 1†Passage 2】", PRECEDENTS)["valid_precedents"], [])
        cc = check_citations(for_check(normalize("Held in 【Precedent 1†Passage 2】.")), PRECEDENTS)
        self.assertEqual(cc["valid_precedents"], [1])
        self.assertEqual(cc["citation_precision"], 1.0)

    def test_mixed_forms_in_one_answer(self):
        answer = ("Bail is the rule 【Precedent 1†Passage 2】. The bar applies 【Precedent 3】, "
                  "see also [Precedent 2†Passage 1] and [Precedent 4]; both lines agree [Precedents 1, 3].")
        expected = ("Bail is the rule [Precedent 1, passage 2]. The bar applies [Precedent 3], "
                    "see also [Precedent 2, passage 1] and [Precedent 4]; both lines agree [Precedents 1, 3].")
        norm = normalize(answer)
        self.assertEqual(norm, expected)
        self.assertNotIn("【", norm)
        self.assertNotIn("†", norm)
        cc = check_citations(for_check(norm), PRECEDENTS)
        self.assertEqual(cc["valid_precedents"], [1, 2, 3, 4])
        self.assertEqual(cc["unknown_precedent_numbers"], [])
        self.assertEqual(cc["citation_precision"], 1.0)

    def test_unknown_number_still_flagged(self):
        cc = check_citations(for_check(normalize("See 【Precedent 7†Passage 1】.")), PRECEDENTS)
        self.assertEqual(cc["unknown_precedent_numbers"], [7])

    def test_plain_form_unchanged_and_idempotent(self):
        for text in ("Held in [Precedent 1].", "[Precedents 1 and 3]", "[Precedent 2, passage 1]", ""):
            self.assertEqual(normalize(text), text)
        once = normalize("【Precedent 1†Passage 2】")
        self.assertEqual(normalize(once), once)

    def test_for_check_strips_passage_only(self):
        self.assertEqual(for_check("[Precedent 1, passage 2] and [Precedent 3]"), "[Precedent 1] and [Precedent 3]")

    def test_none_is_safe(self):
        self.assertEqual(normalize(None), "")


if __name__ == "__main__":
    unittest.main()
