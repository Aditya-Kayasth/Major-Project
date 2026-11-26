"""Tests for the formatting helpers in scripts/aws_sc_explore.py.

Regression: in a DataFrame built from parsed records, missing optional fields come back as
NaN (a truthy float), so `value or ''` does not apply and slicing crashed fmt_example.
No I/O, no network.
"""

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from src.sc_metadata import parse_record
from tests.test_sc_metadata import TEXT_2019_3, TEXT_2025_1, TITLE_2019_3, TITLE_2025_1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("aws_sc_explore", PROJECT_ROOT / "scripts" / "aws_sc_explore.py")
explore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(explore)


def _frame() -> pd.DataFrame:
    """Three parsed records, built into a DataFrame the way main() does: full, no author '*', nothing parsed."""
    full = dict(parse_record(TEXT_2025_1, TITLE_2025_1), year=2025, path="2025_7_2012_2052")
    no_author = dict(parse_record(TEXT_2019_3, TITLE_2019_3), year=2019, path="2019_17_405_421")
    empty = dict(parse_record(None), year=2019, path="2019_x", coram_context="(no 'Coram' in text) ...")
    return pd.DataFrame([full, no_author, empty]).reset_index(drop=True)


class TestExploreFormatting(unittest.TestCase):
    def test_missing_fields_are_null_in_the_frame(self):
        df = _frame()
        self.assertTrue(pd.isna(df.iloc[2]["subject_hint"]))
        self.assertTrue(pd.isna(df.iloc[1]["author_judge_coram"]))
        self.assertTrue(pd.isna(df.iloc[2]["bench_strength"]))
        self.assertTrue(pd.isna(df.iloc[0]["coram_context"]))  # key absent in that record

    def test_fmt_example_with_missing_fields(self):
        df = _frame()
        full = explore.fmt_example(df.iloc[0])
        self.assertIn("B.V. NAGARATHNA; K.V. VISWANATHAN", full)
        self.assertIn("author=B.V. NAGARATHNA", full)
        self.assertIn("Penal Code, 1860", full)

        no_author = explore.fmt_example(df.iloc[1])
        self.assertIn("author= |", no_author)
        self.assertNotIn("nan", no_author)

        empty = explore.fmt_example(df.iloc[2])
        self.assertIn("2019_x", empty)
        self.assertIn("| ? |", empty)
        self.assertNotIn("nan", empty)

    def test_fmt_failure_with_missing_fields(self):
        df = _frame()
        line = explore.fmt_failure(df.iloc[2])
        self.assertIn("no_coram", line)
        self.assertIn("coram=0 declared=-", line)
        report = explore.fmt_failure_report(df.iloc[2])
        self.assertNotIn("nan", report)
        self.assertIn("declared=2", explore.fmt_failure(df.iloc[0]))

    def test_txt_num_positions(self):
        self.assertEqual(explore.txt(None), "")
        self.assertEqual(explore.txt(float("nan")), "")
        self.assertEqual(explore.txt(pd.NA), "")
        self.assertEqual(explore.txt(["A", "B"]), "A; B")
        self.assertEqual(explore.num(2.0), "2")
        self.assertEqual(explore.num(float("nan")), "")
        # Duplicate index labels (e.g. after concatenating years) do not matter: positions are used
        self.assertEqual(explore.positions(pd.Series([True, False, True], index=[5, 5, 7])), [0, 2])
        dup = pd.concat([_frame(), _frame()])
        self.assertEqual(list(dup.index[:3]), [0, 1, 2])
        self.assertIn("2019_x", explore.fmt_example(dup.reset_index(drop=True).iloc[5]))


if __name__ == "__main__":
    unittest.main()
