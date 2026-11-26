"""Offline tests for scripts/download_aws_sc.py helpers (temp dirs only, no network)."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests.test_sc_metadata import TEXT_2019_3, TEXT_2025_1, TITLE_2019_3, TITLE_2025_1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("download_aws_sc", PROJECT_ROOT / "scripts" / "download_aws_sc.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


class TestDownloadHelpers(unittest.TestCase):
    def test_doc_id_prefers_cnr(self):
        self.assertEqual(dl.doc_id_for("ESCR010006352019", "2019_3_400_410"),
                         hashlib.sha1(b"ESCR010006352019").hexdigest()[:16])
        self.assertEqual(dl.doc_id_for("", "2019_3_400_410"), hashlib.sha1(b"2019_3_400_410").hexdigest()[:16])

    def test_keys_and_paths(self):
        self.assertEqual(dl.s3_key(2019, "2019_16_971_992"), "data/pdf/year=2019/english/2019_16_971_992_EN.pdf")
        self.assertEqual(dl.local_pdf_path(Path("/x/sc_aws"), 2019, "2019_3_400_410"),
                         Path("/x/sc_aws/year=2019/2019_3_400_410_EN.pdf"))

    def test_is_valid_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad, empty = Path(tmp, "a.pdf"), Path(tmp, "b.pdf"), Path(tmp, "c.pdf")
            good.write_bytes(b"%PDF-1.7\n...")
            bad.write_bytes(b"<html>AccessDenied</html>")
            empty.write_bytes(b"")
            self.assertTrue(dl.is_valid_pdf(good))
            self.assertFalse(dl.is_valid_pdf(bad))
            self.assertFalse(dl.is_valid_pdf(empty))
            self.assertFalse(dl.is_valid_pdf(Path(tmp, "missing.pdf")))

    def test_existing_valid_file_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp, "year=2019", "x_EN.pdf")
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"%PDF-1.4 test")
            status, nbytes, err = dl.download_one("data/pdf/never-requested.pdf", dest, retries=0, timeout=1)
            self.assertEqual((status, nbytes, err), ("exists", 13, ""))

    def test_pick_citation(self):
        self.assertEqual(dl.pick_citation("2019 INSC 270", "2019 INSC 270", "[2019] 3 S.C.R. 400", None), "2019 INSC 270")
        self.assertEqual(dl.pick_citation("", None, "[2019] 3 S.C.R. 400", None), "[2019] 3 S.C.R. 400")
        self.assertEqual(dl.pick_citation("", None, "", "[2025] 7 S.C.R. 2012"), "[2025] 7 S.C.R. 2012")

    def test_build_row_from_real_records(self):
        raw = {"raw_html": TEXT_2025_1, "title": TITLE_2025_1, "path": "2025_7_2012_2052", "cnr": "ESCR010008282025",
               "judge": "B.V. NAGARATHNA", "decision_date": "29-07-2025", "case_id": "2025 INSC 969",
               "citation": "[2025] 7 S.C.R. 2012"}
        row = dl.build_row(raw, 2025, Path("/x/sc_aws"))
        self.assertEqual(row["doc_id"], dl.doc_id_for("ESCR010008282025", ""))
        self.assertEqual(row["decision_date"], "2025-07-29")
        self.assertEqual(row["judges"], "B.V. NAGARATHNA; K.V. VISWANATHAN")
        self.assertEqual((row["bench_strength"], row["bench_source"], row["bench_type"]), (2, "coram", "Division (2)"))
        self.assertEqual(row["citation"], "2025 INSC 969")
        self.assertEqual(row["s3_key"], "data/pdf/year=2025/english/2025_7_2012_2052_EN.pdf")
        self.assertTrue(row["local_path"].replace("\\", "/").endswith("sc_aws/year=2025/2025_7_2012_2052_EN.pdf"))

        # 2019 fixture has no trailing block, so no case number -> not selected as criminal
        self.assertIsNone(dl.build_row({"raw_html": TEXT_2019_3, "title": TITLE_2019_3}, 2019, Path("/x")))


if __name__ == "__main__":
    unittest.main()
