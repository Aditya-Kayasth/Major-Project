"""Unit tests for configuration validation and hierarchical legal chunking."""

import json
import unittest
from pathlib import Path

from src.config import Settings, get_settings
from src.document_loader import LegalDocumentLoader


class TestLegalRAGComponents(unittest.TestCase):
    """Test suite for Phase 1 ingestion and hierarchical chunking modules."""

    def test_settings_loaded_and_validated(self):
        settings = Settings.validate()
        self.assertIsInstance(settings, Settings)
        self.assertGreater(settings.parent_chunk_size, 0)
        self.assertGreater(settings.child_chunk_size, 0)
        self.assertLess(settings.child_chunk_size, settings.parent_chunk_size)
        self.assertLess(settings.chunk_overlap, settings.child_chunk_size)
        self.assertTrue(settings.parent_store_dir.exists())
        self.assertTrue(settings.raw_data_dir.exists())

    def test_hierarchical_legal_chunking(self):
        loader = LegalDocumentLoader()
        sample_judgment = (
            "# Facts\n"
            "The appellant was convicted under Section 302 of the Indian Penal Code by the Sessions Court.\n\n"
            "# Submissions\n"
            "Learned senior counsel for the appellant submitted that there are material contradictions in witness testimony.\n\n"
            "# Precedents Cited\n"
            "The High Court relied on the decision reported in AIR 2020 SC 154.\n\n"
            "# Ratio Decidendi\n"
            "We hold that mere suspicion cannot take the place of proof beyond reasonable doubt in circumstantial evidence.\n\n"
            "# Order\n"
            "The appeal is allowed and the judgment of conviction is set aside."
        )
        fake_path = "data/raw_judgments/State_vs_Sharma_2024.pdf"

        parents, children = loader.create_parent_child_chunks(sample_judgment, fake_path)

        # Assertions on parents
        self.assertGreater(len(parents), 0)
        for parent in parents:
            self.assertIn("parent_id", parent)
            self.assertIn("case_title", parent)
            self.assertIn("text", parent)
            self.assertIn("metadata", parent)
            # Verify JSON persistence in parent store
            parent_file = loader.settings.parent_store_dir / f"{parent['parent_id']}.json"
            self.assertTrue(parent_file.exists())
            with open(parent_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data["parent_id"], parent["parent_id"])

        # Assertions on children
        self.assertGreater(len(children), 0)
        roles = set()
        for child in children:
            self.assertIn("parent_id", child)
            self.assertIn("chunk_id", child)
            self.assertIn("case_title", child)
            self.assertIn("court", child)
            self.assertIn("year", child)
            self.assertIn("role", child)
            self.assertIn("char_length", child)
            self.assertIn("text", child)
            self.assertEqual(child["char_length"], len(child["text"]))
            roles.add(child["role"])

        # Ensure legal roles were detected
        self.assertTrue(any(r in roles for r in ["Facts", "Submissions", "Precedents Cited", "Ratio Decidendi", "Order"]))

    def test_record_failed_file(self):
        """Verify that failed file metadata is appended to data/processed/failed_files.json."""
        loader = LegalDocumentLoader()
        fake_path = Path("data/raw_judgments/corrupted_scan.pdf")
        loader._record_failed_file(fake_path, "OCR processing failed after 3 retries")

        failed_file_path = loader.settings.processed_data_dir / "failed_files.json"
        self.assertTrue(failed_file_path.exists())
        with open(failed_file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertGreater(len(records), 0)
        latest = records[-1]
        self.assertEqual(latest["filename"], "corrupted_scan.pdf")
        self.assertIn("OCR processing failed", latest["error"])

    def test_record_scanned_for_gpu_ocr(self):
        """Verify that scanned PDFs needing GPU/background OCR are queued in scanned_for_gpu_ocr.json."""
        loader = LegalDocumentLoader()
        fake_path = Path("data/raw_judgments/scanned_appeal_2024.pdf")
        loader._record_scanned_for_gpu_ocr(fake_path, "503 High Demand Fast-Skip")

        queue_file = loader.settings.processed_data_dir / "scanned_for_gpu_ocr.json"
        self.assertTrue(queue_file.exists())
        with open(queue_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertGreater(len(records), 0)
        matching = [r for r in records if r["filename"] == "scanned_appeal_2024.pdf"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["reason"], "503 High Demand Fast-Skip")

    def test_disk_caching(self):
        """Verify that successful document text is cached to data/processed/{stem}.txt and loaded instantly."""
        loader = LegalDocumentLoader()
        test_txt = "IN THE SUPREME COURT OF INDIA\nArticle 21 Constitutional Ruling test content."
        fake_file = loader.settings.raw_data_dir / "test_cache_doc.txt"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fake_file, "w", encoding="utf-8") as f:
            f.write(test_txt)

        # First load: writes cache
        text1, meta1, was_ocr1 = loader.load_document(fake_file)
        self.assertEqual(text1, test_txt)
        cache_file = loader.settings.processed_data_dir / "test_cache_doc.txt"
        self.assertTrue(cache_file.exists())

        # Second load: reads cache directly
        text2, meta2, was_ocr2 = loader.load_document(fake_file)
        self.assertEqual(text2, test_txt)
        self.assertTrue(meta2.get("cached", False))

        # Cleanup
        fake_file.unlink(missing_ok=True)
        cache_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
