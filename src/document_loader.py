"""Document loading, Multimodal OCR fallback, and Hierarchical Legal Chunking.

Phase 1 Ingestion Pipeline specialized for Indian Supreme Court, High Court,
and District Court legal judgments, orders, and case records.
"""

import datetime
import json
import logging
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pypdf
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tqdm import tqdm

from src.config import Settings, get_settings

# Configure logger
logger = logging.getLogger("IndianLegalRAG.DocumentLoader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Multimodal OCR System Prompt for Gemini
LEGAL_OCR_SYSTEM_INSTRUCTION = """You are a specialized Multimodal Legal OCR & Document Parsing Engine for Indian Legal Documents (Supreme Court, High Courts, and District Courts).
Your mission is to accurately transcribe and structurally normalize the provided judicial document:
1. Strict Verbatim Transcription: Faithfully transcribe all text. For regional Indian scripts (Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Urdu, Kannada, etc.), transcribe the original script verbatim, followed immediately by its accurate English translation in square brackets [English translation: ...].
2. Courtroom Tables & Schedules: Convert all courtroom tables, charge sheets, witness lists, sentencing matrices, maintenance calculation tables, and property schedules into clean, valid GitHub-flavored Markdown tables.
3. Structural Legal Headings: Segment the document logically using clear Markdown headings:
   - # Facts
   - # Submissions
   - # Precedents Cited
   - # Ratio Decidendi
   - # Order
   (Include relevant subheadings as needed, preserving legal paragraph numbering).
4. Preserve Citations: Retain all legal citations (AIR, SCC, SCR, ILR, INSC) exactly as printed without alteration."""


class LegalDocumentLoader:
    """Production document loader with Multimodal OCR fallback and Hierarchical Legal Chunking."""

    def __init__(self, settings: Optional[Settings] = None, enable_ocr: bool = True):
        self.settings = settings or get_settings()
        self.enable_ocr = enable_ocr
        self.client: Optional[genai.Client] = None
        if self.settings.gemini_api_key and self.settings.gemini_api_key != "your_gemini_api_key_here":
            self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def _get_gemini_client(self) -> genai.Client:
        """Ensure authenticated Gemini Client is available."""
        if not self.client:
            if not self.settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not configured in environment.")
            self.client = genai.Client(api_key=self.settings.gemini_api_key)
        return self.client

    def extract_digital_pdf(self, file_path: Path) -> str:
        """Extract text from digital PDF using pypdf."""
        try:
            reader = pypdf.PdfReader(str(file_path))
            extracted_pages: List[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    extracted_pages.append(text.strip())
            return "\n\n".join(extracted_pages)
        except Exception as err:
            logger.warning(f"Digital PDF extraction failed for {file_path.name}: {err}")
            return ""

    def extract_via_gemini_ocr(self, file_path: Path) -> str:
        """Perform fast-fail multimodal OCR on scanned PDF/image via Gemini Files API with auto-cleanup."""
        client = self._get_gemini_client()
        uploaded_file = None
        logger.info(f"Triggering Gemini Multimodal OCR fallback for: {file_path.name}")

        try:
            # Upload file to Gemini Files API
            uploaded_file = client.files.upload(file=str(file_path))

            # Try configured model once, then ONE alternate candidate (gemini-3.5-flash / gemini-3.8-flash)
            primary_model = self.settings.gemini_model
            alternate_model = "gemini-3.5-flash" if primary_model != "gemini-3.5-flash" else "gemini-3.8-flash"
            candidate_models = [primary_model, alternate_model]

            extracted_text = ""
            last_exception = None

            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            uploaded_file,
                            "Extract and structure the complete legal document following the system instructions.",
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=LEGAL_OCR_SYSTEM_INSTRUCTION,
                            temperature=0.1,
                            http_options=types.HttpOptions(timeout=20000),  # Strict 20s timeout
                        ),
                    )
                    if response and response.text:
                        extracted_text = response.text
                        break
                except APIError as api_err:
                    last_exception = api_err
                    err_msg = str(api_err)
                    logger.warning(
                        f"OCR model {model_name} failed with API error ({api_err.code}): {err_msg[:90]}. "
                        "Fast-failing to next candidate without sleep..."
                    )
                    # Do not sleep for minutes on 503/429; immediately try alternate or abort
                    continue
                except Exception as err:
                    last_exception = err
                    logger.warning(f"OCR model {model_name} failed with error: {err}")
                    continue

            if not extracted_text:
                raise RuntimeError(
                    f"Fast-fail OCR aborted across candidates ({candidate_models}) for {file_path.name}. "
                    f"Last error: {last_exception}"
                )

            return extracted_text

        finally:
            # Enforce deletion of uploaded remote file in finally block
            if uploaded_file and hasattr(uploaded_file, "name"):
                try:
                    client.files.delete(name=uploaded_file.name)
                    logger.debug(f"Deleted remote Gemini file: {uploaded_file.name}")
                except Exception as del_err:
                    logger.warning(f"Failed to delete uploaded file {uploaded_file.name}: {del_err}")

    def load_document(
        self, file_path: Union[str, Path], allow_ocr: Optional[bool] = None
    ) -> Tuple[str, Dict[str, Any], bool]:
        """Load document with immediate disk caching, sparsity detection, and OCR fallback.

        Returns:
            Tuple of (extracted_text, metadata, was_ocr_fallback_used)
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target document does not exist: {path}")

        effective_ocr = self.enable_ocr if allow_ocr is None else allow_ocr
        start_time = time.perf_counter()
        cache_file = self.settings.processed_data_dir / f"{path.stem}.txt"

        # Step 0: Immediate Disk Cache Check (read in 0.001s)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8", errors="replace") as f:
                    cached_text = f.read()
                if len(cached_text.strip()) > 0:
                    elapsed = time.perf_counter() - start_time
                    logger.info(
                        f"Loaded cached text for {path.name} from disk cache in {elapsed:.4f}s"
                    )
                    metadata = self._extract_legal_metadata(cached_text, path)
                    metadata["cached"] = True
                    metadata["ocr_used"] = False
                    metadata["extraction_time_sec"] = round(elapsed, 4)
                    return cached_text, metadata, False
            except Exception as e:
                logger.warning(f"Failed reading disk cache for {path.name}: {e}")

        # Step 0b: Check if previously marked as scanned for GPU OCR
        scanned_queue_file = self.settings.processed_data_dir / "scanned_for_gpu_ocr.json"
        if scanned_queue_file.exists():
            try:
                with open(scanned_queue_file, "r", encoding="utf-8") as f:
                    queued_records = json.load(f)
                queued_names = {r.get("filename") for r in queued_records} | {r.get("file_path") for r in queued_records}
                if path.name in queued_names or str(path) in queued_names:
                    return "", {}, False
            except Exception:
                pass

        suffix = path.suffix.lower()
        extracted_text = ""
        was_ocr = False

        if suffix == ".pdf":
            extracted_text = self.extract_digital_pdf(path)
            # Sparse text check (< 50 characters)
            if len(extracted_text.strip()) < 50:
                if not effective_ocr:
                    logger.info(
                        f"Sparse digital text ({len(extracted_text.strip())} chars) in {path.name}. "
                        "OCR disabled: fast-skipping."
                    )
                    return "", {}, False
                logger.info(
                    f"Sparse digital text ({len(extracted_text.strip())} chars) detected in {path.name}. "
                    f"Routing to Multimodal OCR."
                )
                extracted_text = self.extract_via_gemini_ocr(path)
                was_ocr = True
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            if not effective_ocr:
                return "", {}, False
            extracted_text = self.extract_via_gemini_ocr(path)
            was_ocr = True
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                extracted_text = f.read()

        elapsed = time.perf_counter() - start_time

        # Step 1: Immediate Disk Caching for future runs
        if len(extracted_text.strip()) > 0:
            self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
            except Exception as cache_err:
                logger.warning(f"Could not write cache file {cache_file.name}: {cache_err}")

        logger.info(
            f"Extracted {len(extracted_text)} chars from {path.name} in {elapsed:.2f}s "
            f"(OCR: {was_ocr})"
        )

        metadata = self._extract_legal_metadata(extracted_text, path)
        metadata["ocr_used"] = was_ocr
        metadata["cached"] = False
        metadata["extraction_time_sec"] = round(elapsed, 4)

        return extracted_text, metadata, was_ocr

    def _extract_legal_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """Extract legal domain metadata heuristics (Court, Title, Year, Case Type)."""
        filename = file_path.name
        stem = file_path.stem

        # 1. Category from folder name (CIVIL, CRIMINAL, SPACE)
        parent_dir_name = file_path.parent.name.upper()
        case_category = "GENERAL"
        if "CIVIL" in parent_dir_name:
            case_category = "CIVIL"
        elif "CRIMINAL" in parent_dir_name:
            case_category = "CRIMINAL"
        elif "SPACE" in parent_dir_name:
            case_category = "SPACE"

        # 2. Case Title Extraction
        case_title = stem.replace("_", " ").strip()
        # Clean common Indian Kanoon / archive patterns like "... on 12 May 2023"
        clean_title = re.sub(r"\s+on\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}", "", case_title, flags=re.IGNORECASE)
        # Check text for standard versus pattern: [Party A] v. [Party B]
        first_chunk = text[:2000]
        title_match = re.search(
            r"([A-Z][A-Za-z\s\.\,\&]{3,80})\s+(?:v\.|vs\.|VERSUS)\s+([A-Z][A-Za-z\s\.\,\&]{3,80})",
            first_chunk,
        )
        if title_match:
            p1 = " ".join(title_match.group(1).split())
            p2 = " ".join(title_match.group(2).split())
            extracted_title = f"{p1} vs. {p2}"
            if len(extracted_title) > 10 and not clean_title.startswith("20"):
                clean_title = extracted_title

        # 3. Court Detection
        court = "High Court"
        upper_text = first_chunk.upper()
        if "SUPREME COURT OF INDIA" in upper_text or "INSC" in stem.upper():
            court = "Supreme Court of India"
        elif "DELHI HIGH COURT" in upper_text or "HIGH COURT OF DELHI" in upper_text:
            court = "High Court of Delhi"
        elif "BOMBAY HIGH COURT" in upper_text or "HIGH COURT OF BOMBAY" in upper_text:
            court = "High Court of Bombay"
        elif "KARNATAKA HIGH COURT" in upper_text or "HIGH COURT OF KARNATAKA" in upper_text:
            court = "High Court of Karnataka"
        elif "MADRAS HIGH COURT" in upper_text or "HIGH COURT OF JUDICATURE AT MADRAS" in upper_text:
            court = "High Court of Judicature at Madras"
        elif "CENTRAL ADMINISTRATIVE TRIBUNAL" in upper_text or "TRIBUNAL" in upper_text:
            court = "Central Administrative Tribunal"

        # 4. Year Extraction
        year: Optional[int] = None
        # Try finding year in stem first
        stem_year = re.findall(r"\b(?:19|20)\d{2}\b", stem)
        if stem_year:
            year = int(stem_year[-1])
        else:
            # Try text
            text_years = re.findall(r"\b(?:19|20)\d{2}\b", first_chunk)
            if text_years:
                year = int(text_years[0])

        return {
            "source_path": str(file_path),
            "filename": filename,
            "case_title": clean_title,
            "case_category": case_category,
            "court": court,
            "year": year,
        }

    def _detect_legal_role(self, chunk_text: str, current_heading: Optional[str]) -> str:
        """Classify chunk legal role (Ratio Decidendi, Facts, Submissions, Precedents Cited, Order, Obiter Dicta)."""
        # 1. Heading-based detection
        if current_heading:
            h = current_heading.upper()
            if "RATIO" in h or "HELD" in h or "FINDING" in h:
                return "Ratio Decidendi"
            if "FACT" in h:
                return "Facts"
            if "SUBMISSION" in h or "CONTENTION" in h or "ARGUMENT" in h:
                return "Submissions"
            if "PRECEDENT" in h or "AUTHORIT" in h or "CASE LAW" in h:
                return "Precedents Cited"
            if "ORDER" in h or "DISPOSAL" in h or "CONCLUSION" in h:
                return "Order"

        # 2. Text linguistic heuristic cues
        sample = chunk_text[:400].lower()
        if any(w in sample for w in ["we hold that", "held that", "it is settled law", "question of law", "ratio decidendi"]):
            return "Ratio Decidendi"
        if any(w in sample for w in ["factual matrix", "prosecution case", "fir no", "allegations in", "complaint filed"]):
            return "Facts"
        if any(w in sample for w in ["learned counsel submitted", "contended that", "learned senior counsel argued"]):
            return "Submissions"
        if any(w in sample for w in ["in the case of", "relied on", "air 19", "air 20", "scc", "scr"]):
            return "Precedents Cited"
        if any(w in sample for w in ["appeal is allowed", "appeal is dismissed", "decreed", "bail is granted", "disposed of"]):
            return "Order"

        return "Obiter Dicta"

    def create_parent_child_chunks(
        self, text: str, source_path: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Implement production Hierarchical Legal Chunking.

        1. Parent Chunks: Split text into coherent legal sections (~1800 characters)
           and persist each to data/parent_store/{parent_id}.json.
        2. Child Chunks: Subdivide Parent Chunks into granular chunks (~450 characters, 80 char overlap)
           tagged with parent_id, chunk_id, case_title, court, year, role, and char_length.

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        path = Path(source_path)
        base_meta = self._extract_legal_metadata(text, path)
        case_title = base_meta["case_title"]
        court = base_meta["court"]
        year = base_meta["year"]

        # Ensure parent store directory exists
        parent_store_dir = self.settings.parent_store_dir
        parent_store_dir.mkdir(parents=True, exist_ok=True)

        parent_size = self.settings.parent_chunk_size
        child_size = self.settings.child_chunk_size
        overlap = self.settings.chunk_overlap

        # Split document into paragraphs / structural blocks
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [text.strip()] if text.strip() else [""]

        # Build Parent Chunks respecting paragraph boundaries
        parent_blocks: List[str] = []
        current_block: List[str] = []
        current_len = 0

        for para in raw_paragraphs:
            para_len = len(para)
            if current_len + para_len > parent_size and current_block:
                parent_blocks.append("\n\n".join(current_block))
                current_block = [para]
                current_len = para_len
            else:
                current_block.append(para)
                current_len += para_len + 2

        if current_block:
            parent_blocks.append("\n\n".join(current_block))

        parent_chunks: List[Dict[str, Any]] = []
        child_chunks: List[Dict[str, Any]] = []

        active_heading: Optional[str] = None

        for p_idx, p_text in enumerate(parent_blocks):
            parent_id = str(uuid.uuid4())

            # Detect structural heading if parent starts with #
            heading_match = re.match(r"^(#+\s+[^\n]+)", p_text)
            if heading_match:
                active_heading = heading_match.group(1).replace("#", "").strip()

            parent_chunk_data = {
                "parent_id": parent_id,
                "case_title": case_title,
                "text": p_text,
                "metadata": {
                    **base_meta,
                    "parent_index": p_idx,
                    "char_length": len(p_text),
                    "structural_heading": active_heading or "General",
                },
            }

            # Save full parent section context to local disk
            parent_file = parent_store_dir / f"{parent_id}.json"
            with open(parent_file, "w", encoding="utf-8") as f:
                json.dump(parent_chunk_data, f, ensure_ascii=False, indent=2)

            parent_chunks.append(parent_chunk_data)

            # Subdivide Parent Chunk into Child Chunks with sliding window overlap
            child_start = 0
            p_len = len(p_text)

            while child_start < p_len:
                child_end = min(child_start + child_size, p_len)
                c_text = p_text[child_start:child_end].strip()

                if c_text:
                    role = self._detect_legal_role(c_text, active_heading)
                    child_id = str(uuid.uuid4())

                    child_chunk_data = {
                        "parent_id": parent_id,
                        "chunk_id": child_id,
                        "case_title": case_title,
                        "court": court,
                        "year": year,
                        "role": role,
                        "char_length": len(c_text),
                        "text": c_text,
                        "source_path": str(path),
                        "parent_index": p_idx,
                    }
                    child_chunks.append(child_chunk_data)

                if child_end >= p_len:
                    break
                child_start += child_size - overlap

        return parent_chunks, child_chunks

    def process_file(
        self, file_path: Union[str, Path], allow_ocr: Optional[bool] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Ingest, extract (with OCR fallback if needed), and hierarchically chunk a legal file."""
        path = Path(file_path).resolve()
        start_t = time.perf_counter()

        text, meta, was_ocr = self.load_document(path, allow_ocr=allow_ocr)
        if not text or not text.strip():
            return [], []
        parents, children = self.create_parent_child_chunks(text, str(path))

        elapsed = time.perf_counter() - start_t
        logger.info(
            f"Finished {path.name}: {len(parents)} parents, {len(children)} children "
            f"in {elapsed:.2f}s"
        )
        return parents, children

    def _record_failed_file(self, file_path: Path, reason: str) -> None:
        """Append failed file details to data/processed/failed_files.json."""
        failed_file_path = self.settings.processed_data_dir / "failed_files.json"
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        if failed_file_path.exists():
            try:
                with open(failed_file_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        records.append(
            {
                "file_path": str(file_path),
                "filename": file_path.name,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": reason,
            }
        )

        with open(failed_file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def _record_scanned_for_gpu_ocr(self, file_path: Path, reason: str) -> None:
        """Record scanned PDF that failed/fast-skipped OCR to data/processed/scanned_for_gpu_ocr.json."""
        queue_file = self.settings.processed_data_dir / "scanned_for_gpu_ocr.json"
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        if queue_file.exists():
            try:
                with open(queue_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        # Deduplicate by file_path
        existing_paths = {r.get("file_path") for r in records}
        if str(file_path) not in existing_paths:
            records.append(
                {
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reason": reason,
                }
            )
            with open(queue_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

    def process_directory(
        self,
        directory_path: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        max_files: Optional[int] = None,
        allow_ocr: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Batch process all legal PDF and judgment documents in directory with fault tolerance."""
        target_dir = Path(directory_path or self.settings.raw_data_dir).resolve()
        if not target_dir.exists():
            raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

        pattern = "**/*" if recursive else "*"
        all_candidates = [
            p for p in target_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".txt"]
        ]

        if max_files:
            all_candidates = all_candidates[:max_files]

        total_files = len(all_candidates)
        logger.info(f"Discovered {total_files} judgment files in {target_dir}")

        successful_files = 0
        failed_files = 0
        all_parent_chunks: List[Dict[str, Any]] = []
        all_child_chunks: List[Dict[str, Any]] = []
        batch_start_t = time.perf_counter()

        # Telemetry loop with tqdm
        with tqdm(total=total_files, desc="Ingesting Legal Judgments", unit="file") as pbar:
            for file_path in all_candidates:
                file_start = time.perf_counter()
                try:
                    parents, children = self.process_file(file_path, allow_ocr=allow_ocr)
                    if not parents and not children:
                        failed_files += 1
                        print(f"[FAST-SKIP] Scanned PDF queued for background OCR: {file_path.name}")
                        self._record_scanned_for_gpu_ocr(file_path, "Zero text extracted / OCR skipped")
                    else:
                        all_parent_chunks.extend(parents)
                        all_child_chunks.extend(children)
                        successful_files += 1
                except Exception as err:
                    failed_files += 1
                    print(f"[FAST-SKIP] Scanned PDF queued for background OCR: {file_path.name}")
                    logger.warning(
                        f"[FAST-SKIP] Skipping file {file_path.name} due to OCR failure: {err}"
                    )
                    self._record_scanned_for_gpu_ocr(file_path, str(err))
                    self._record_failed_file(file_path, str(err))
                finally:
                    file_duration = time.perf_counter() - file_start
                    pbar.set_postfix({"last_file_sec": f"{file_duration:.1f}"})
                    pbar.update(1)

        total_time = time.perf_counter() - batch_start_t
        total_parents = len(all_parent_chunks)
        total_children = len(all_child_chunks)

        summary = {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_parents": total_parents,
            "total_children": total_children,
            "parent_chunks": all_parent_chunks,
            "child_chunks": all_child_chunks,
            "parents": all_parent_chunks,
            "children": all_child_chunks,
            "duration_sec": round(total_time, 2),
            "directory": str(target_dir),
            "parent_store_path": str(self.settings.parent_store_dir),
        }

        logger.info(
            f"Ingestion batch complete: {successful_files}/{total_files} processed successfully, "
            f"{failed_files} queued for background OCR. Generated {total_parents} parents, {total_children} children."
        )
        return summary
