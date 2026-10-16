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
from src.resource_guard import ResourceGuard
from src.sqlite_store import SQLiteParentStore

# Configure logger
logger = logging.getLogger("IndianLegalRAG.DocumentLoader")

# Hard Rhetorical Transition Markers in Indian Judgments
RHETORICAL_ANCHORS_PATTERN = re.compile(
    r"(?:\n|^)(?:"
    r"Learned\s+(?:Senior\s+)?Counsel\s+(?:for\s+the\s+appellant|for\s+the\s+respondent|submitted|contended|argued)|"
    r"Having\s+heard\s+(?:the\s+learned\s+counsel|both\s+sides|the\s+parties)|"
    r"Having\s+considered\s+the\s+(?:rival\s+submissions|material\s+on\s+record|evidence)|"
    r"We\s+are\s+of\s+the\s+(?:considered\s+)?opinion\s+that|"
    r"In\s+our\s+(?:considered\s+)?view|"
    r"The\s+question\s+(?:that\s+falls\s+for\s+our\s+consideration|for\s+determination\s+is)|"
    r"We\s+hold\s+and\s+direct\s+that|It\s+is\s+held\s+that|We\s+accordingly\s+hold|"
    r"For\s+the\s+reasons\s+stated\s+above|In\s+the\s+result|The\s+appeal\s+is\s+(?:allowed|dismissed)"
    r")",
    re.IGNORECASE,
)
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
    """Production document loader with Multimodal OCR fallback and 3-Tier Hierarchical Legal Chunking."""

    def __init__(self, settings: Optional[Settings] = None, enable_ocr: bool = True):
        self.settings = settings or get_settings()
        self.enable_ocr = enable_ocr
        self.client: Optional[genai.Client] = None
        if self.settings.gemini_api_key and self.settings.gemini_api_key != "your_gemini_api_key_here":
            self.client = genai.Client(api_key=self.settings.gemini_api_key)

        # Initialize embedded SQLite parent store
        self.sqlite_store = SQLiteParentStore(self.settings.sqlite_db_path)

        # Initialize resource guardrail monitor
        self.guard = ResourceGuard(
            max_ram_gb=self.settings.max_ram_gb,
            warn_ram_gb=self.settings.warn_ram_gb,
            max_storage_gb=self.settings.max_storage_gb,
            storage_check_paths=[
                self.settings.chroma_db_dir,
                self.settings.parent_store_dir,
                self.settings.sqlite_db_path.parent,
            ],
            num_cpu_threads=self.settings.num_threads,
        )

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
        first_chunk = text[:3000]
        lines = [line.strip() for line in first_chunk.splitlines() if line.strip()]

        invalid_prefixes = {
            "hence", "therefore", "where", "whereas", "wherein", "learned", "counsel",
            "observed", "held", "court", "order", "decision", "paragraph", "section",
            "article", "dated", "in", "the", "that", "this", "it", "moreover", "further",
            "whether", "having", "taking", "perusal", "matter", "case"
        }

        # Look for explicit VERSUS / vs / v on a dedicated line or standard case caption
        extracted_from_lines = ""
        for i, line in enumerate(lines[:35]):
            line_clean = line.strip()
            if re.fullmatch(r"(?:versus|vs\.?|v\.?)", line_clean, flags=re.IGNORECASE):
                if i > 0 and i + 1 < len(lines):
                    p1_cand = lines[i - 1]
                    p2_cand = lines[i + 1]
                    p1_cand = re.sub(r"\b(appellant|petitioner|applicant|plaintiff)s?\b.*", "", p1_cand, flags=re.IGNORECASE).strip(" .:,")
                    p2_cand = re.sub(r"\b(respondent|opposite\s+party|defendant)s?\b.*", "", p2_cand, flags=re.IGNORECASE).strip(" .:,")
                    if len(p1_cand) >= 3 and len(p2_cand) >= 3:
                        p1_first = p1_cand.split()[0].lower()
                        if p1_first not in invalid_prefixes:
                            extracted_from_lines = f"{p1_cand} vs. {p2_cand}"
                            break

        if extracted_from_lines:
            clean_title = extracted_from_lines
        else:
            title_matches = re.finditer(
                r"([A-Z][A-Za-z0-9\s\.\,\&]{2,60})\s+(?:v\.|vs\.|VERSUS)\s+([A-Z][A-Za-z0-9\s\.\,\&]{2,60})",
                first_chunk,
            )
            for tm in title_matches:
                p1 = " ".join(tm.group(1).split()).strip(" .:,")
                p2 = " ".join(tm.group(2).split()).strip(" .:,")
                p1_words = p1.split()
                if not p1_words:
                    continue
                first_w = p1_words[0].lower()
                if first_w in invalid_prefixes:
                    continue
                p1 = re.sub(r"\.{2,}.*", "", p1).strip(" .:,")
                p2 = re.sub(r"\.{2,}.*", "", p2).strip(" .:,")
                p1 = re.sub(r"\b(appellant|petitioner|applicant)s?\b.*", "", p1, flags=re.IGNORECASE).strip(" .:,")
                p2 = re.sub(r"\b(respondent|opposite\s+party)s?\b.*", "", p2, flags=re.IGNORECASE).strip(" .:,")
                if len(p1) >= 3 and len(p2) >= 3:
                    cand_title = f"{p1} vs. {p2}"
                    if len(cand_title) > 8 and not clean_title.startswith("20"):
                        clean_title = cand_title
                        break

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

        # 5. Judicial Profile & Coram Extraction
        judge_info = self._extract_judicial_profile(text)

        return {
            "source_path": str(file_path),
            "filename": filename,
            "case_title": clean_title,
            "case_category": case_category,
            "court": court,
            "year": year,
            "author_judge": judge_info["author_judge"],
            "coram": judge_info["coram"],
            "bench_type": judge_info["bench_type"],
            "disposition": judge_info["disposition"],
        }

    def _extract_judicial_profile(self, text: str) -> Dict[str, Any]:
        """Extract Coram (sitting judges), Author Judge, Bench Type, and Case Disposition."""
        first_chunk = text[:3500]
        coram: List[str] = []
        author_judge: Optional[str] = None
        disposition: str = "PENDING/NOT_STATED"

        # Coram Extraction: Supreme Court bracketed list [JUSTICE A, JUSTICE B, JJ.]
        sc_coram = re.search(r"\[([A-Z\s\,\.\n]+?(?:JJ\.|J\.))\]", first_chunk)
        if sc_coram:
            raw = sc_coram.group(1).replace("\n", " ")
            raw = re.sub(r"\b(?:JJ\.|J\.)\b", "", raw)
            parts = re.split(r"\,|\bAND\b", raw)
            for p in parts:
                clean_p = p.strip().title()
                if len(clean_p) > 3 and not any(w in clean_p.lower() for w in ["court", "judgment", "report"]):
                    coram.append(clean_p)

        # Coram Extraction: High Court CORAM: / BEFORE:
        if not coram:
            c_match = re.search(r"(?:CORAM|BEFORE)\s*:\s*([^\n\r]+(?:\n[^\n\r]+)?)", first_chunk, re.IGNORECASE)
            if c_match:
                raw = c_match.group(1).replace("\n", " ")
                raw = re.sub(r"J\s*U\s*D\s*G\s*M\s*E\s*N\s*T", "", raw, flags=re.IGNORECASE)
                names = re.findall(
                    r"(?:HON(?:'BLE|OURABLE)\s+)?(?:MR\.|MS\.|MRS\.|DR\.)?\s*(?:JUSTICE\s+)?([A-Z\s\.]+)",
                    raw,
                )
                for n in names:
                    clean_n = n.strip().title()
                    if len(clean_n) > 3 and clean_n.lower() not in ["court", "judgment", "cav", "order"]:
                        coram.append(clean_n)

        # Author Judge Extraction
        deliv = re.search(
            r"Judgment\s+of\s+the\s+Court\s+was\s+delivered\s+by\s*[:\n\s]*([A-Z\s\.]+?)(?:,|\.|\n)",
            first_chunk,
            re.IGNORECASE,
        )
        if deliv:
            raw_author = deliv.group(1).strip().title()
            if len(raw_author) > 3:
                author_judge = raw_author
        elif re.search(r"([A-Z\s\.]+),\s*J\b", first_chunk):
            matches = re.findall(r"([A-Z\s\.]+),\s*J\b", first_chunk)
            for m in matches:
                clean_m = m.strip().replace("\n", " ").title()
                if len(clean_m) > 3 and clean_m.lower() not in ["court", "judgment"]:
                    author_judge = clean_m
                    break
        elif len(coram) == 1:
            author_judge = coram[0]

        # Bench Type Classification
        num_judges = max(1, len(coram))
        if num_judges >= 5:
            bench_type = "Constitution Bench"
        elif num_judges >= 3:
            bench_type = "Full Bench"
        elif num_judges == 2:
            bench_type = "Division Bench"
        else:
            bench_type = "Single Bench"

        # Disposition Classification
        last_chunk = text[-3000:].lower()
        if any(w in last_chunk for w in ["appeal is allowed", "petition is allowed", "bail is granted", "conviction is set aside"]):
            disposition = "ALLOWED"
        elif any(w in last_chunk for w in ["appeal is dismissed", "petition is dismissed", "bail is rejected", "no merit"]):
            disposition = "DISMISSED"
        elif any(w in last_chunk for w in ["disposed of", "stands disposed", "partly allowed"]):
            disposition = "DISPOSED OF"

        return {
            "coram": ", ".join(coram) if coram else "Not Recorded",
            "author_judge": author_judge or (coram[0] if coram else "Not Recorded"),
            "bench_type": bench_type,
            "disposition": disposition,
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
        """Implement production 3-Tier Hierarchical Legal Chunking.

        1. Macro Sections & Rhetorical Anchoring: Split text into coherent legal sections
           (~1500 characters) while immediately flushing on counsel/court transition markers.
        2. SQLite Storage: Store all parent sections atomically in SQLite database for O(1) retrieval.
        3. Child Chunks with Context Prepending: Subdivide into ~400-char granular chunks with
           prepended rhetorical breadcrumbs '[<Role> | <Case Title>]: <text>' to eliminate lexical inversion.

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        path = Path(source_path)
        base_meta = self._extract_legal_metadata(text, path)
        case_title = base_meta["case_title"]
        court = base_meta["court"]
        year = base_meta["year"]

        parent_size = self.settings.parent_chunk_size
        child_size = self.settings.child_chunk_size
        overlap = self.settings.chunk_overlap

        # Split document into paragraphs / structural blocks
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [text.strip()] if text.strip() else [""]

        # Build Parent Chunks respecting paragraph boundaries and Hard Rhetorical Flush
        parent_blocks: List[str] = []
        current_block: List[str] = []
        current_len = 0

        for para in raw_paragraphs:
            para_len = len(para)
            is_rhetorical_transition = bool(RHETORICAL_ANCHORS_PATTERN.search(para))

            # Force immediate flush if rhetorical transition marker is detected
            # (Ensures counsel submissions and judicial findings never contaminate the same parent chunk)
            if is_rhetorical_transition and current_block:
                parent_blocks.append("\n\n".join(current_block))
                current_block = [para]
                current_len = para_len
            elif current_len + para_len > parent_size and current_block:
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

                    # Context Prepending: Inject structural breadcrumbs into vector representation
                    prepended_vector_text = f"[{role} | {case_title}]: {c_text}"

                    child_chunk_data = {
                        "parent_id": parent_id,
                        "chunk_id": child_id,
                        "case_title": case_title,
                        "court": court,
                        "year": year,
                        "role": role,
                        "char_length": len(prepended_vector_text),
                        "raw_char_length": len(c_text),
                        "text": prepended_vector_text,  # Vectors capture role & case authority
                        "raw_text": c_text,             # Clean snippet for prompt injection
                        "source_path": str(path),
                        "parent_index": p_idx,
                    }
                    child_chunks.append(child_chunk_data)

                if child_end >= p_len:
                    break
                child_start += child_size - overlap

        # Persist parent chunks atomically to SQLite database
        try:
            self.sqlite_store.insert_parents_batch(parent_chunks)
        except Exception as sql_err:
            logger.warning(f"Could not persist parents to SQLite: {sql_err}")

        # Also maintain JSON backup in parent_store_dir for backward compatibility
        parent_store_dir = self.settings.parent_store_dir
        parent_store_dir.mkdir(parents=True, exist_ok=True)
        for pc in parent_chunks:
            p_file = parent_store_dir / f"{pc['parent_id']}.json"
            try:
                with open(p_file, "w", encoding="utf-8") as f:
                    json.dump(pc, f, ensure_ascii=False)
            except Exception:
                pass

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
