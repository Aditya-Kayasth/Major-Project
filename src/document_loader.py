"""Document loading and chunking on top of the official manifest (Phase 3).

Metadata comes ONLY from $METADATA_DIR/manifest.parquet (joined on local_path / doc_id). There
is no file-name regex. Text extraction, cleaning, headnote split and chunking live in
src/judgment_text.py (pure; also used by the spawned extraction workers in index_corpus.py).

This class keeps a small API for single documents and tests:
    load_document(path)                 -> (text, info, was_ocr)
    create_parent_child_chunks(text, path, meta=None, section_kind="body")
Parents are written to the per-collection SQLite store (settings.sqlite_path_for(collection)).
"""

import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src import judgment_text as jt
from src.config import Settings, get_settings
from src.llm_budget import make_gemini_client
from src.sqlite_store import SQLiteParentStore

logger = logging.getLogger("IndianLegalRAG.DocumentLoader")

# Kept for backward compatibility (the rhetorical flush now lives in judgment_text.ANCHOR_RE)
RHETORICAL_ANCHORS_PATTERN = jt.ANCHOR_RE

LEGAL_OCR_SYSTEM_INSTRUCTION = (
    "You are a legal OCR engine for Indian court judgments. Transcribe the document verbatim, keep "
    "paragraph numbering and all citations (AIR, SCC, SCR, INSC) exactly as printed."
)


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """manifest.parquet -> {resolved local_path: row}. Only rows with a downloaded PDF."""
    import pandas as pd

    df = pd.read_parquet(manifest_path)
    df = df[df["status"].isin(["ok", "exists"])]
    return {str(Path(r["local_path"]).resolve()): r for r in df.to_dict("records")}


class LegalDocumentLoader:
    """Loads judgments with manifest metadata and writes parents to the collection's SQLite store."""

    def __init__(self, settings: Optional[Settings] = None, enable_ocr: bool = False,
                 collection: Optional[str] = None, manifest: Optional[Dict[str, Dict[str, Any]]] = None):
        self.settings = settings or get_settings()
        self.enable_ocr = enable_ocr
        self.collection = collection or self.settings.collection_name
        self.client: Optional[Any] = make_gemini_client(self.settings.gemini_api_key) if enable_ocr else None
        self.sqlite_store = SQLiteParentStore(self.settings.sqlite_path_for(self.collection))
        self._manifest = manifest

    # ------------------------------------------------------------------ manifest
    @property
    def manifest(self) -> Dict[str, Dict[str, Any]]:
        if self._manifest is None:
            path = self.settings.manifest_path
            self._manifest = load_manifest(path) if path.exists() else {}
        return self._manifest

    def manifest_row(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        return self.manifest.get(str(Path(file_path).resolve()))

    # ------------------------------------------------------------------ whole judgment
    def process_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract + clean + chunk one PDF that is in the manifest, and store its parents/document row."""
        row = self.manifest_row(file_path)
        if row is None:
            return {"doc_id": None, "error": "not in manifest"}
        s = self.settings
        result = jt.build_document(str(file_path), row, s.parent_chunk_size, s.child_chunk_size, s.chunk_overlap)
        if "error" not in result:
            self.sqlite_store.insert_parents_batch(result["parents"])
            self.sqlite_store.upsert_documents([result["doc"]])
        return result

    # ------------------------------------------------------------------ single text (tests, ad-hoc)
    def extract_digital_pdf(self, file_path: Path) -> str:
        try:
            return "\n\n".join(jt.extract_pages(str(file_path)))
        except Exception as err:
            logger.warning(f"Digital PDF extraction failed for {Path(file_path).name}: {err}")
            return ""

    def load_document(self, file_path: Union[str, Path], allow_ocr: Optional[bool] = None
                      ) -> Tuple[str, Dict[str, Any], bool]:
        """Text of a PDF or text file, cached in PROCESSED_DATA_DIR/<stem>.txt. Returns (text, info, was_ocr)."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target document does not exist: {path}")
        t0 = time.perf_counter()
        cache_file = self.settings.processed_data_dir / f"{path.stem}.txt"
        info: Dict[str, Any] = {"source_path": str(path), "filename": path.name, "ocr_used": False}
        if cache_file.exists():
            cached = cache_file.read_text(encoding="utf-8", errors="replace")
            if cached.strip():
                return cached, {**info, "cached": True, "extraction_time_sec": round(time.perf_counter() - t0, 4)}, False

        was_ocr = False
        if path.suffix.lower() == ".pdf":
            text = self.extract_digital_pdf(path)
            if len(text.strip()) < 50 and (self.enable_ocr if allow_ocr is None else allow_ocr):
                text, was_ocr = self.extract_via_gemini_ocr(path), True
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        if text.strip():
            self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text, encoding="utf-8")
        return text, {**info, "cached": False, "ocr_used": was_ocr,
                      "extraction_time_sec": round(time.perf_counter() - t0, 4)}, was_ocr

    def create_parent_child_chunks(self, text: str, source_path: str, meta: Optional[Dict[str, Any]] = None,
                                   section_kind: str = "body") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Chunk raw text (no PDF cleaning) with the same parent/child logic as the indexer."""
        meta = dict(meta or {})
        path = Path(source_path)
        doc = {
            "doc_id": meta.get("doc_id") or hashlib.sha1(path.name.lower().encode("utf-8")).hexdigest()[:16],
            "case_title": meta.get("case_title") or meta.get("title") or path.stem.replace("_", " "),
            "court": "Supreme Court of India",
            "year": meta.get("year"),
            "source_path": str(path),
            **{k: meta.get(k) for k in jt.DOC_FIELDS if k in meta and k not in ("doc_id", "case_title")},
        }
        s = self.settings
        parents, children = jt.chunk_sections([(section_kind, text.splitlines())], doc,
                                              s.parent_chunk_size, s.child_chunk_size, s.chunk_overlap)
        try:
            self.sqlite_store.insert_parents_batch(parents)
        except Exception as sql_err:
            logger.warning(f"Could not persist parents to SQLite: {sql_err}")
        return parents, children

    # ------------------------------------------------------------------ OCR fallback (off for this corpus)
    def _get_gemini_client(self) -> Any:
        if not self.client:
            self.client = make_gemini_client(self.settings.gemini_api_key)
            if not self.client:
                raise ValueError("GEMINI_API_KEY is not configured (or google-genai is not installed).")
        return self.client

    def extract_via_gemini_ocr(self, file_path: Path) -> str:
        client = self._get_gemini_client()
        from google.genai import types

        uploaded = client.files.upload(file=str(file_path))
        try:
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[uploaded, "Transcribe this judgment."],
                config=types.GenerateContentConfig(system_instruction=LEGAL_OCR_SYSTEM_INSTRUCTION, temperature=0.1),
            )
            return (response.text or "") if response else ""
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    # ------------------------------------------------------------------ failure records
    def _append_json_record(self, name: str, record: Dict[str, Any], dedupe_key: Optional[str] = None) -> None:
        path = self.settings.processed_data_dir / name
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        try:
            records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            records = []
        if dedupe_key and any(r.get(dedupe_key) == record.get(dedupe_key) for r in records):
            return
        records.append(record)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _record_failed_file(self, file_path: Path, reason: str) -> None:
        self._append_json_record("failed_files.json", {
            "file_path": str(file_path), "filename": Path(file_path).name,
            "timestamp": datetime.datetime.now().isoformat(), "error": reason,
        })

    def _record_scanned_for_gpu_ocr(self, file_path: Path, reason: str) -> None:
        self._append_json_record("scanned_for_gpu_ocr.json", {
            "file_path": str(file_path), "filename": Path(file_path).name,
            "timestamp": datetime.datetime.now().isoformat(), "reason": reason,
        }, dedupe_key="file_path")
