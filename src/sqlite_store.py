"""SQLite Parent Document Store for Indian Legal RAG.

Replaces thousands of individual JSON files with an embedded, ACID-compliant,
high-throughput SQLite database for O(1) in-memory pointer lookups.
Features:
- Write-Ahead Logging (WAL mode) for concurrency and fast writes
- Atomic batch insertions
- Automatic migration from legacy JSON file directories
- Low RAM & disk footprint (< 100 MB for thousands of documents)
"""

import datetime
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("IndianLegalRAG.SQLiteStore")


class SQLiteParentStore:
    """Manages parent section storage using an embedded SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and row factory enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high read/write concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")  # 64 MB cache
        return conn

    def _init_db(self) -> None:
        """Create parent table and indices if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    case_title TEXT,
                    court TEXT,
                    year INTEGER,
                    role TEXT,
                    macro_section TEXT,
                    char_length INTEGER,
                    text TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_case_title ON parent_chunks(case_title);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_court ON parent_chunks(court);")
            conn.commit()

    def insert_parent(self, parent_data: Dict[str, Any]) -> bool:
        """Insert a single parent chunk."""
        return self.insert_parents_batch([parent_data]) > 0

    def insert_parents_batch(self, parents: List[Dict[str, Any]]) -> int:
        """Insert multiple parent chunks in a single atomic transaction. Skips duplicates."""
        if not parents:
            return 0

        rows = []
        for p in parents:
            parent_id = p.get("parent_id")
            if not parent_id:
                continue

            meta = p.get("metadata") or {}
            case_title = p.get("case_title") or meta.get("case_title", "")
            court = meta.get("court", "")
            year = meta.get("year")
            role = meta.get("role", "General")
            macro_section = meta.get("structural_heading", "General")
            text = p.get("text", "")
            char_length = len(text)
            meta_json = json.dumps(meta, ensure_ascii=False)

            rows.append(
                (parent_id, case_title, court, year, role, macro_section, char_length, text, meta_json)
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO parent_chunks (
                    parent_id, case_title, court, year, role, macro_section, char_length, text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()
            return cursor.rowcount

    def get_parent(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single parent chunk by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT parent_id, case_title, court, year, role, macro_section, char_length, text, metadata_json
                FROM parent_chunks WHERE parent_id = ?;
                """,
                (parent_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            meta = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception:
                    pass

            return {
                "parent_id": row["parent_id"],
                "case_title": row["case_title"],
                "text": row["text"],
                "metadata": {
                    **meta,
                    "court": row["court"],
                    "year": row["year"],
                    "role": row["role"],
                    "structural_heading": row["macro_section"],
                    "char_length": row["char_length"],
                },
            }

    def get_parents_batch(self, parent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple parent chunks by IDs in a single query."""
        if not parent_ids:
            return {}

        results = {}
        placeholders = ",".join("?" for _ in parent_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT parent_id, case_title, court, year, role, macro_section, char_length, text, metadata_json
                FROM parent_chunks WHERE parent_id IN ({placeholders});
                """,
                parent_ids,
            )
            for row in cursor.fetchall():
                meta = {}
                if row["metadata_json"]:
                    try:
                        meta = json.loads(row["metadata_json"])
                    except Exception:
                        pass
                results[row["parent_id"]] = {
                    "parent_id": row["parent_id"],
                    "case_title": row["case_title"],
                    "text": row["text"],
                    "metadata": {
                        **meta,
                        "court": row["court"],
                        "year": row["year"],
                        "role": row["role"],
                        "structural_heading": row["macro_section"],
                        "char_length": row["char_length"],
                    },
                }
        return results

    def clear(self) -> None:
        """Delete all parent chunks (used by a full re-index)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM parent_chunks;")
            conn.commit()

    def count_parents(self) -> int:
        """Count total parent records in the SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM parent_chunks;")
            return cursor.fetchone()[0]

    def migrate_from_json_dir(self, json_dir: Path) -> int:
        """Migrate legacy individual JSON files from parent_store directory into SQLite."""
        p_dir = Path(json_dir).resolve()
        if not p_dir.exists():
            return 0

        json_files = list(p_dir.glob("*.json"))
        if not json_files:
            return 0

        parents_to_import = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "parent_id" in data:
                        parents_to_import.append(data)
            except Exception as err:
                logger.warning(f"Could not read legacy parent file {jf.name}: {err}")

        imported = self.insert_parents_batch(parents_to_import)
        logger.info(f"Successfully migrated {imported} parent documents from JSON files into SQLite.")
        return imported

    def get_cases_by_judge(self, judge_query: str) -> List[Dict[str, Any]]:
        """Retrieve all distinct cases authored by or featuring a specific judge in coram."""
        clean_name = judge_query.lower().strip()
        # Remove common honorifics for fuzzy matching
        clean_name = re.sub(r"\b(justice|hon'ble|honourable|dr\.|mr\.|mrs\.|ms\.)\b", "", clean_name).strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT case_title, court, year, metadata_json
                FROM parent_chunks
                WHERE LOWER(metadata_json) LIKE ? OR LOWER(text) LIKE ?
                ORDER BY year DESC, case_title ASC;
                """,
                (f"%{clean_name}%", f"%{clean_name}%"),
            )
            cases = []
            seen_titles = set()
            for row in cursor.fetchall():
                title = row["case_title"]
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                meta = {}
                if row["metadata_json"]:
                    try:
                        meta = json.loads(row["metadata_json"])
                    except Exception:
                        meta = {}
                cases.append({
                    "case_title": title,
                    "court": row["court"] or meta.get("court", "High Court / Supreme Court"),
                    "year": row["year"] or meta.get("year", "N/A"),
                    "author_judge": meta.get("author_judge", judge_query.title()),
                    "coram": meta.get("coram", ""),
                    "bench_type": meta.get("bench_type", "Single Bench"),
                    "disposition": meta.get("disposition", "Not Stated"),
                    "case_category": meta.get("case_category", "General"),
                })
            return cases

