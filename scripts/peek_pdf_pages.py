#!/usr/bin/env python3
"""Read-only peek at real judgment PDFs, used to design the Phase 3 cleaner and page-1 parsers.

Prints:
  1. the Phase 1 summary block from $METADATA_DIR/sc_explore_report.md
  2. a one-line download status from $METADATA_DIR/manifest.parquet
  3. for one downloaded PDF from each of 2019, 2023 and 2025 (seed 42, Criminal Appeal preferred):
     - its manifest fields and page count
     - full page-1 text, the first 1,500 chars of page 2, and the last 600 chars of the final page
  4. the first 1,200 chars of page 1 of one PDF whose author is unknown in the manifest

Every printed PDF line is prefixed with '|' so line breaks and blank lines stay visible.
Writes nothing.

Usage (server):
    python scripts/peek_pdf_pages.py
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402

AVAILABLE = ("ok", "exists")
SHOW_FIELDS = ["doc_id", "year", "path", "title", "case_no", "judges", "bench_strength", "bench_source",
               "author_judge", "author_source", "decision_date", "citation"]


def phase1_summary(metadata_dir: Path) -> str:
    report = metadata_dir / "sc_explore_report.md"
    if not report.exists():
        return "(sc_explore_report.md not found)"
    text = report.read_text(encoding="utf-8")
    if "```text" not in text:
        return "(summary block not found in sc_explore_report.md)"
    return text.split("```text", 1)[1].split("```", 1)[0].strip()


def pick(df: pd.DataFrame, year: int, rng: random.Random) -> Optional[Dict[str, Any]]:
    """One available row from `year`, else from the nearest year that has one."""
    years = sorted(df["year"].unique(), key=lambda y: (abs(int(y) - year), int(y)))
    for y in years:
        cand = df[df["year"] == y]
        pref = cand[cand["case_type"] == "Criminal Appeal"]
        pool = pref if len(pref) else cand
        if len(pool):
            row = pool.iloc[rng.randrange(len(pool))].to_dict()
            if int(y) != year:
                row["_note"] = f"no downloaded PDF for {year}; using {int(y)}"
            return row
    return None


def show_lines(label: str, text: str) -> None:
    print(f"--- {label} ({len(text)} chars)")
    for line in text.splitlines():
        print(f"|{line}")


def peek(row: Dict[str, Any], page1_chars: Optional[int] = None, full: bool = True) -> None:
    print("=" * 100)
    if row.get("_note"):
        print(f"NOTE: {row['_note']}")
    print(" | ".join(f"{k}={row.get(k)}" for k in SHOW_FIELDS))
    path = Path(str(row.get("local_path")))
    try:
        reader = PdfReader(str(path))
    except Exception as err:
        print(f"[ERROR] cannot open PDF: {type(err).__name__}: {err}")
        return
    pages: List[Any] = list(reader.pages)
    print(f"pages={len(pages)} | file_kb={path.stat().st_size / 1024:.0f}")

    def page_text(i: int) -> str:
        try:
            return pages[i].extract_text() or ""
        except Exception as err:
            return f"[extract_text failed: {type(err).__name__}: {err}]"

    p1 = page_text(0) if pages else ""
    show_lines("PAGE 1" if page1_chars is None else f"PAGE 1 [:{page1_chars}]", p1 if page1_chars is None else p1[:page1_chars])
    if not full:
        return
    if len(pages) > 1:
        show_lines("PAGE 2 [:1500]", page_text(1)[:1500])
    if len(pages) > 2:
        show_lines(f"LAST PAGE ({len(pages)}) [-600:]", page_text(len(pages) - 1)[-600:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Peek at real judgment PDFs (read-only)")
    parser.add_argument("--years", default="2019,2023,2025")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    metadata_dir = settings.metadata_dir
    manifest = metadata_dir / "manifest.parquet"
    if not manifest.exists():
        sys.exit("[ERROR] manifest.parquet not found in METADATA_DIR (run scripts/download_aws_sc.py first).")

    print("=" * 100)
    print("PHASE 1 EXPLORE SUMMARY (from sc_explore_report.md)")
    print(phase1_summary(metadata_dir))

    df = pd.read_parquet(manifest)
    status = df["status"].value_counts().to_dict()
    avail = df[df["status"].isin(AVAILABLE)]
    mb = avail["bytes"].sum() / (1024 * 1024)
    print("=" * 100)
    print(f"DOWNLOAD STATUS | manifest rows {len(df)} | " + " ".join(f"{k}:{v}" for k, v in sorted(status.items()))
          + f" | available PDFs {len(avail)} ({mb:.1f} MB)")

    rng = random.Random(args.seed)
    for year in [int(y) for y in args.years.split(",")]:
        row = pick(avail, year, rng)
        if row is None:
            print(f"[{year}] no downloaded PDFs available")
            continue
        peek(row)

    unknown = avail[avail["author_source"] == "unknown"]
    print("=" * 100)
    print(f"AUTHOR-UNKNOWN EXAMPLE ({len(unknown)} available rows with author_source=unknown)")
    if len(unknown):
        peek(unknown.iloc[rng.randrange(len(unknown))].to_dict(), page1_chars=1200, full=False)
    print("=" * 100)


if __name__ == "__main__":
    main()
