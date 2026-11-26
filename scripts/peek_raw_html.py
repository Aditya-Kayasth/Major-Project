#!/usr/bin/env python3
"""Read-only peek at the official SC metadata, used to design src/sc_metadata.py from real text.

For a few CRIMINAL judgments per year it prints: path, cnr, judge, decision_date, the parsed
case number, where key markers sit in the cleaned raw_html, and the first N characters of
raw_html with tags stripped and whitespace collapsed. It also prints total / criminal counts
for each sampled year as a check against CLAUDE.md D2.

Reads $METADATA_DIR/sc/year=YYYY/metadata.parquet. Writes nothing.

Usage (server):
    python scripts/peek_raw_html.py                       # 3 from 2019, 2 from 2025, 2000 chars
    python scripts/peek_raw_html.py --sample 2018=2 2024=2 --chars 3000
"""

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CASE_NO_RE = re.compile(r"Case No\s*:\s*</span>\s*<font[^>]*>\s*([^<]+)", re.IGNORECASE)
CRIMINAL_RE = re.compile(r"CRIMINAL|CRL", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# CLAUDE.md D2: criminal counts verified on the server (26 Nov)
EXPECTED_CRIMINAL = {2018: 185, 2019: 264, 2020: 132, 2021: 196, 2022: 244, 2023: 250, 2024: 273, 2025: 347}
MARKERS = ["Case No", "S.C.R", "[", "Headnote", "Issue for Consideration"]
COLUMNS = ["title", "path", "cnr", "judge", "decision_date", "case_id", "nc_display", "citation", "raw_html"]


def clean_html(raw: Optional[str]) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    if not isinstance(raw, str):
        return ""
    return WS_RE.sub(" ", html.unescape(TAG_RE.sub(" ", raw))).strip()


def parse_case_no(raw: Optional[str]) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    m = CASE_NO_RE.search(raw)
    return WS_RE.sub(" ", html.unescape(m.group(1))).strip() if m else None


def parse_sample_args(items) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for item in items:
        year, _, n = item.partition("=")
        out[int(year)] = int(n or 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Peek at cleaned raw_html of criminal SC judgments")
    parser.add_argument("--sample", nargs="+", default=["2019=3", "2025=2"], help="YEAR=N pairs")
    parser.add_argument("--chars", type=int, default=2000, help="Characters of cleaned text to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-dir", default=None, help="Overrides METADATA_DIR from .env")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    md = args.metadata_dir or os.getenv("METADATA_DIR")
    if not md:
        sys.exit("[ERROR] METADATA_DIR is not set in .env (and --metadata-dir not given).")
    metadata_dir = Path(md).expanduser()
    if not metadata_dir.is_dir():
        sys.exit("[ERROR] METADATA_DIR is set but the directory does not exist.")
    print("METADATA_DIR: OK")

    for year, n in parse_sample_args(args.sample).items():
        pq = metadata_dir / "sc" / f"year={year}" / "metadata.parquet"
        if not pq.exists():
            print(f"[{year}] metadata.parquet NOT FOUND under METADATA_DIR/sc/year={year}/")
            continue

        df = pd.read_parquet(pq)
        missing = [c for c in COLUMNS if c not in df.columns]
        df["case_no"] = df["raw_html"].apply(parse_case_no) if "raw_html" in df.columns else None
        crim = df[df["case_no"].fillna("").str.contains(CRIMINAL_RE)]

        expected = EXPECTED_CRIMINAL.get(year)
        verdict = "" if expected is None else (" OK" if len(crim) == expected else f" MISMATCH (D2 expects {expected})")
        print("=" * 100)
        print(
            f"[{year}] rows={len(df)} | case_no parsed={df['case_no'].notna().sum()} | "
            f"criminal={len(crim)}{verdict}" + (f" | missing columns: {missing}" if missing else "")
        )

        sample = crim.sample(n=min(n, len(crim)), random_state=args.seed) if len(crim) else crim
        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            raw = row.get("raw_html")
            raw_len = len(raw) if isinstance(raw, str) else 0
            text = clean_html(raw)
            lower = text.lower()
            offsets = ", ".join(f"{m!r}@{lower.find(m.lower())}" for m in MARKERS)
            print("-" * 100)
            print(
                f"[{year} #{i}] path={row.get('path')} | cnr={row.get('cnr')} | judge={row.get('judge')} | "
                f"decision_date={row.get('decision_date')}"
            )
            print(
                f"  case_no={row.get('case_no')!r} | case_id={row.get('case_id')} | nc_display={row.get('nc_display')} | "
                f"citation={row.get('citation')}"
            )
            print(f"  title={row.get('title')}")
            print(f"  raw_html_len={raw_len} | clean_len={len(text)} | offsets (-1 = absent): {offsets}")
            print(f"  TEXT[:{args.chars}]: {text[: args.chars]}")
    print("=" * 100)


if __name__ == "__main__":
    main()
