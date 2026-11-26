#!/usr/bin/env python3
"""Phase 1: apply src/sc_metadata.py to the official SC metadata and report what it finds.

Reads  $METADATA_DIR/sc/year=YYYY/metadata.parquet (already downloaded, CLAUDE.md D1).
Prints a summary of at most 60 lines to paste back, and writes the full report to
$METADATA_DIR/sc_explore_report.md. Network use: 5 HTTPS HEAD requests to the public S3 bucket.

The summary covers:
- per year: rows, case_no parsed, criminal count vs CLAUDE.md D2 (must total 1,891)
- case types, the Coram bench (parsed share, bench_strength, cross-check with 'Bench : N Judges'),
  author marks, headnote labels, subject hints, decision-date agreement, duplicates
- 5 parsed examples, 5 failures, HEAD checks of 5 PDF keys

Usage (server):
    python scripts/aws_sc_explore.py --years 2018-2025
"""

import argparse
import datetime
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.sc_metadata import clean_html, iso_date, parse_record  # noqa: E402

S3_BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"
EXPECTED_CRIMINAL = {2018: 185, 2019: 264, 2020: 132, 2021: 196, 2022: 244, 2023: 250, 2024: 273, 2025: 347}
EXPECTED_TOTAL = 1891
SUMMARY_MAX_LINES = 60


def parse_years(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def pdf_key(year: int, path: str) -> str:
    return f"data/pdf/year={year}/english/{path}_EN.pdf"


def head_check(key: str) -> Tuple[Any, Optional[str]]:
    req = urllib.request.Request(f"{S3_BASE}/{key}", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.headers.get("Content-Length")
    except urllib.error.HTTPError as err:
        return err.code, None
    except Exception as err:  # network errors are reported, not raised
        return "ERR", type(err).__name__


def coram_context(raw_html: Any, width: int = 140) -> str:
    text = clean_html(raw_html)
    i = text.find("Coram")
    if i < 0:
        return "(no 'Coram' in text) " + text[-width:]
    return text[i : i + width]


def _s(value: Any) -> str:
    """String value of a parquet cell; '' for None / NaN / <NA>."""
    return value.strip() if isinstance(value, str) else ""


def txt(value: Any) -> str:
    """Display string for any DataFrame cell: '' for None / NaN / <NA>; lists joined with '; '."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(txt(v) for v in value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # array-like cells
        pass
    return str(value)


def num(value: Any) -> str:
    """Integer display for a possibly-NaN numeric cell (bench_strength becomes float when NaN is present)."""
    s = txt(value)
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def n_items(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


def fmt_example(r: pd.Series, width: int = 220) -> str:
    line = (f"  {txt(r.get('path'))} | {txt(r.get('case_type'))} | {txt(r.get('bench_type')) or '?'} | "
            f"{txt(r.get('judges'))} | author={txt(r.get('author_judge_coram'))} | {txt(r.get('subject_hint'))[:40]}")
    return line[:width]


def fmt_failure(r: pd.Series, width: int = 220) -> str:
    line = (f"  {txt(r.get('path'))} | {txt(r.get('bench_check'))} | coram={n_items(r.get('judges'))} "
            f"declared={num(r.get('bench_declared')) or '-'} | {txt(r.get('coram_context'))}")
    return line[:width]


def fmt_failure_report(r: pd.Series) -> str:
    return (f"- {txt(r.get('path'))} | {txt(r.get('bench_check'))} | judges={txt(r.get('judges'))} | "
            f"declared={num(r.get('bench_declared')) or '-'} | judge column={txt(r.get('judge_column'))} | "
            f"{txt(r.get('coram_context'))}")


def positions(mask: pd.Series) -> List[int]:
    """Row positions (for .iloc) where mask is True."""
    return [i for i, ok in enumerate(mask.tolist()) if ok]


def share(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the official SC metadata (Phase 1)")
    parser.add_argument("--years", default="2018-2025", help="e.g. 2018-2025 or 2019,2025")
    parser.add_argument("--metadata-dir", default=None, help="Overrides METADATA_DIR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-head", action="store_true", help="Skip the 5 HTTPS HEAD checks")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).expanduser() if args.metadata_dir else get_settings().metadata_dir
    if not metadata_dir.is_dir():
        sys.exit("[ERROR] METADATA_DIR does not exist (set it in .env or pass --metadata-dir).")
    years = parse_years(args.years)
    rng = random.Random(args.seed)

    # ------------------------------------------------------------------ parse
    records: List[Dict[str, Any]] = []
    per_year: Dict[int, Dict[str, Any]] = {}
    for year in years:
        pq = metadata_dir / "sc" / f"year={year}" / "metadata.parquet"
        if not pq.exists():
            per_year[year] = {"rows": None}
            continue
        df = pd.read_parquet(pq)
        n_case_no = n_crim = 0
        for row in df.to_dict("records"):
            raw = row.get("raw_html")
            p = parse_record(raw, row.get("title"), judge_column=row.get("judge"))
            n_case_no += p["case_no"] is not None
            n_crim += p["is_criminal"]
            if not p["is_criminal"]:
                continue
            p.update(
                year=year,
                path=_s(row.get("path")),
                cnr=_s(row.get("cnr")),
                title=_s(row.get("title")),
                decision_date_parquet=iso_date(row.get("decision_date")),
                judge_column=_s(row.get("judge")),
            )
            if p["bench_check"] in ("no_coram", "mismatch"):
                p["coram_context"] = coram_context(raw)
            records.append(p)
        per_year[year] = {"rows": len(df), "case_no": n_case_no, "criminal": n_crim}

    # Positional index (0..n-1); all row access below uses .iloc
    crim = pd.DataFrame(records).reset_index(drop=True)
    n = len(crim)

    S: List[str] = []  # summary (stdout, <= 60 lines)
    R: List[str] = []  # extra detail for the report only
    S.append(f"PHASE 1 EXPLORE | years {years[0]}-{years[-1]} | {datetime.datetime.now():%Y-%m-%d %H:%M}")
    S.append("year   rows  case_no  criminal  expected")
    for year in years:
        y = per_year[year]
        if y["rows"] is None:
            S.append(f"{year}   metadata.parquet NOT FOUND")
            continue
        exp = EXPECTED_CRIMINAL.get(year)
        flag = "" if exp is None else ("OK" if y["criminal"] == exp else "MISMATCH")
        S.append(f"{year} {y['rows']:6d} {y['case_no']:8d} {y['criminal']:9d} {exp if exp else '-':>9} {flag}")
    total_ok = "OK" if n == EXPECTED_TOTAL else f"MISMATCH (expected {EXPECTED_TOTAL})"
    S.append(f"TOTAL criminal = {n}  {total_ok}")
    if n == 0:
        print("\n".join(S))
        return

    # ------------------------------------------------------------------ case types
    S.append("-- case types (criminal): count | raw | label")
    ct = crim.groupby(["case_type_raw", "case_type"], dropna=False).size().sort_values(ascending=False)
    for (raw_t, label), cnt in list(ct.items())[:12]:
        S.append(f"  {cnt:5d} | {txt(raw_t) or '(none)'} | {txt(label) or '(none)'}")
    if len(ct) > 12:
        S.append(f"  ... {len(ct) - 12} more raw types (see report)")
    R.append("## All case types (criminal)\n")
    R.extend(f"- {cnt} | {txt(raw_t) or '(none)'} | {txt(label) or '(none)'}" for (raw_t, label), cnt in ct.items())

    # ------------------------------------------------------------------ bench
    has_bench = crim["bench_strength"].notna()
    strength_counts = Counter(int(s) for s in crim.loc[has_bench, "bench_strength"])
    checks = Counter(crim["bench_check"])
    S.append("-- bench (Coram)")
    S.append(f"  parsed bench: {share(int(has_bench.sum()), n)}")
    S.append("  bench_strength: " + "  ".join(f"{k}:{v}" for k, v in sorted(strength_counts.items()))
             + f"  none:{int((~has_bench).sum())}")
    S.append("  bench_type: " + "  ".join(f"{k or 'unknown'}:{v}" for k, v in Counter(crim["bench_type"].map(txt)).most_common()))
    S.append("  vs 'Bench : N Judges': " + "  ".join(f"{k}:{checks.get(k, 0)}"
                                                    for k in ("match", "mismatch", "no_declared", "no_coram")))
    S.append(f"  author marked '*': {share(int(crim['author_judge_coram'].notna().sum()), n)}")

    # ------------------------------------------------------------------ other fields
    S.append("-- other fields")
    S.append(f"  headnote labelled: {share(int(crim['headnote_labelled'].sum()), n)} | "
             f"issue_for_consideration: {share(int(crim['issue_for_consideration'].notna().sum()), n)} | "
             f"subject_hint: {share(int(crim['subject_hint'].notna().sum()), n)}")
    labelled_by_year = crim.groupby("year")["headnote_labelled"].sum().astype(int).to_dict()
    S.append("  headnote labelled by year: " + " ".join(f"{y}:{c}" for y, c in labelled_by_year.items()))
    both = crim["decision_date_html"].notna() & crim["decision_date_parquet"].notna()
    agree = int((crim.loc[both, "decision_date_html"] == crim.loc[both, "decision_date_parquet"]).sum())
    S.append(f"  decision_date parquet parsed: {share(int(crim['decision_date_parquet'].notna().sum()), n)} | "
             f"html==parquet: {share(agree, int(both.sum()))}")
    S.append(f"  scr_citation: {share(int(crim['scr_citation'].notna().sum()), n)} | "
             f"neutral_citation: {share(int(crim['neutral_citation'].notna().sum()), n)}")
    disp = Counter(crim["disposal_nature_html"].map(txt).replace("", "(none)")).most_common(6)
    S.append("  disposal (top 6): " + " | ".join(f"{k}:{v}" for k, v in disp))
    dup_key = crim["cnr"].map(txt) + "|" + crim["decision_date_parquet"].map(txt)
    dup_mask = dup_key.duplicated(keep=False)
    dups = int(dup_mask.sum())
    S.append(f"  duplicates on (cnr, decision_date): {dups} rows | duplicate paths: "
             f"{int(crim['path'].duplicated(keep=False).sum())} | empty cnr: {int((crim['cnr'].map(txt) == '').sum())}")
    if dups:
        R.append("\n## Duplicate (cnr, decision_date) rows\n")
        for i in positions(dup_mask):
            r = crim.iloc[i]
            R.append(f"- {txt(r.get('cnr'))} | {txt(r.get('decision_date_parquet'))} | {txt(r.get('path'))} | "
                     f"{txt(r.get('case_no'))}")

    # ------------------------------------------------------------------ examples
    ok_pos = positions(crim["bench_check"].isin(["match", "no_declared"]))
    S.append("-- 5 parsed examples")
    for i in rng.sample(ok_pos, min(5, len(ok_pos))):
        S.append(fmt_example(crim.iloc[i]))
    R.append("\n## 20 parsed examples\n")
    R.extend("-" + fmt_example(crim.iloc[i])[1:] for i in rng.sample(ok_pos, min(20, len(ok_pos))))

    fail_pos = positions(crim["bench_check"].isin(["no_coram", "mismatch"]))
    S.append(f"-- failures: {len(fail_pos)} (no_coram {checks.get('no_coram', 0)}, "
             f"mismatch {checks.get('mismatch', 0)}); 5 shown")
    for i in rng.sample(fail_pos, min(5, len(fail_pos))):
        S.append(fmt_failure(crim.iloc[i]))
    R.append("\n## All failures (no_coram / mismatch)\n")
    R.extend(fmt_failure_report(crim.iloc[i]) for i in fail_pos)

    # ------------------------------------------------------------------ HEAD checks
    S.append("-- HEAD check of 5 PDF keys (public S3, HTTPS)")
    if args.no_head:
        S.append("  skipped (--no-head)")
    else:
        for i in rng.sample(range(n), min(5, n)):
            r = crim.iloc[i]
            key = pdf_key(int(r["year"]), txt(r.get("path")))
            status, length = head_check(key)
            size = f"{int(length) / 1024:.0f} KB" if (length and str(length).isdigit()) else (length or "-")
            S.append(f"  {status} | {size} | {key}")

    # ------------------------------------------------------------------ output
    report_path = metadata_dir / "sc_explore_report.md"
    S.append("full report: $METADATA_DIR/sc_explore_report.md")
    if len(S) > SUMMARY_MAX_LINES:
        S = S[: SUMMARY_MAX_LINES - 1] + [f"... summary truncated ({len(S)} lines); see the report"]
    report = ["# SC metadata exploration (Phase 1)\n", "## Summary\n", "```text", *S, "```\n", *R, ""]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(S))


if __name__ == "__main__":
    main()
