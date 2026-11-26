#!/usr/bin/env python3
"""Phase 2: download the English PDFs of the SC criminal judgments and write the manifest.

Selection (CLAUDE.md D2, verified in Phase 1): every row of $METADATA_DIR/sc/year=YYYY/metadata.parquet
whose case number contains CRIMINAL or CRL. It is de-duplicated on (cnr, decision_date). No sampling.

Source : https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=YYYY/english/{path}_EN.pdf
Dest   : $RAW_DATA_DIR/sc_aws/year=YYYY/{path}_EN.pdf
Outputs: $METADATA_DIR/manifest.parquet + manifest.csv, one row per selected judgment, with all parsed
         fields, doc_id, local_path and download status.
         $METADATA_DIR/download_failures.csv (rows whose status is failed / not_found).

How a download works:
- Resumable: a file that already exists and starts with %PDF is not downloaded again.
- Each file goes to <name>.part first and is renamed only after the size and %PDF checks pass.
- Errors are retried with backoff. HTTP 403/404 counts as not_found and is not retried.
- --max-docs N downloads only the first N selected rows, sorted by (year, path); use it for tests only.
  The manifest still lists every selected row (status not_attempted for the rest).
- The year-tar fallback from CLAUDE.md Phase 2 is NOT implemented. It is needed only if HTTPS fails.

Usage (server):
    python scripts/download_aws_sc.py --years 2018-2025 --criminal-only --workers 8 --max-docs 10
    python scripts/download_aws_sc.py --years 2018-2025 --criminal-only --workers 8
    python scripts/download_aws_sc.py --status
"""

import argparse
import csv
import datetime
import hashlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.sc_metadata import iso_date, parse_record  # noqa: E402

S3_BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"
USER_AGENT = "sc-criminal-precedent-research/1.0"
PDF_MAGIC = b"%PDF"
EXPECTED_CRIMINAL = {2018: 185, 2019: 264, 2020: 132, 2021: 196, 2022: 244, 2023: 250, 2024: 273, 2025: 347}
NEUTRAL_RE = re.compile(r"^\d{4}\s+INSC\s+\d+$")
FAILURE_STATUSES = ("failed", "not_found")


# --------------------------------------------------------------------------- helpers
def parse_years(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def _s(value: Any) -> str:
    """String value of a parquet cell; '' for None / NaN / <NA>."""
    return value.strip() if isinstance(value, str) else ""


def doc_id_for(cnr: str, path: str) -> str:
    """CLAUDE.md Section 7: sha1(cnr or path)[:16]."""
    key = (cnr or "").strip() or (path or "").strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def s3_key(year: int, path: str) -> str:
    return f"data/pdf/year={year}/english/{path}_EN.pdf"


def local_pdf_path(dest_root: Path, year: int, path: str) -> Path:
    return dest_root / f"year={year}" / f"{path}_EN.pdf"


def is_valid_pdf(file_path: Path) -> bool:
    try:
        if not file_path.is_file() or file_path.stat().st_size == 0:
            return False
        with open(file_path, "rb") as f:
            return f.read(4) == PDF_MAGIC
    except OSError:
        return False


def pick_citation(case_id: str, parsed_neutral: Optional[str], parquet_scr: str, parsed_scr: Optional[str]) -> str:
    """Display citation: neutral citation (parquet case_id, else parsed), else the SCR citation."""
    if case_id and NEUTRAL_RE.match(case_id):
        return case_id
    if parsed_neutral:
        return parsed_neutral
    return parquet_scr or parsed_scr or ""


def build_row(raw_row: Dict[str, Any], year: int, dest_root: Path) -> Optional[Dict[str, Any]]:
    """Manifest row for a criminal judgment, or None if the row is not criminal."""
    p = parse_record(raw_row.get("raw_html"), raw_row.get("title"), judge_column=raw_row.get("judge"))
    if not p["is_criminal"]:
        return None
    path, cnr = _s(raw_row.get("path")), _s(raw_row.get("cnr"))
    case_id, parquet_scr = _s(raw_row.get("case_id")), _s(raw_row.get("citation"))
    decision_raw = _s(raw_row.get("decision_date"))
    neutral = case_id if NEUTRAL_RE.match(case_id) else (p["neutral_citation"] or "")
    return {
        "doc_id": doc_id_for(cnr, path),
        "year": year,
        "path": path,
        "cnr": cnr,
        "case_id": case_id,
        "nc_display": _s(raw_row.get("nc_display")),
        "title": _s(raw_row.get("title")),
        "petitioner": _s(raw_row.get("petitioner")),
        "respondent": _s(raw_row.get("respondent")),
        "court": "Supreme Court of India",
        "decision_date": iso_date(decision_raw) or p["decision_date_html"] or "",
        "decision_date_raw": decision_raw,
        "case_no": p["case_no"] or "",
        "case_type": p["case_type"] or "",
        "case_type_raw": p["case_type_raw"] or "",
        "judges": "; ".join(p["judges"]),
        "bench_coram_count": p["bench_coram_count"],
        "bench_declared": p["bench_declared"],
        "bench_strength": p["bench_strength"],
        "bench_source": p["bench_source"],
        "bench_type": p["bench_type"],
        "bench_check": p["bench_check"],
        "needs_pdf_bench": p["bench_check"] == "no_coram",
        "author_judge": p["author_judge"] or "",
        "author_source": p["author_source"],
        "judge_column": _s(raw_row.get("judge")),
        "author_judge_parquet": _s(raw_row.get("author_judge")),
        "disposal_nature": _s(raw_row.get("disposal_nature")) or (p["disposal_nature_html"] or ""),
        "citation": pick_citation(case_id, p["neutral_citation"], parquet_scr, p["scr_citation"]),
        "neutral_citation": neutral,
        "scr_citation": parquet_scr or (p["scr_citation"] or ""),
        "headnote_labelled": bool(p["headnote_labelled"]),
        "issue_for_consideration": p["issue_for_consideration"] or "",
        "headnote_snippet": p["headnote_snippet"] or "",
        "subject_hint": p["subject_hint"] or "",
        "s3_key": s3_key(year, path),
        "local_path": str(local_pdf_path(dest_root, year, path)),
        "status": "",
        "bytes": 0,
        "error": "",
        "downloaded_at": "",
    }


def select_criminal(metadata_dir: Path, years: List[int], dest_root: Path) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]:
    """All criminal rows for the years, de-duplicated on (cnr, decision_date). Returns rows, per-year counts, dropped dups."""
    rows: List[Dict[str, Any]] = []
    per_year: Dict[int, int] = {}
    dropped: List[Dict[str, Any]] = []
    seen = set()
    for year in years:
        pq = metadata_dir / "sc" / f"year={year}" / "metadata.parquet"
        if not pq.exists():
            per_year[year] = -1
            continue
        count = 0
        for raw_row in pd.read_parquet(pq).to_dict("records"):
            row = build_row(raw_row, year, dest_root)
            if row is None:
                continue
            key = (row["cnr"], row["decision_date"]) if row["cnr"] else (row["path"], row["decision_date"])
            if key in seen:
                dropped.append(row)
                continue
            seen.add(key)
            rows.append(row)
            count += 1
        per_year[year] = count
    rows.sort(key=lambda r: (r["year"], r["path"]))
    return rows, per_year, dropped


def download_one(key: str, dest: Path, retries: int, timeout: float) -> Tuple[str, int, str]:
    """Download one key to dest. Returns (status, bytes, error); status in ok|exists|failed|not_found."""
    if is_valid_pdf(dest):
        return "exists", dest.stat().st_size, ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = f"{S3_BASE}/{key}"
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            n = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                expected = resp.headers.get("Content-Length")
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
            if expected and expected.isdigit() and int(expected) != n:
                raise IOError(f"short read {n}/{expected} bytes")
            with open(tmp, "rb") as f:
                if f.read(4) != PDF_MAGIC:
                    tmp.unlink(missing_ok=True)
                    return "failed", n, "not a PDF (missing %PDF header)"
            tmp.replace(dest)
            return "ok", n, ""
        except urllib.error.HTTPError as err:
            last_err = f"HTTP {err.code}"
            tmp.unlink(missing_ok=True)
            if err.code in (403, 404):
                return "not_found", 0, last_err
        except Exception as err:
            last_err = f"{type(err).__name__}: {err}"[:200]
            tmp.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(min(30.0, 2.0 ** (attempt + 1)))
    return "failed", 0, last_err


def write_manifest(rows: List[Dict[str, Any]], metadata_dir: Path) -> None:
    df = pd.DataFrame(rows)
    for suffix, writer in ((".parquet", lambda d, p: d.to_parquet(p, index=False)),
                           (".csv", lambda d, p: d.to_csv(p, index=False))):
        final = metadata_dir / f"manifest{suffix}"
        tmp = metadata_dir / f"manifest.tmp{suffix}"
        writer(df, tmp)
        tmp.replace(final)


def write_failures(rows: List[Dict[str, Any]], metadata_dir: Path) -> int:
    failures = [r for r in rows if r["status"] in FAILURE_STATUSES]
    path = metadata_dir / "download_failures.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "year", "path", "s3_key", "status", "error"])
        for r in failures:
            w.writerow([r["doc_id"], r["year"], r["path"], r["s3_key"], r["status"], r["error"]])
    return len(failures)


def disk_stats(dest_root: Path) -> Tuple[int, float, int]:
    """(valid PDF files, MB, leftover .part files) under dest_root."""
    files = mb = parts = 0
    if dest_root.exists():
        for p in dest_root.rglob("*"):
            if p.name.endswith(".part"):
                parts += 1
            elif p.suffix.lower() == ".pdf" and is_valid_pdf(p):
                files += 1
                mb += p.stat().st_size / (1024 * 1024)
    return files, round(mb, 1), parts


def print_status(dest_root: Path, metadata_dir: Path) -> None:
    files, mb, parts = disk_stats(dest_root)
    fail_csv = metadata_dir / "download_failures.csv"
    n_fail = 0
    if fail_csv.exists():
        with open(fail_csv, encoding="utf-8") as f:
            n_fail = max(0, sum(1 for _ in f) - 1)
    manifest = metadata_dir / "manifest.parquet"
    line = f"STATUS | PDFs on disk: {files} | {mb} MB | .part leftovers: {parts} | failures CSV rows: {n_fail}"
    if manifest.exists():
        st = Counter(pd.read_parquet(manifest, columns=["status"])["status"])
        line += " | manifest: " + " ".join(f"{k}:{v}" for k, v in sorted(st.items()))
    print(line)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Download SC criminal judgment PDFs and write the manifest")
    parser.add_argument("--years", default="2018-2025")
    parser.add_argument("--criminal-only", action="store_true", help="Required: the scope is criminal judgments (D2)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=None, help="Tests only: download the first N selected rows")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--dest-dir", default=None, help="Default: $RAW_DATA_DIR/sc_aws")
    parser.add_argument("--status", action="store_true", help="Print files / MB / failures and exit")
    args = parser.parse_args()

    settings = get_settings()
    metadata_dir = Path(args.metadata_dir).expanduser() if args.metadata_dir else settings.metadata_dir
    dest_root = Path(args.dest_dir).expanduser() if args.dest_dir else settings.raw_data_dir / "sc_aws"
    if not metadata_dir.is_dir():
        sys.exit("[ERROR] METADATA_DIR does not exist (set it in .env or pass --metadata-dir).")
    if args.status:
        print_status(dest_root, metadata_dir)
        return
    if not args.criminal_only:
        sys.exit("[ERROR] Pass --criminal-only: the locked scope is SC criminal judgments (CLAUDE.md D2).")

    years = parse_years(args.years)
    t0 = time.perf_counter()
    rows, per_year, dropped = select_criminal(metadata_dir, years, dest_root)
    n = len(rows)
    targets = rows if args.max_docs is None else rows[: max(0, args.max_docs)]
    target_ids = {r["doc_id"] for r in targets}

    S: List[str] = []
    S.append(f"PHASE 2 DOWNLOAD | years {years[0]}-{years[-1]} | workers {args.workers} | "
             f"max-docs {args.max_docs if args.max_docs is not None else 'all'} | {datetime.datetime.now():%Y-%m-%d %H:%M}")
    full_range = set(years) == set(EXPECTED_CRIMINAL) and all(per_year.get(y, -1) >= 0 for y in years)
    check = (" OK" if n + len(dropped) == sum(EXPECTED_CRIMINAL.values()) else " MISMATCH vs D2") if full_range else ""
    S.append(f"selected (criminal, deduped): {n}{check} | duplicates dropped: {len(dropped)} | "
             f"doc_id collisions: {n - len({r['doc_id'] for r in rows})}")
    S.append("per year: " + " ".join(f"{y}:{c if c >= 0 else 'MISSING'}" for y, c in per_year.items()))

    # ------------------------------------------------------------------ download
    lock = threading.Lock()
    counts: Counter = Counter()
    done = 0
    progress_every = max(1, len(targets) // 20)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, r["s3_key"], Path(r["local_path"]), args.retries, args.timeout): r
            for r in targets
        }
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                status, nbytes, err = fut.result()
            except Exception as exc:  # defensive: never lose the manifest because of one file
                status, nbytes, err = "failed", 0, f"{type(exc).__name__}: {exc}"[:200]
            r.update(status=status, bytes=nbytes, error=err,
                     downloaded_at=datetime.datetime.now().isoformat(timespec="seconds") if status == "ok" else "")
            with lock:
                counts[status] += 1
                done += 1
                if done % progress_every == 0 or done == len(targets):
                    print(f"[progress] {done}/{len(targets)} " + " ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                          flush=True)

    for r in rows:
        if r["doc_id"] in target_ids:
            continue
        p = Path(r["local_path"])
        if is_valid_pdf(p):
            r.update(status="exists", bytes=p.stat().st_size)
        else:
            r.update(status="not_attempted")

    write_manifest(rows, metadata_dir)
    n_fail = write_failures(rows, metadata_dir)
    elapsed = time.perf_counter() - t0
    run_mb = sum(r["bytes"] for r in targets if r["status"] == "ok") / (1024 * 1024)
    on_disk = sum(1 for r in rows if r["status"] in ("ok", "exists"))
    disk_mb = sum(r["bytes"] for r in rows if r["status"] in ("ok", "exists")) / (1024 * 1024)

    S.append(f"this run: attempted {len(targets)} | ok {counts['ok']} | already on disk {counts['exists']} | "
             f"failed {counts['failed']} | not_found {counts['not_found']}")
    S.append(f"downloaded {run_mb:.1f} MB in {elapsed:.0f}s ({run_mb / elapsed if elapsed else 0:.2f} MB/s)")
    S.append(f"manifest rows with a valid PDF: {on_disk}/{n} | {disk_mb:.1f} MB")
    bs = Counter(r["bench_source"] or "none" for r in rows)
    au = Counter(r["author_source"] for r in rows)
    S.append("bench_source: " + " ".join(f"{k}:{v}" for k, v in bs.most_common())
             + " | needs PDF bench: " + str(sum(r["needs_pdf_bench"] for r in rows)))
    S.append("author_source: " + " ".join(f"{k}:{v}" for k, v in au.most_common()))
    S.append("bench_type: " + " ".join(f"{k or 'unknown'}:{v}" for k, v in Counter(r["bench_type"] for r in rows).most_common()))
    fails = [r for r in rows if r["status"] in FAILURE_STATUSES]
    S.append(f"failures: {n_fail} (first 5 below)")
    for r in fails[:5]:
        S.append(f"  {r['status']} | {r['s3_key']} | {r['error']}"[:200])
    S.append("manifest: $METADATA_DIR/manifest.parquet + .csv | failures: $METADATA_DIR/download_failures.csv")
    S.append("PDFs: $RAW_DATA_DIR/sc_aws/year=YYYY/{path}_EN.pdf")
    print("\n".join(S))


if __name__ == "__main__":
    main()
