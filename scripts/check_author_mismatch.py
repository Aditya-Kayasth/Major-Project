#!/usr/bin/env python3
"""Read-only: why do manifest and PDF authors disagree? (no indexing, writes nothing)

Reads documents with author_pdf_agrees = 0 from the collection's SQLite store, re-parses each PDF
with src/judgment_text.py and prints one compact block per document:
  manifest author (+source, raw judge column) | PDF author (+source) | name keys equal?
  PDF bench line judges | every '[NAME, J.]' running header with its count | 'delivered by' line

Usage (server):
    python scripts/check_author_mismatch.py --collection sc_criminal_cpu_test
"""

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import judgment_text as jt  # noqa: E402
from src.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain manifest vs PDF author disagreements")
    parser.add_argument("--collection", default="sc_criminal_cpu_test")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    settings = get_settings()
    db = settings.sqlite_path_for(args.collection)
    if not db.exists():
        sys.exit(f"[ERROR] no SQLite store for collection {args.collection}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT doc_id, year, layout, author_judge, author_source, pdf_author, pdf_author_source, judges, "
        "pdf_judges, source_path FROM documents WHERE author_pdf_agrees = 0 ORDER BY year, doc_id LIMIT ?;",
        (args.limit,),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
    print(f"{len(rows)} disagreeing of {total} documents in {args.collection}")

    manifest = {}
    if settings.manifest_path.exists():
        import pandas as pd

        mf = pd.read_parquet(settings.manifest_path, columns=["doc_id", "judge_column", "author_judge_parquet"])
        manifest = {r["doc_id"]: r for r in mf.to_dict("records")}

    for r in rows:
        print("-" * 100)
        m = manifest.get(r["doc_id"], {})
        same_key = jt.name_key(r["author_judge"]) == jt.name_key(r["pdf_author"])
        print(f"{r['doc_id']} | {r['year']} | {r['layout']} | manifest author={r['author_judge']!r} ({r['author_source']}) "
              f"| parquet judge={m.get('judge_column')!r} author_judge={m.get('author_judge_parquet')!r}")
        print(f"  PDF author={r['pdf_author']!r} ({r['pdf_author_source']}) | name keys equal: {same_key} | "
              f"manifest judges={r['judges']!r} | PDF bench={r['pdf_judges']!r}")
        try:
            pages = jt.extract_pages(r["source_path"])
        except Exception as err:
            print(f"  [cannot open PDF: {err}]")
            continue
        running = Counter()
        delivered = []
        for page in pages:
            for line in page.splitlines():
                line = line.strip()
                mm = jt.RUNNING_AUTHOR_RE.match(line)
                if mm:
                    running[mm.group(1).strip()] += 1
                if jt.DELIVERED_RE.match(line):
                    delivered.append(line[:120])
        sigs = [l.strip() for p in pages for l in p.splitlines() if jt.JUDGE_SIGNATURE_RE.match(l.strip())][:4]
        print(f"  running headers: {dict(running)} | delivered by: {delivered[:2]} | 'NAME, J.' lines: {sigs}")
    conn.close()


if __name__ == "__main__":
    main()
