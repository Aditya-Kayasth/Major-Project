#!/usr/bin/env python3
"""Synthetic Known-Item Benchmark Generator for the Indexed Judgment Corpus.

Builds a larger, unbiased retrieval benchmark over the documents that are actually
indexed (e.g. the 400 Kaggle Supreme Court judgments), following the synthetic query
generation approach of InPars / Promptagator:

1. Sample N distinct judgments from the SQLite parent store (fixed seed => reproducible).
2. From each judgment pick the parent section that most looks like the court's holding
   ("we hold", "held that", "in our view", "the appeal is allowed/dismissed", ...).
3. Ask Gemini to write ONE question a lawyer would ask that this passage answers.
4. Leakage controls: the question must not name the parties, the case, citations or
   dates; questions containing party-name tokens from the case title are rejected.

Each query records ``target_doc_stem`` (the judgment's file stem), so evaluation matches
on exact document identity. Output: data/benchmarks/golden_queries.json. Gemini calls go
through the free-tier budget guard (src/llm_budget.py) and are cached, so re-running with
the same seed costs nothing.

Usage:
    python scripts/generate_synthetic_queries.py --num-docs 40
"""

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.llm_budget import get_budget  # noqa: E402

HOLDING_CUES = [
    "we hold", "held that", "in our view", "in our considered view", "we are of the",
    "the question", "accordingly", "appeal is allowed", "appeal is dismissed",
    "set aside", "it is well settled", "the principle",
]

TITLE_STOPWORDS = {
    "the", "of", "and", "vs", "versus", "state", "union", "india", "ors", "anr", "others",
    "another", "ltd", "pvt", "limited", "through", "its", "secretary", "government", "on",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "pradesh", "uttar", "madhya", "west", "bengal",
}

GENERATION_PROMPT = """You are building a retrieval benchmark for an Indian legal research engine.

Below is a passage from an Indian court judgment. Write ONE realistic research question that an
advocate would type into a legal search engine, which THIS passage answers.

Strict rules:
- Do NOT mention party names, the case title, judge names, citations, case numbers or dates.
- Ask about the legal issue (statute, doctrine, procedure, right) in general terms, as a lawyer
  researching precedent would. 15-40 words.
- The answer must be supported by the passage.

Return strictly valid JSON:
{{"query": "...", "answer": "1-2 sentence answer grounded in the passage",
  "domain": "short legal domain label", "statutes": ["statutes/articles mentioned, if any"]}}

Passage:
\"\"\"{passage}\"\"\"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic known-item retrieval queries")
    parser.add_argument("--num-docs", type=int, default=40, help="Number of distinct judgments to sample")
    parser.add_argument(
        "--path-filter", type=str, default="",
        help="Only sample judgments whose source_path contains this substring (e.g. 'sc_2025')",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output benchmark file (default: data/benchmarks/golden_queries.json)",
    )
    return parser.parse_args()


def holding_score(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(cue) for cue in HOLDING_CUES)


def party_tokens(case_title: str) -> List[str]:
    words = re.findall(r"[A-Za-z]{4,}", case_title or "")
    return [w.lower() for w in words if w.lower() not in TITLE_STOPWORDS]


def load_documents(db_path: Path, path_filter: str) -> Dict[str, Dict[str, Any]]:
    """Group SQLite parent sections by source document."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    docs: Dict[str, Dict[str, Any]] = {}
    for row in conn.execute("SELECT case_title, text, metadata_json FROM parent_chunks"):
        meta = json.loads(row["metadata_json"] or "{}")
        source_path = str(meta.get("source_path", "")).replace("\\", "/")
        if not source_path or (path_filter and path_filter not in source_path):
            continue
        stem = Path(source_path).stem
        doc = docs.setdefault(stem, {"case_title": row["case_title"], "source_path": source_path, "parents": []})
        doc["parents"].append(row["text"] or "")
    conn.close()
    return docs


def generate_query(client: genai.Client, model: str, passage: str) -> Optional[Dict[str, Any]]:
    raw = get_budget().generate(
        client=client,
        models=[model],
        contents=GENERATION_PROMPT.format(passage=passage[:3500]),
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3),
    )
    try:
        data = json.loads(raw.strip()) if raw else None
    except Exception:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if isinstance(data, dict) and data.get("query") and data.get("answer"):
        return data
    return None


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        sys.exit("[ERROR] GEMINI_API_KEY is required to generate synthetic queries.")

    docs = load_documents(settings.sqlite_db_path, args.path_filter)
    print(f"[INFO] {len(docs)} indexed judgments match path filter '{args.path_filter}'.")
    if not docs:
        sys.exit("[ERROR] No documents found. Index the corpus first (python index_corpus.py ...).")

    rng = random.Random(args.seed)
    stems = sorted(docs)
    rng.shuffle(stems)

    client = genai.Client(api_key=settings.gemini_api_key)
    synthetic: List[Dict[str, Any]] = []
    rejected = 0

    for stem in stems:
        if len(synthetic) >= args.num_docs:
            break
        doc = docs[stem]
        candidates = [p for p in doc["parents"] if len(p) >= 600]
        if not candidates:
            continue
        passage = max(candidates, key=holding_score)

        if get_budget().remaining_today() <= 0:
            print("[WARN] Gemini daily budget exhausted; stopping early.")
            break
        data = generate_query(client, settings.gemini_model, passage)
        if not data:
            rejected += 1
            continue

        query = str(data["query"]).strip()
        leaked = [t for t in party_tokens(doc["case_title"]) if re.search(rf"\b{re.escape(t)}\b", query.lower())]
        if leaked:
            rejected += 1
            print(f"  [reject] party-name leakage {leaked}: {query[:80]}")
            continue

        qid = f"SQ-{len(synthetic) + 1:03d}"
        answer = str(data["answer"]).strip()
        synthetic.append({
            "query_id": qid,
            "query": query,
            "query_text": query,
            "target_case_title": doc["case_title"],
            "target_doc_stem": stem,
            "target_citations": [],
            "domain": str(data.get("domain") or "General"),
            "target_role": "Ratio Decidendi",
            "expected_statutes": [str(s) for s in (data.get("statutes") or [])],
            "expected_ratio": answer,
            "ground_truth_answer": answer,
            "source": "synthetic",
            "verified": False,  # flip to true after a human reads the query against source_passage
            "source_passage": passage[:1500],
        })
        print(f"  [{qid}] {doc['case_title'][:50]} :: {query[:90]}")

    output_path = Path(args.output) if args.output else settings.benchmarks_dir / "golden_queries.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Benchmark queries : {len(synthetic)} (rejected: {rejected})  -> {output_path}")
    print(f"Gemini requests left today: {get_budget().remaining_today()}")
    print("Next: python evaluate_pipeline.py --skip-llm-judge   (retrieval metrics, zero API calls)")


if __name__ == "__main__":
    main()
