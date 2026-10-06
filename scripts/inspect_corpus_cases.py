import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
import chromadb
from chromadb.config import Settings as ChromaSettings

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings

def inspect_corpus():
    settings = get_settings()
    chroma_dir = settings.chroma_db_dir
    collection_name = settings.collection_name
    parent_dir = settings.parent_store_dir
    processed_dir = settings.processed_data_dir

    print("=" * 80)
    print("  INDIAN LEGAL PRECEDENTS - CORPUS CASE INVENTORY")
    print("=" * 80)
    print(f"ChromaDB Directory : {chroma_dir}")
    print(f"Collection Name    : {collection_name}")
    print(f"Parent Store Dir   : {parent_dir}")
    print(f"Processed Dir      : {processed_dir}")
    print("-" * 80)

    # 1. Inspect ChromaDB Collection
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    collection = client.get_or_create_collection(collection_name)
    count = collection.count()
    print(f"\n[ChromaDB] Total chunks indexed: {count}")

    # Fetch all records
    records = collection.get(include=["metadatas", "documents"])
    ids = records.get("ids", [])
    metadatas = records.get("metadatas", [])
    documents = records.get("documents", [])

    # Group by unique case
    cases = defaultdict(lambda: {
        "chunk_count": 0,
        "sample_ids": [],
        "case_titles": set(),
        "courts": set(),
        "years": set(),
        "source_paths": set(),
        "sample_texts": [],
        "roles": defaultdict(int),
    })

    for cid, meta, doc in zip(ids, metadatas, documents):
        source = meta.get("source_path", "unknown")
        # Extract filename / stem
        stem = Path(source).stem if source != "unknown" else "unknown"
        case_title = meta.get("case_title", stem)
        court = meta.get("court", "Unknown Court")
        year = meta.get("year", "Unknown Year")
        role = meta.get("role", "Unknown Role")

        case_key = stem if stem != "unknown" else case_title

        cases[case_key]["chunk_count"] += 1
        cases[case_key]["case_titles"].add(case_title)
        cases[case_key]["courts"].add(str(court))
        cases[case_key]["years"].add(str(year))
        cases[case_key]["source_paths"].add(source)
        cases[case_key]["roles"][role] += 1
        if len(cases[case_key]["sample_texts"]) < 3:
            cases[case_key]["sample_texts"].append(doc)
        if len(cases[case_key]["sample_ids"]) < 3:
            cases[case_key]["sample_ids"].append(cid)

    print(f"[ChromaDB] Unique cases identified: {len(cases)}\n")

    catalog = []
    for idx, (case_key, info) in enumerate(sorted(cases.items(), key=lambda x: -x[1]["chunk_count"]), 1):
        titles = list(info["case_titles"])
        preferred_title = titles[0] if titles else case_key
        courts = list(info["courts"])
        years = list(info["years"])
        sources = list(info["source_paths"])
        primary_source = sources[0] if sources else ""
        
        # Look for parent store files matching this case
        parent_files = list(parent_dir.glob(f"{case_key}*.json")) + list(parent_dir.glob(f"*{case_key}*.json"))
        
        # Analyze text for statutory references and key legal holding
        combined_text = " ".join(info["sample_texts"])
        statutes = set(re.findall(r"(?:Section|Sec\.|Art\.|Article)\s+[\dA-Za-z]+(?:\([A-Za-z\d]+\))?(?:\s+[A-Za-z]+)?", combined_text, re.IGNORECASE))
        acts = set(re.findall(r"\b(?:IPC|CrPC|CPC|Evidence Act|NDPS|Constitution|BNS|BNSS|BSA|Arbitration|Contract Act|Motor Vehicles Act|Service Rules|Central Civil Services|Customs Act)\b", combined_text, re.IGNORECASE))
        
        case_info = {
            "index": idx,
            "case_key": case_key,
            "preferred_title": preferred_title,
            "chunk_count": info["chunk_count"],
            "courts": courts,
            "years": years,
            "primary_source": primary_source,
            "roles": dict(info["roles"]),
            "statutes": list(statutes)[:5],
            "acts": list(acts)[:5],
            "sample_snippet": info["sample_texts"][0][:300] if info["sample_texts"] else ""
        }
        catalog.append(case_info)

        print(f"[{idx:02d}] Case Key / Stem : {case_key}")
        print(f"     Title in Meta  : {preferred_title}")
        print(f"     Chunks in DB   : {info['chunk_count']}")
        print(f"     Court / Year   : {', '.join(courts)} | {', '.join(years)}")
        print(f"     Source Path    : {primary_source}")
        print(f"     Roles Dist.    : {dict(info['roles'])}")
        print(f"     Acts / Statutes: {', '.join(list(acts)[:4])} | {', '.join(list(statutes)[:4])}")
        print(f"     Excerpt        : {case_info['sample_snippet'][:180]}...")
        print("-" * 80)

    # Save catalog as JSON
    out_dir = PROJECT_ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "corpus_case_catalog.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"\n[OK] Catalog saved to {out_file}")

if __name__ == "__main__":
    inspect_corpus()

