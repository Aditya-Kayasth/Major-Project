import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings
from evaluate_pipeline import is_case_match

with open('data/benchmarks/golden_queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

client = chromadb.PersistentClient('chromadb_store', settings=ChromaSettings(anonymized_telemetry=False))
col = client.get_collection('indian_legal_precedents')
res = col.get(include=['metadatas', 'documents'])

all_chunks = [{'chunk_id': cid, 'metadata': m, 'text': d} for cid, m, d in zip(res['ids'], res['metadatas'], res['documents'])]

print(f'Total chunks in collection: {len(all_chunks)}')
for q in queries:
    target = q['target_case_title']
    matches = [c for c in all_chunks if is_case_match(c, target)]
    print(f"{q['query_id']}: Target '{target}' -> Matches found: {len(matches)}")
    assert len(matches) > 0, f"No matches for {target}!"
print('\n[SUCCESS] All 10 queries verified and matched in the collection!')
