# Phase 0 audit: findings and the phase that fixes each one

Audit of the codebase as of commit `69f27eb` (26 Nov 2026), done before the move to the
CPU-only Supreme Court criminal build described in `CLAUDE.md`. Tick each box when its fix is merged.

## Current architecture (summary)

- **Indexing** (`index_corpus.py`):
  - PDFs are processed in micro-batches of 8.
  - `LegalDocumentLoader` extracts text with pypdf (cached in `processed/{stem}.txt`) and reads metadata with regex from the file name and the first ~3.5k characters.
  - Each judgment is split into parents of ~1,500 chars (flushed at rhetorical markers) and children of 400 chars with 80 overlap, each prefixed `[role | title]:`. Parents go to the SQLite table `parent_chunks`.
  - `VectorStoreManager` embeds the children into Chroma. A JSON checkpoint is saved after each batch, and the rank_bm25 pickle is rebuilt once at the end.
- **Query** (`src/rag_pipeline.py`):
  - concordance expansion
  - optional Chroma `where` filter, which also supplies BM25 `allowed_ids`
  - dense top 60 + BM25 top 60
  - RRF (k=60) × bounded authority factor, keep the top 30
  - CrossEncoder, sigmoid × authority factor, keep the top k
  - SQLite parent expansion
  - `llm_budget` provider chain (Groq → Gemini → local, disk cache) for synthesis
- **Evaluation:** `evaluate_pipeline.py` runs 4 configs and matches results on the file stem.

## Phase 3: loader, metadata, indexing, resources

- [ ] Chunk sizes 1500/400/80 → D5: parent 2000, child 800, overlap 120 (from `.env`).
- [ ] Role names: "Ratio Decidendi" → `Ratio/Analysis`; the "Obiter Dicta" fallback → `Unclassified`; add the `Issues` and `Headnote` roles.
- [ ] Text cleaning: remove headers, footers, page numbers and "Signature Not Verified" / "Digitally signed by" blocks; store REPORTABLE as a flag.
- [ ] Section 7 metadata on every child chunk:
  - fields: doc_id, decision_date (ISO), judges, bench_strength, bench_type, citation, case_id, cnr, case_type, disposal_nature, reportable, statutes_cited
  - lists joined with `"; "`, not `", "`
- [ ] `doc_id = sha1(cnr or path)[:16]` (today it is `sha1(filename)`).
- [ ] Bench labels: Single / Division (2) / Full (3+) / Constitution Bench, taken from the parsed bench list. Remove the "Single Bench" default.
- [ ] SQLite `documents` table (one row per judgment) next to `parent_chunks`.
- [ ] Join on the manifest. Replace the Kaggle file-name metadata (`_vs_`, `" on <date>"`, CIVIL/CRIMINAL/SPACE folder, year taken from the stem, court defaulting to "High Court").
- [ ] Resource guard:
  - measure the RSS of the whole process tree (parent + children), not just this process
  - `MAX_RAM_GB=18`, `WARN_RAM_GB=15`
  - stop walking the whole store with `os.walk` on every batch (slow on NFS)
- [ ] `NUM_THREADS` default 8 → 14; call `torch.set_num_threads` in the embedding engine as well.
- [ ] Extract PDFs with a pool of 6 workers (today it is one file at a time).
- [ ] Embedding batch size: `VectorStoreManager` hard-codes 32, and `--embedding-batch-size` only sets the upsert batch. Pass 64 through to the encoder.
- [ ] transformers 5: `torch_dtype=` → `dtype=` (`local_embeddings.py` AutoModel fallback, ST `model_kwargs`).
- [ ] BGE query prefix `"Represent this sentence for searching relevant passages: "`. It is probably not applied today: the code uses only a `query` prompt defined in the model's sentence-transformers config, and bge-small likely has none. Check on the server.
- [ ] CPU-safe defaults:
  - embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`, collection `sc_criminal_cpu`
  - remove the zembed-only VRAM swap and the bf16 assumptions from the defaults
  - update the B200/MIG comments in `.env.example`
- [ ] `index_corpus.py:36`: `LOG_DIR = Path("data/logs")` writes into the git folder at import time. Move it to `$LOGS_DIR` in the workspace.
- [ ] One checkpoint per collection, and SQLite parents/documents keyed by collection:
  - after the pilot, the full run would otherwise skip the 100 pilot PDFs
  - `--reset` must not wipe the other collections' parents
- [ ] Unit tests:
  - write only into a temp dir; today they insert a fake parent into the production SQLite and write `failed_files.json` into the real processed dir
  - make no real LLM calls (`test_rag_pipeline.test_05` can call the API)
- [ ] `scripts/smoke_test.py` must work on CPU. Today it asserts `torch.cuda.is_available()` and runs a bf16 SDPA check on `device="cuda"`.
- [ ] Import `google.genai` lazily. Today it is imported at module top in `vector_store`, `document_loader`, `rag_pipeline`, `llm_budget` and `evaluate_pipeline`.
- [ ] `requirements.txt`: set minimums to the verified versions (chromadb 1.5, sentence-transformers 6, transformers 5, pypdf 7) and add bm25s, pandas, pyarrow and boto3.
- [ ] `scripts/setup_server.sh` builds `./venv` inside the repo with /tmp constraints. Replace it with the `~/Pratik/venv` + `~/Pratik/pins.txt` procedure, or remove it.

## Phase 4: retrieval and synthesis

- [ ] Double-sigmoid check: if ST 6 `CrossEncoder.predict` already applies a sigmoid to 1-label models, `rag_pipeline` applies a second one. That squashes scores to ~0.50–0.73, so the ±5% prior outweighs relevance. Print raw scores on the server, then fix.
- [ ] Chroma cosine check: `metadata={"hnsw:space": "cosine"}` is the legacy setting in chromadb 1.x. Confirm a new collection really uses cosine, otherwise `1 - distance` is wrong. Also check the import path `chromadb.api.models.Collection`.
- [ ] Concordance regex: the Act name is optional, so "Section 107 IPC" also adds "Section 126 BNSS". Require the Act name for each mapping.
- [ ] Authority prior from `bench_strength` (D6: Constitution > 3-judge > 2-judge), bounded at ±5% and switched off by `AUTHORITY_STRENGTH=0`. It replaces court tier × role weight × judge affinity.
- [ ] Filters go through SQLite `documents` → doc_ids → Chroma `where={"doc_id":{"$in":[...]}}` and BM25 `allowed_ids`. They cover year range, minimum bench strength, and statute with its concordance equivalent.
- [ ] Switch BM25 to `bm25s` (today rank_bm25 scores every chunk in a Python loop).
- [ ] Synthesis header `[Precedent N] title | citation | date | bench_type: judges | author | disposal | role`. Prompt rules:
  - cite only `[Precedent N]`
  - separate ratio from obiter; prefer larger benches; flag conflicts with dates
  - offences before 1 July 2024 are governed by the IPC
  - say "insufficient precedent in the indexed corpus" when needed
  - end with "Verify on the official source."
- [ ] Citation check (no LLM): `citation_precision`, unknown citations, UI badge.
- [ ] Remove the unused top-level `import torch` in `rag_pipeline.py`.

## Phase 6: evaluation

- [ ] Metrics: Hit@1/3/5/10, MRR@10, nDCG@10, Recall@10, latency mean/p50/p95, 95% bootstrap CIs. Today it reports only Hit@1/3/5 and MRR@5, with top_k=5.
- [ ] Ground truth matched on `doc_id` / `cnr`, not the file stem.
- [ ] `run_manifest.json` plus a compact `results.json` (<15 KB) and `results.md` in `$BENCHMARKS_DIR/v2/`.
- [ ] Remove `CORPUS_CASE_ALIASES` and the fuzzy title-match fallback in `evaluate_pipeline.py`, and update `tests/test_evaluation.py`.
- [ ] `generate_synthetic_queries.py`: group by doc_id instead of the file stem; drop `--path-filter sc_2025`.
- [ ] **Leakage rule:** synthetic queries are generated only from BODY sections (Ratio/Analysis, Facts), never from Headnote parents. A headnote is the editor's summary of the holding, so a query written from it would match the headnote chunk almost word for word. Every chunk records `section_kind = headnote | body` (set in Phase 3) so the generator can filter on it.

## Phase 7: demo and docs

- [ ] README headline numbers (400 Kaggle judgments, zembed-1, Hit@5 1.00) are from the earlier GPU run. Label them as a separate earlier experiment and take the new numbers only from `results/v2/results.json`.
- [ ] Kaggle leftovers: `scripts/download_kaggle_corpus.py`, `kagglehub` in requirements, the README clone URL and `~/adkmajorproject`, `server._clean_title` (Kaggle " 1" suffix), the README `MAX_RAM_GB` 24 setting.
- [ ] `server._doc_key` uses the file stem → switch it to `doc_id`.

## No change needed (notes)

- On CPU, the embedding engine and reranker already pick the device themselves and run in fp32. The CUDA branches can stay for the stretch GPU run.
- Keep `LOCAL_LLM_FALLBACK=false`: Qwen-7B in fp32 on CPU needs about 30 GB, over the 20 GB limit.
- pypdf 7: only `PdfReader` and `extract_text` are used, so the risk is low.
- The 4-config ablation, checkpointing, deterministic IDs and the LLM provider chain/cache stay (CLAUDE.md rule 2).
