# CLAUDE.md — Indian Legal Precedents RAG (Major Project, final build)

> Lives in the repo root. Claude Code reads it at the start of every session.
> Work phase by phase (Section 8). Read Sections 1–6 before doing anything.

---

## 1. Context

- **What this is:** final-year B.E. academic major project. Final evaluation is a 20-minute presentation plus a live demo, judged by faculty and an external industry person.
- **Framing:** an *academic research project*, not a product. Our claim: "we built a measured, reproducible precedent-retrieval system for Indian criminal law and showed which components matter."
- **User problem:** junior lawyers spend hours finding precedents. A lawyer types a legal question **or a set of facts**. The system returns ranked Supreme Court judgments, the exact passages, and a grounded answer citing only those judgments.
- **Not claimed:** replacing legal research, full overruling tracking, all courts. A practising Mumbai lawyer confirmed the full problem needs a specialised search engine. We solve a measurable part of it.
- **Deadline:** the whole pipeline must run end to end by tomorrow morning. Prefer simple, working and measured over clever.

## 2. How we work: PC → GitHub → server (IMPORTANT)

```
 PC (Windows, VS Code + Claude Code)  --git push-->  GitHub  --git pull-->  College server (CPU)
   - all code edits                                                  - downloads data from AWS S3
   - git only, nothing runs on the PC                                - data, tests, indexing, eval, demo
                                         <--- user pastes summaries/results back ---
```

- **You (Claude Code) run on the PC.** You can't reach the server. All server work happens as commands **you give me**. I run them in a web terminal (one command box, persistent shell) and paste the output back.
- So every server step must be:
  1. one copy-pasteable command block, starting with `cd ~/Pratik/Major-Project && git pull && source ~/Pratik/venv/bin/activate`
  2. long jobs with `nohup ... > ~/Pratik/workspace/logs/<step>.log 2>&1 &`, plus a separate status command
  3. scripts that print a **compact summary (≤60 lines)** made to be pasted back, with full detail in the log file
- **Never** ask me to edit server files by hand, except `.env`.
- **Code only goes to GitHub from the PC.** The server only ever pulls. Server outputs (results.json and the like) come back to the PC by paste, and you write them into the repo.
- Before telling me to run something on the server, make sure the code is committed and pushed. Give me the exact `git add / commit / push` commands, or run them yourself on the PC.
- **The PC runs NOTHING.** No Python, no venv, no downloads, no tests on the PC. You only read and write code and run git there.
  - All execution (data exploration, unit tests, indexing, evaluation, charts, demo) happens on the server through commands you give me.
  - Unit tests run on the server: `python -m unittest discover tests`.
- **Go one step at a time.** Give me ONE server command block per reply, wait for my pasted output, then continue. Never give me several phases at once.

## 3. Server environment (facts, checked 26 Nov 2026)

- Ubuntu 24.04, **Python 3.12.3**, system PyTorch **2.11.0+cu128**, transformers 5.17.0, numpy 2.4.4, pandas 3.0.5, pyarrow 25, fastapi 0.136, openai 3.16, uvicorn 0.53.
- **GPU: NOT available.** NVML fails and `torch.cuda.is_available()` is False. **The plan is CPU-only.** Code must auto-detect the device (`cuda` if available, else `cpu`) and run fully on CPU in fp32: no bf16/fp16 on CPU, no hard `.cuda()`. Never use `nvidia-smi` or `pynvml`. A watcher logs GPU status to `~/Pratik/gpu_watch.log`.
- CPU: 16 cores available. Use `NUM_THREADS=14` and call `torch.set_num_threads`.
- **Measured on the server (26 Nov, 18:30, CPU, 14 threads):**
  - bge-small-en-v1.5: 384-dim, **170 chunks/sec** (~800-char chunks, batch 64)
  - bge-reranker-base: **0.42 s per 20 pairs**
  - process RAM: 2.0 GB with both models loaded
- **Verified venv:** torch 2.11.0+cu128 (system, CPU) | transformers 5.17.0 | pandas 3.0.5 | numpy 2.4.4 | sentence-transformers 6.3.0 | chromadb 1.5.9 | bm25s 0.3.16 | pypdf 7.1.1 | duckdb 2.0.1 | boto3 1.44.12 | onnxruntime 1.32.0.
  - Write code against THESE APIs: chromadb 1.x, sentence-transformers 6.x, transformers 5.x, pypdf 7.x. Fix any removed or renamed APIs and say what changed.
  - Update `requirements.txt` with matching minimum versions.
- `.env` already exists on the server with absolute workspace paths and the CPU models (created 26 Nov). Add new keys only by giving me a `printf ... >> .env` command.
- **RAM: hard limit 20 GB for everything we run together** (indexing + server + anything else). Set `MAX_RAM_GB=18` and `WARN_RAM_GB=15`.
  - The resource guard must count the **whole process tree** (parent + children).
  - Indexing and the web server must not run at the same time.
  - Process the corpus in micro-batches; never load all PDFs or texts at once.
- Storage: NFS home with 13 TB free, so it isn't a constraint.
- **Do not modify the system Python environment.** Everything goes into the venv `~/Pratik/venv`, created with `--system-site-packages` so it reuses the system torch and transformers.
  - Install only with: `pip install -c ~/Pratik/pins.txt <pkg>` (pins torch, transformers, numpy, pandas, tokenizers, safetensors, huggingface-hub, pydantic, fastapi).
  - If pip wants to change a pinned package, stop and tell me.
- **Layout on the server (all under `~/Pratik`):**
  ```
  ~/Pratik/Major-Project/        git clone (code only)
  ~/Pratik/venv/                 venv
  ~/Pratik/pins.txt              version pins
  ~/Pratik/workspace/data/       raw PDFs, metadata, manifest
  ~/Pratik/workspace/store/      chromadb, sqlite, bm25, checkpoints
  ~/Pratik/workspace/benchmarks/ queries, results, llm cache
  ~/Pratik/workspace/logs/       logs
  ~/Pratik/workspace/hf_cache/   HuggingFace model cache (HF_HOME)
  ```
  All paths come from `.env` as absolute paths (`src/config.py` already supports this). Add `METADATA_DIR` to the config. Never write data into the git folder.
- LLM: Groq (`LLM_API_KEY`, gpt-oss-120b with fallbacks) and Gemini (`GEMINI_API_KEY`). Free tiers; every call goes through `src/llm_budget.py`, which caches.
- **Nothing is trained or fine-tuned.** All models are pretrained. "Data" means the corpus we index and the benchmark queries.

## 4. Hard rules

1. **Honesty:** never invent or estimate a metric. Every number in README, slides or UI comes from a file a script produced. If something wasn't run, say "not run".
2. **Don't break what works:** keep the 4-config ablation, resource guard, checkpointing, deterministic IDs, and the LLM provider chain and cache.
3. **Inspect real data before writing code that depends on it.**
4. Small changes. Commit and push after each step. Unit tests run on the server (`python -m unittest discover tests`) as part of the next server command.
5. Never print or commit `.env` values.
6. **Reporting style:** simple bullets. Before a phase, a 3-line plan. After it: files changed, the push command, the server command block, what output to paste back.

## 5. What exists today (read README.md and src/)

- `index_corpus.py`: streamed indexing with checkpoints and the resource guard.
- `src/document_loader.py`: pypdf text, regex metadata, parent/child chunking with rhetorical-marker flushing and role tags.
- `src/vector_store.py`: ChromaDB (dense) plus `BM25Registry` (`rank_bm25`, pickled).
- `src/rag_pipeline.py`: concordance, then dense + BM25, then RRF, then cross-encoder × bounded authority prior, then parent expansion, then grounded synthesis.
- `src/sqlite_store.py`, `src/concordance.py` (IPC/CrPC/IEA ↔ BNS/BNSS/BSA), `src/local_embeddings.py`, `src/resource_guard.py`.
- `evaluate_pipeline.py` (4-config ablation plus LLM judge), `scripts/generate_synthetic_queries.py`, `server.py` with `web/`.

**Known weaknesses to fix:**
- Corpus: 400 random Kaggle PDFs. Being replaced.
- Metadata from file name and regex; bench type defaults to "Single Bench". Replaced by official metadata.
- 400-char child chunks (~112 per judgment).
- `BM25Registry.query` scores every chunk in a Python loop.
- 40 synthetic queries only; no fact-pattern queries, no CIs, no citation check.
- GPU-only assumptions (bf16, zembed-1, CUDA paths).

## 6. Decisions (already made, don't re-open)

**D1. Data:** AWS Open Data bucket `indian-supreme-court-judgments` (public, ap-south-1, no account: boto3 `UNSIGNED` or HTTPS `https://indian-supreme-court-judgments.s3.amazonaws.com/<key>`).
- **Verified facts (checked on the server, 26 Nov):**
  - Metadata `metadata/parquet/year=YYYY/metadata.parquet` is already downloaded for 2010–2025 to `$METADATA_DIR/sc/year=YYYY/metadata.parquet` (14 MB). Reuse it; don't re-download.
  - Columns: `title, petitioner, respondent, description, judge, author_judge, citation, case_id, cnr, decision_date, disposal_nature, court, available_languages, raw_html, path, nc_display, scraped_at, year`.
  - **PDF key = `data/pdf/year=YYYY/english/{path}_EN.pdf`** (e.g. `path=2019_16_971_992` → `.../2019_16_971_992_EN.pdf`).
  - `case_id` is the **neutral citation** (`2019 INSC 1193`), NOT the case type. `nc_display` = `2019INSC1193`. `citation` = SCR citation (`[2019] 16 S.C.R. 971`). These are reported (SCR) judgments.
  - `decision_date` format is `DD-MM-YYYY`. `description` is often empty. `author_judge` is often `<NA>`.
  - **`judge` usually holds ONE name** (only 14% have several). Don't use it for bench strength.
  - **`raw_html` is the rich field.** It is mostly web-page boilerplate, but it contains:
    - `| Case No :</span><font color='green'> CRIMINAL APPEAL No. 645/2019</font>`, the **case type and number**
    - the SCR header, e.g. `MISS XYZ v. STATE OF GUJARAT & ANR (Criminal Appeal No.1619 of 2019) OCTOBER 25, 2019 [UDAY UMESH LALIT, INDU MALHOTRA AND R...]`, the **full bench in square brackets**
    - possibly headnote text
  - Parse `raw_html` with regex/BeautifulSoup to get `case_no`, `case_type` and the bench list. Fallback for the bench: the PDF's first page (same `[NAME, NAME AND NAME, JJ.]` header).
  - English PDF sizes 2018–2025: 6,674 files, ~2.56 GB in total, avg ~200–525 KB per file.
- English only.

**D2. Scope (LOCKED):** **all Supreme Court criminal judgments 2018–2025 — about 1,970 judgments, ~0.75 GB of PDFs.**
- Criminal = the `Case No` value in `raw_html` matches `CRIMINAL|CRL|Crl\.` (e.g. CRIMINAL APPEAL, SLP (CRL.), REVIEW PETITION (CRIMINAL), WRIT PETITION (CRIMINAL)).
- A loose "criminal anywhere in raw_html" match gives 1,973 for 2018–2025 (per year: 193 / 272 / 144 / 207 / 260 / 259 / 283 / 355). The strict Case No filter may give slightly fewer. Report both.
- No sampling: index all of them. Extend to 2015–2017 (+~600) only if I ask.
- Domain: IPC/BNS, CrPC/BNSS, Evidence Act/BSA, NDPS, POCSO, etc. Typical issues: murder, attempt, culpable homicide, dowry death, rape, bail, quashing, evidence, sentencing.
- High Court data is out of scope (stretch only).

**D3. Vector DB:** keep **ChromaDB**. No Qdrant (future scope for full-corpus scale).

**D4. Models (CPU plan):**
- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, 33M parameters). Queries get the BGE instruction prefix `"Represent this sentence for searching relevant passages: "`; documents get none.
- Reranker: `BAAI/bge-reranker-base`. If the pilot shows p50 query latency under 2 s with `BAAI/bge-reranker-v2-m3`, use v2-m3 (continuity with earlier results).
- Rerank the top 20 (top 30 if latency allows).
- LLM: gpt-oss-120b via Groq for answers; Gemini as judge.
- Collection name: `sc_criminal_cpu`.
- If the GPU comes back, a zembed-1 run goes into a separate collection as an extra comparison row. Never mix the two.

**D5. Chunking:** parent 2,000 chars, child 800 chars, overlap 120. Keep the rhetorical-marker flushing, role tags and the `[role | title]` prefix. Add a `Headnote` role when an e-SCR headnote block ("Issue for Consideration" / "Headnotes") appears before the judgment body.

**D6. Hierarchy from metadata:**
- `bench_strength` = number of judges in the bench list parsed from `raw_html` (SCR header in square brackets), else from PDF page 1. NOT from the `judge` column.
- `bench_type`: 1 Single, 2 Division (2), 3–4 Full (3+), ≥5 Constitution Bench.
- Authority order: Constitution > 3-judge > 2-judge.
- Stays a **bounded ±5% tie-breaker**; `AUTHORITY_STRENGTH=0` switches it off for the ablation.

**D7. Every child chunk carries the document metadata in Section 7.**

## 7. Target metadata schema

Chroma metadata must be scalar; lists are stored as `"; "`-joined strings.

```text
doc_id            sha1(cnr or path)[:16]
parent_id         <doc_id>-p<section>
chunk_id          <doc_id>-p<section>-c<chunk>
case_title        parquet `title`
petitioner        parquet
respondent        parquet
court             "Supreme Court of India"
decision_date     "YYYY-MM-DD"
year              int
judges            "; "-joined names
bench_strength    int
bench_type        Single | Division (2) | Full (3+) | Constitution Bench
author_judge      parquet
citation          nc_display, else `citation`
case_id           parquet (verify real format)
cnr               parquet
disposal_nature   parquet
case_type         derived: Criminal Appeal | SLP (Crl) | Writ (Crl) | ...
reportable        bool, if detectable
statutes_cited    "IPC 302; IPC 34; CrPC 313; Evidence Act 32" (doc level)
role              Headnote | Facts | Issues | Submissions | Precedents Cited | Ratio/Analysis | Order | Unclassified
source_path       local PDF path
raw_text          clean chunk text for the UI
```

SQLite gets a `documents` table (one row per judgment) next to the parent table.

## 8. Phases

### Phase 0: Orient (PC, no code changes)
- Read the whole repo. Report in under 15 bullets: the architecture in your own words, every GPU-only assumption (bf16, `.cuda()`, zembed-specific code, cuDNN switch), every hard-coded path, and anything that contradicts this file.

### Phase 1: Parse raw_html and build the selection (PC writes, server runs)
- The metadata is already downloaded (D1). Write `src/sc_metadata.py` with a parser for `raw_html` → `case_no`, `case_type`, `bench` (list), `bench_strength`, plus headnote text if present. Add unit tests built on real snippets I paste.
- Write `scripts/aws_sc_explore.py`:
  - apply the parser to 2018–2025
  - print, per year: total, loose criminal count, strict Case No criminal count, the distribution of `case_type` values, the share with a parsed bench, and the `bench_strength` distribution
  - show 5 parsed examples and 5 failures
  - HEAD-check 5 PDF keys (`{path}_EN.pdf`)
- The script prints a ≤60-line summary for me to paste back, and writes the full report to `$METADATA_DIR/sc_explore_report.md`. After I paste the summary, you save it as `docs/data_exploration.md` in the repo. **Stop and wait for my paste.**

### Phase 2: Download script and manifest (PC writes, server runs)
- `scripts/download_aws_sc.py --years 2018-2025 --criminal-only --workers 8` (plus `--max-docs N` for tests only)
  - Select with the strict Case No filter from Phase 1. Dedupe on `(cnr, decision_date)`. No sampling: all selected judgments.
  - Resumable HTTPS/boto3 downloads with retries; check the `%PDF` header.
  - Save to `$RAW_DATA_DIR/sc_aws/year=YYYY/`.
  - Write `$METADATA_DIR/manifest.parquet` (and `.csv`) with all metadata, `local_path` and `doc_id`.
  - Log failures. Fallback: the year's tar, extracting only the selected members.
- First give me a server test command with `--max-docs 10`. After I paste the result, give me the full run (nohup) and a status command.

### Phase 3: Loader uses official metadata (PC writes, test on the server with 10–20 PDFs)
- Join on the manifest; fall back to regex only for missing fields and mark them `inferred`.
- Parse `bench_strength`, `bench_type`, `case_type`. Extract `statutes_cited` (IPC, CrPC, Evidence Act, BNS, BNSS, BSA, NDPS, POCSO, Arms Act, SC/ST (PoA), Prevention of Corruption).
- Cleaning: remove headers and footers, page numbers, "Signature Not Verified" / "Digitally signed by" blocks. Store REPORTABLE as a flag.
- Headnote role. D5 chunk sizes from `.env`. SQLite `documents` table.
- **CPU-safe models:** auto device, fp32 on CPU, BGE query prefix, `torch.set_num_threads(NUM_THREADS)`.
- **PDF extraction in parallel** with a multiprocessing pool of 6 workers. Embedding stays in the main process with batch size 64.
- Resource guard: measure RSS of the whole process tree against `MAX_RAM_GB`.
- Tests: manifest join, bench parsing, statute regex, headnote detection, device selection.

### Phase 4: Retrieval upgrades (PC writes, server tests)
1. BM25: switch to `bm25s` (same tokenizer and interface, `allowed_ids` supported). If it can't be installed, use numpy `argpartition` instead of the Python loop.
2. Authority prior from `bench_strength` (D6).
3. Filters: year range, minimum bench strength, statute section including its concordance equivalent (IPC 307 ↔ BNS 109). Path: SQLite `documents` → `doc_id`s → Chroma `where={"doc_id":{"$in":[...]}}` and BM25 `allowed_ids`.
4. Synthesis header per precedent: `[Precedent N] <title> | <citation> | <date> | <bench_type>: <judges> | author | disposal | role`.
   - The prompt makes the model: cite only `[Precedent N]`; separate ratio from obiter; prefer larger benches; flag conflicts with dates; apply the IPC↔BNS rule that offences before 1 July 2024 are governed by the IPC; say "insufficient precedent in the indexed corpus" when needed; end with "Verify on the official source."
5. Citation check (no LLM): `citation_precision`, unknown citations in the response JSON, and a UI badge.
6. (P1) Fact-pattern mode: one cached LLM call rewrites facts into `{legal_issues, likely_sections, key_facts}`. Run hybrid on both the original and the rewrite, fuse with RRF, rerank against the original. Must be toggleable.

### Phase 5: Pilot, then full indexing (server)
- Give me one command: download a 100-judgment pilot if it isn't already there, then index it into `sc_criminal_cpu_pilot`.
- Print docs/min, chunks/doc, peak RAM (process tree) and p50/p95 latency of 10 test queries (both rerankers).
- From that, project the time and RAM for the full ~1,970 judgments. **I choose the reranker.**
- Then the full run with nohup, and a status command (progress + RAM).

### Phase 6: Evaluation (PC writes, server runs)
- **Benchmark v2:**
  - 100 legal-issue queries (stratified by year, seed 42, from holding sections)
  - 50 fact-pattern queries (from Facts sections, written the way a lawyer briefs facts: no names, dates, case numbers or citations)
  - a leakage filter (report rejections)
  - `human_queries.json` template with 20 slots, which the team fills in
  - 5 out-of-scope queries that should be refused
- **Metrics:** Hit@1/3/5/10, MRR@10, nDCG@10, Recall@10, latency mean/p50/p95, 95% bootstrap CIs (1,000 resamples), breakdowns by query type and year. Ground truth matched on `doc_id`/`cnr`.
- **Configs:**
  1. Dense
  2. BM25
  3. Hybrid RRF
  4. Full funnel
  5. Full without the authority prior
  6. Full with fact-pattern mode (fact queries only, if P1 is done)
- **Generation** (20–30 queries, cached): Gemini judge (groundedness, ratio/obiter, statutory accuracy, reasoning), citation precision, refusal rate on out-of-scope queries.
- **Server output:** `results.json` (compact, under 15 KB) and `results.md` in `$BENCHMARKS_DIR/v2/`, plus `run_manifest.json` (git commit, device, models, chunk settings, corpus counts, seed).
- **I paste `results.json` to you.** You save it to `results/v2/results.json` in the repo and write the README tables from it.
- **Charts:** no Python on the PC. Add an **"Evaluation" page to the web UI** that draws the charts in the browser (plain JS/SVG, no build step) from `results.json`: Hit@k by config, MRR with CIs, latency, by query type. It doubles as the results slide in the live demo.

### Phase 7: Demo and docs
- **UI:** metadata chips, filters, fact-pattern toggle, citation badge, source passages, stage-timing panel, footer "Academic research prototype, not legal advice."
- **Demo queries:** pick 5 after indexing, from what's really in the corpus, and verify each one:
  1. section-based question
  2. fact pattern
  3. BNS query hitting IPC judgments
  4. filter demo
  5. out-of-scope refusal
  - Save to `data/demo_queries.json` (in git). Give me one command that runs them all on the server, to warm the LLM cache.
- **Web demo:** `server.py` runs on the server with nohup. Give me the cloudflared or port-forward command.
- **README:** numbers only from `results/v2/results.json`. Also `docs/DATA_CARD.md` (source, licence from the registry page — don't guess — selection criteria, counts, gaps).
- **Limitations:**
  - CPU-only small models (GPU unavailable)
  - synthetic queries favour BM25
  - small human set
  - no overruling detection
  - SC criminal only
  - heuristic role tags
- Keep the earlier GPU run (400 Kaggle judgments, zembed-1, Hit@5 1.00) labelled as a separate earlier experiment, never in the same table.

## 9. Definition of done
- [ ] Corpus indexed on the server (N and the years come from the manifest).
- [ ] Unit tests and the smoke test pass on the server.
- [ ] Benchmark v2 run; `results/v2/results.json` is in the repo; README tables and the UI Evaluation page come from it.
- [ ] Web demo running and reachable; 5 demo queries verified and cached.
- [ ] `docs/DATA_CARD.md` and honest limitations.

## 10. Stretch (only after everything above)
- GPU back: zembed-1 + bge-reranker-v2-m3 run into a separate collection, added as an extra results row.
- Overruling flag (regex "is overruled" / "stands overruled").
- Bombay HC slice (`court=27_1`, criminal, 2024–25, ≤500).
- A note on scaling to the full corpus with Qdrant.
