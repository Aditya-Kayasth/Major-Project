# Data exploration: official SC metadata (Phase 1)

- **Source:** AWS Open Data bucket `indian-supreme-court-judgments`, `metadata/parquet/year=YYYY/metadata.parquet`. It was downloaded earlier to `$METADATA_DIR/sc/` on the server.
- **Script:** `scripts/aws_sc_explore.py --years 2018-2025` (commit `1cb98dc`), with the parser in `src/sc_metadata.py`.
- **Full report:** written by the script on the server to `$METADATA_DIR/sc_explore_report.md` (not in git).

## Results (server run, as reported)

| Item | Result |
|---|---|
| Unit tests (`tests.test_sc_metadata`) | 15/15 OK |
| Criminal judgments 2018–2025 (case number contains CRIMINAL or CRL) | **1,891**: every year matches CLAUDE.md D2 |
| Bench parsed from `Coram :` | 1,888 / 1,891 |
| Coram count ≠ declared `Bench : N Judges` | 11, all Coram = 1, declared = 2 |
| No Coram found | 3 |
| Duplicates on (cnr, decision_date) | 0 |
| `neutral_citation` missing | 15 |
| Headnote labels (`Issue for Consideration` / `Headnotes†`) in raw_html | only from 2023 onwards |
| HTTPS HEAD of 5 `data/pdf/year=YYYY/english/{path}_EN.pdf` keys | 5 / 5 OK |

The per-year table, the case-type distribution, the bench-strength distribution and the examples are in the full report on the server. They are not copied here because the full summary was not pasted back.

Counts per year (CLAUDE.md D2, reproduced exactly by this run): 2018: 185, 2019: 264, 2020: 132, 2021: 196, 2022: 244, 2023: 250, 2024: 273, 2025: 347.

## Download (Phase 2, `scripts/download_aws_sc.py`)

| Item | Result |
|---|---|
| Selected criminal judgments (deduped on cnr, decision_date) | 1,891 (0 duplicates) |
| PDFs downloaded and valid (`%PDF` header) | **1,889** (564.8 MB) |
| Not found on S3 (HTTP 404) | 2: `2023_10_160_288`, `2023_3_30_compressed` |
| `bench_source` | coram 1,877, declared 14, PDF needed 3 |
| `author_source` | coram_star 1,647, judge_column 9, unknown 235 |

**Corpus = 1,889 judgments.** The 2 missing PDFs stay in the manifest with status `not_found` and are not indexed.

## Index (Phase 5, `index_corpus.py --collection sc_criminal_cpu`)

| Item | Result |
|---|---|
| Judgments indexed | **1,889** (0 failed) |
| Child chunks / parent sections | 129,707 / 52,101 |
| Time | 1,747 s (64.9 docs/min) on CPU, 4 spawned extraction workers, 12 torch threads |
| Peak RAM (process tree) | 2.29 GB |
| Headnote split | details 1,870, body_marker 13, none 6 |
| Author | from the PDF 1,711, manifest `coram_star` 136, unknown 36 |
| Disposition (PDF end text vs `disposal_nature`) | agrees 1,667 / 1,690 |
| Judgments with ≥ 1 statute section found | 1,751 / 1,889 |

## Benchmark v3: citation benchmarks (v3a held-out 2026, v3b in-corpus)

- **Idea:** the query is the court-written "Issue for Consideration" of a judgment; the relevant documents are **the precedents the Court itself cited** in that judgment.
- **v3a, held-out:** the citing judgments are **unseen 2026 judgments**.
  - The first build gave 77 criminal 2026 judgments, all with an issue text. They contained 4 SCR, 33 neutral, 371 SCC and 46 AIR citations, plus 453 case names.
  - Of these, 14 neutral citations and 11 name matches mapped to the corpus, leaving **13 queries kept** (mean 1.92 targets). That is too small alone.
- **v3b, in-corpus:** the citing judgments are **our own indexed 2023–2025 judgments** that have an "Issue for Consideration" headnote.
  - Sampling: seed 42, capped at 150 queries.
  - The citing judgment is removed from its own targets, and with `--exclude-self` from **every config's ranking before scoring**. Its own headnote chunk would otherwise sit at rank 1.
  - The removal happens after retrieval. Its chunks can still take some of the reranked slots (top 30 passages), which slightly handicaps the Full configs; we accept that.
- **Name matching** (both variants):
  - normalised separators: v / v. / vs / vs. / versus
  - stripped suffixes: "& Ors.", "and another", "(Dead) through LRs"
  - leading sentence words dropped ("Reliance was placed on …")
  - both party keys, plus the same year 2018–2025, plus exactly one corpus hit

  The build log prints 5 unmatched corpus-year names.
- **Build:** `scripts/build_citation_benchmark.py --mode both`. It writes `$BENCHMARKS_DIR/v3a/citation_queries_v3a.json` and `$BENCHMARKS_DIR/v3b/citation_queries_v3b.json`, each with a build report.
- **Source judgments:** 2026 Supreme Court criminal judgments, selected with the same case-number rule (CRIMINAL / CRL), read from the S3 2026 metadata and English PDFs.
  - They are kept apart from the corpus: stored in `$RAW_DATA_DIR/sc_aws_2026/` with their own `manifest_2026.parquet`, and **never indexed** into `sc_criminal_cpu`.
- **Query:**
  - The "Issue for Consideration" block, or the first headnote paragraph if that block is missing.
  - It is scrubbed of case names ("A v. B"), SCR / neutral / SCC / AIR citations, case numbers, dates, years (except Act years such as "Act, 1881"), and the 2026 case's own party and judge names.
  - A query is dropped if the leakage filter still fires afterwards or fewer than 8 words remain.
- **Targets:** cited judgments that are in our 1,889-judgment corpus. They are matched in three ways:
  1. by SCR citation (`[2019] 4 SCR 208` or `[2019] 4 S.C.R. 208`), using year + volume + start page against the manifest's SCR citation
  2. by neutral citation (`2019 INSC 386`)
  3. optionally, by a conservative case-name match (both party keys + the same year 2018–2025 + exactly one corpus hit), flagged `match_type=name`

  SCC and AIR citations are counted but cannot be mapped (the corpus has no SCC/AIR metadata). A query is kept if it has at least 1 target.
- **Evaluation:** `scripts/evaluate_v2.py --queries citation_queries_v3a.json --tag v3a --skip-llm`, and for v3b `--queries citation_queries_v3b.json --tag v3b --exclude-self --skip-llm`. Both run the same 5 configs, retrieval only, with multi-target metrics:
  - Hit@k = any target in the top k
  - MRR@10 on the first target found
  - Recall@10
  - nDCG@10
  - all with 95% bootstrap CIs

  Output goes to `$BENCHMARKS_DIR/v3a/` and `$BENCHMARKS_DIR/v3b/`.
- **Limitation:** the labels are **incomplete**. A relevant judgment that the 2026 Court did not cite, or cited only by SCC/AIR, counts as a miss. Absolute v3 scores are therefore **lower bounds**; comparisons between configs on the same labels remain fair.
- Build reports and results: *to be filled from `build_report_v3a.json` / `build_report_v3b.json` and `results/v3a/`, `results/v3b/` after the run.*

## Findings and rules adopted

1. **Bench data gap.** In 11 records the `Coram :` list names 1 judge while the trailing block declares `Bench : 2 Judges`.
   - Rule: `bench_strength = max(coram_count, declared_count)`.
   - `bench_source` = `coram` | `declared` | `pdf` (the last is set only by the Phase 3 PDF fallback).
   - Implemented in `src/sc_metadata.resolve_bench`.
2. **No Coram (3 records).** These are flagged for a PDF page-1 fallback in Phase 3. If the declared count exists, it is used meanwhile.
3. **Author (revised after the Phase 3 test index).**
   - **What we saw:** in the 20-PDF test index, the manifest author and the PDF author disagreed for 13 documents. Of the 11 shown by `scripts/check_author_mismatch.py`:
     - 3 were formatting only (`D.Y. CHANDRACHUD` / `DR. DHANANJAYA Y CHANDRACHUD`, `B.V. NAGARATHNA` / `NAGARATHNA`, `BHUSHAN RAMKRISHNA GAVAI` / `B.R. GAVAI`).
     - 8 named a different judge. In all 8 the raw_html `*` (and the parquet `judge` column) pointed at the **first-listed, presiding judge**. The PDF named another bench member, with explicit evidence: the e-SCR `* Author` footnote, the old-SCR `[NAME, J.]` running header on every body page, "The Judgment of the Court was delivered by …", or the opening `NAME, J.` line.
   - **Rule now** (`src/judgment_text.resolve_final_author`):
     1. PDF evidence, mapped onto the bench and written in manifest style (`match_bench`)
     2. else `Per Curiam` for an ORDER
     3. else the manifest's `*` / single-name `judge` column
     4. else unknown
   - `author_source` records which source was used (`pdf_star`, `pdf_running_header`, `pdf_delivered_by`, `pdf_body_signature`, `pdf_order`, `coram_star`, `judge_column`, `unknown`). The manifest value is kept as `author_manifest`.
   - `src/sc_metadata.resolve_author` still produces the manifest-side value.
4. **Headnotes.** `raw_html` carries labelled headnotes only from 2023, and cut short. The headnote role is taken from the PDFs in Phase 3. Earlier records keep only a `subject_hint` (for example `Code of Criminal Procedure, 1973`).
5. **Citation.** `neutral_citation` is missing for 15 records. The display citation falls back to the SCR citation (`[YYYY] N S.C.R. P`).
6. **Scope of "criminal".** Selection by case number includes matters that are not IPC offences, for example an Income Tax s.276C prosecution and maintenance/paternity disputes heard as criminal appeals. They are kept on purpose because the selection rule is the case number. This will be stated in `docs/DATA_CARD.md`.
