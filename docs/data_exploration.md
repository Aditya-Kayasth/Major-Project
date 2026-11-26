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

## Findings and rules adopted

1. **Bench data gap.** In 11 records the `Coram :` list names 1 judge while the trailing block declares `Bench : 2 Judges`.
   - Rule: `bench_strength = max(coram_count, declared_count)`.
   - `bench_source` = `coram` | `declared` | `pdf` (the last is set only by the Phase 3 PDF fallback).
   - Implemented in `src/sc_metadata.resolve_bench`.
2. **No Coram (3 records).** These are flagged for a PDF page-1 fallback in Phase 3. If the declared count exists, it is used meanwhile.
3. **Author.** Order of preference:
   - the Coram name marked `*`
   - otherwise the parquet `judge` column, if it holds exactly one name
   - otherwise unknown

   `author_source` = `coram_star` | `judge_column` | `unknown`; implemented in `src/sc_metadata.resolve_author`.
4. **Headnotes.** `raw_html` carries labelled headnotes only from 2023, and cut short. The headnote role is taken from the PDFs in Phase 3. Earlier records keep only a `subject_hint` (for example `Code of Criminal Procedure, 1973`).
5. **Citation.** `neutral_citation` is missing for 15 records. The display citation falls back to the SCR citation (`[YYYY] N S.C.R. P`).
6. **Scope of "criminal".** Selection by case number includes matters that are not IPC offences, for example an Income Tax s.276C prosecution and maintenance/paternity disputes heard as criminal appeals. They are kept on purpose because the selection rule is the case number. This will be stated in `docs/DATA_CARD.md`.
