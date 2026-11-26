# Data card: Supreme Court of India criminal judgments, 2018–2025

## Source

- **Dataset:** "Indian Supreme Court Judgments", AWS Open Data Registry.
  - Registry page: https://registry.opendata.aws/indian-supreme-court-judgments/
  - S3 bucket `arn:aws:s3:::indian-supreme-court-judgments`, region `ap-south-1`, public, read with no account.
- **Managed by:** Dattam Labs (https://dattam.in), per the registry page. Update frequency on the registry: bi-monthly.
- **What it contains:** per the registry page, judgments downloaded from the eCourts website, 1950–2025, with metadata in JSON and Parquet, and judgments in English and regional languages.
- **Licence:** **CC-BY-4.0**, as stated in the registry page's License field (checked 27 Nov 2026). The page gives no separate licence link.
- **How to cite** (the registry's own text): "Indian Supreme Court Judgments was accessed on 26 November 2026 from https://registry.opendata.aws/indian-supreme-court-judgments."
- **What we used:**
  - `metadata/parquet/year=YYYY/metadata.parquet` for 2018–2025
  - the English PDFs at `data/pdf/year=YYYY/english/{path}_EN.pdf`
- All of these are reported judgments, published in the Supreme Court Reports (S.C.R.).

## Selection

- **Scope:** every Supreme Court judgment from 2018 to 2025 whose **case number** contains `CRIMINAL` or `CRL`.
  - The case number is taken from the `raw_html` metadata field: `Case No :</span><font …> CRIMINAL APPEAL No. 645/2019`.
  - No sampling.
  - De-duplicated on (`cnr`, `decision_date`): 0 duplicates.
- **Excluded on purpose:** judgments that mention "criminal" but are civil matters by case number, such as Civil Appeals and Writ Petitions (Civil), including PIL guideline cases. There are 82 of them (count verified on the server, 26 Nov, CLAUDE.md D2).
- **Included on purpose:** the selection rule is the case number, so a few criminal-side matters that are not IPC offences stay in. Examples are an Income Tax s.276C prosecution and maintenance/paternity disputes heard as criminal appeals.

| Year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Total |
|---|---|---|---|---|---|---|---|---|---|
| Criminal judgments | 185 | 264 | 132 | 196 | 244 | 250 | 273 | 347 | **1,891** |

| Case type (raw label in the metadata) | Count |
|---|---|
| CRIMINAL APPEAL | 1,723 |
| SPECIAL LEAVE PETITION (CRIMINAL), incl. 4 "SPECIAL LEAVE TO PETITION" | 76 |
| WRIT PETITION (CRIMINAL) | 56 |
| REVIEW PETITION (CRIMINAL) | 13 |
| TRANSFER PETITION (CRIMINAL) | 6 |
| SUO MOTO WRIT PETITION (CRIMINAL) | 6 |
| CONTEMPT PETITION (CRIMINAL) + SUO MOTO CONTEMPT PETITION (CRIMINAL) | 3 + 3 |
| CRIMINAL MISCELLANEOUS PETITION | 2 |
| TRANSFERRED CASE (CRIMINAL) | 2 |
| CURATIVE PETITION (CRL) | 1 |

Source: `scripts/aws_sc_explore.py` (Phase 1); see `docs/data_exploration.md`.

## Corpus actually indexed

| Item | Value |
|---|---|
| Selected | 1,891 |
| English PDF downloaded and valid | **1,889** (564.8 MB) |
| Not found on S3 (HTTP 404) | 2 (`2023_10_160_288`, `2023_3_30_compressed`) |
| Indexed | 1,889 judgments → 52,101 parent sections → 129,707 passages (chunks) |
| Headnote separated from the judgment text | 1,883 (6 could not be split) |

## Metadata per judgment

- Title, petitioner, respondent, decision date, neutral citation (`YYYY INSC N`, else the S.C.R. citation), CNR, case number and type, and disposal nature all come from the official metadata.
- **Bench** is parsed from the `Coram :` line, cross-checked against `Bench : N Judges`. `bench_strength` is the larger of the two; in 11 records Coram lists 1 judge while 2 are declared. The PDF page 1 is used for the 3 judgments without a Coram line.
- **Author.**
  - **Finding:** the `*` marker in the metadata usually marks the presiding, first-listed judge, not the author. In the 20-judgment test index, 8 of the 11 inspected disagreements named another judge in the PDF.
  - **Rule:** the author therefore comes from the PDF first. The evidence used is the `* Author` footnote, the `[NAME, J.]` running header, "delivered by …", or the opening `NAME, J.`. The metadata is only a fallback.
  - **Result:** PDF 1,711, metadata 136, unknown 36. The original metadata value is kept as `author_manifest`.
- **Statutes cited** are extracted from the text by regex for 11 Acts (IPC, CrPC, Evidence Act, BNS, BNSS, BSA, NDPS, POCSO, Arms Act, SC/ST (PoA), PC Act). At least one section was found in 1,751 of 1,889 judgments.
- **REPORTABLE** is unknown for the whole corpus: the marker is not present in S.C.R. PDFs.

## Known gaps and limitations

- **Coverage.** Supreme Court only (no High Courts), criminal by case number only, 2018–2025 only, English PDFs only.
- **Text extraction.** pypdf extraction with heuristic cleaning: margin letters, running headers, page numbers, signatures. Ligature-split words are rejoined after Unicode NFKC normalisation; some OCR-style artefacts may remain.
- **Role tags.** Ratio/Analysis, Facts, Submissions and the other roles are keyword heuristics. About 47% of chunks were `Unclassified` in the 20-judgment test index. Headnote chunks are tagged from the layout, not from keywords.
- **Headnotes.** These are editorial summaries, not the court's words. Benchmark queries are never generated from them.
- **No overruling or treatment information.** A judgment's later reversal or overruling is not tracked.
- **Dataset quirks.** Metadata fields can be empty (`description`, `author_judge`); the `*` author marker is unreliable (see above).

## Intended use

Academic research prototype for precedent retrieval. **Not legal advice.** Every result must be verified on the official source.
