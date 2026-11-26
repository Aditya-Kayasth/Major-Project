"""Parser for the official Supreme Court metadata (AWS Open Data, `raw_html` column).

Pure functions, no I/O. Built from real records (see CLAUDE.md D1, checked 26 Nov with
scripts/peek_raw_html.py). After tags are stripped, a record reads:

    [disclaimer, 2018-2019 records only]
    <title> - [YYYY] N S.C.R. P YYYY INSC N Coram : NAME * , NAME
    <subject line / "Issue for Consideration ... Headnotes† ..." (cut short)>
    Decision Date : DD-MM-YYYY | Case No : CRIMINAL APPEAL No. 387/2019 |
    Disposal Nature : Appeal(s) allowed | Bench : 2 Judges Split view HTML view Flip view PDF

- The bench is the `Coram :` list. `*` marks the author (some records have no `*`).
- `Bench : N Judges` in the trailing block is used only as a cross-check.
- Square brackets are NOT the bench: `[2019]` is part of the SCR citation.
"""

import html
import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- regexes
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# Case number as it appears in the HTML (CLAUDE.md D2 selection rule)
CASE_NO_HTML_RE = re.compile(r"Case No\s*:\s*</span>\s*<font[^>]*>\s*([^<]+)", re.IGNORECASE)
CRIMINAL_RE = re.compile(r"CRIMINAL|CRL", re.IGNORECASE)

# Trailing "| ... |" block, on cleaned text
DECISION_DATE_RE = re.compile(r"Decision Date\s*:\s*(\d{1,2}-\d{1,2}-\d{4})", re.IGNORECASE)
CASE_NO_TEXT_RE = re.compile(r"Case No\s*:\s*([^|]+?)\s*(?:\||$)", re.IGNORECASE)
DISPOSAL_RE = re.compile(r"Disposal Nature\s*:\s*([^|]+?)\s*(?:\||$)", re.IGNORECASE)
BENCH_DECLARED_RE = re.compile(r"Bench\s*:\s*(\d+)\s*Judges?", re.IGNORECASE)
TRAILER_START_RE = re.compile(r"Decision Date\s*:", re.IGNORECASE)
VIEW_LINKS_RE = re.compile(r"\s*Split view\s+HTML view\s+Flip view\s+PDF\s*$", re.IGNORECASE)

# Header
SCR_RE = re.compile(r"\[(\d{4})\]\s+(\d+)\s+S\.\s?C\.\s?R\.\s+(\d+)")
NEUTRAL_RE = re.compile(r"\b(\d{4})\s+INSC\s+(\d+)\b")
DISCLAIMER_END_RE = re.compile(r"carrying out the corrections\.\s*", re.IGNORECASE)

# Coram: comma-separated names made of UPPER-CASE tokens ("R.F.", "NARIMAN", "K.V.").
# A token may not be followed by a lower-case letter, so the list stops before the
# Title-case subject line ("Code of Criminal Procedure", "Issue for Consideration").
_TOK = r"[A-Z][A-Z.'\-]*(?![a-z])"
_NAME = rf"{_TOK}(?:[ \t]+{_TOK})*"
_ENTRY = rf"{_NAME}(?:\s*\*)?"
CORAM_RE = re.compile(rf"Coram\s*:\s*({_ENTRY}(?:\s*,\s*{_ENTRY})*)")
AND_SPLIT_RE = re.compile(r"\s+AND\s+")

ISSUE_RE = re.compile(r"Issue for Consideration\s*", re.IGNORECASE)
HEADNOTES_RE = re.compile(r"Headnotes?\W*", re.IGNORECASE)
# A subject line normally ends with the Act's year: "Code of Criminal Procedure, 1973"
SUBJECT_YEAR_RE = re.compile(r"^(.{3,150}?,\s*\d{4})\b")
SUBJECT_SEP_RE = re.compile(r"\s*[:–—]\s*|\s+-\s+")

# Ordered: the first matching rule wins (CLAUDE.md D2 case-type table)
CASE_TYPE_RULES = [
    (re.compile(r"CURATIVE"), "Curative (Crl)"),
    (re.compile(r"REVIEW"), "Review Petition (Crl)"),
    (re.compile(r"TRANSFERRED\s+CASE"), "Transferred Case (Crl)"),
    (re.compile(r"TRANSFER\s+PETITION"), "Transfer Petition (Crl)"),
    (re.compile(r"CONTEMPT"), "Contempt (Crl)"),
    (re.compile(r"SUO\s+MOT"), "Suo Motu Writ (Crl)"),
    (re.compile(r"WRIT"), "Writ Petition (Crl)"),
    (re.compile(r"SLP|SPECIAL\s+LEAVE"), "SLP (Crl)"),
    (re.compile(r"MISC"), "Crl Misc. Petition"),
    (re.compile(r"CRIMINAL\s+APPEAL"), "Criminal Appeal"),
]


# --------------------------------------------------------------------------- helpers
def clean_html(raw: Any) -> str:
    """Strip tags, unescape entities, collapse whitespace. Non-strings give ''."""
    if not isinstance(raw, str):
        return ""
    return WS_RE.sub(" ", html.unescape(TAG_RE.sub(" ", raw))).strip()


def _norm(s: str) -> str:
    return WS_RE.sub(" ", html.unescape(s)).strip()


def parse_case_no(raw_html: Any) -> Optional[str]:
    """Case number from the HTML (`Case No :</span><font ...>`), else from the cleaned text."""
    if not isinstance(raw_html, str):
        return None
    m = CASE_NO_HTML_RE.search(raw_html)
    if m:
        return _norm(m.group(1)) or None
    m = CASE_NO_TEXT_RE.search(clean_html(raw_html))
    return _norm(m.group(1)) if m else None


def is_criminal(case_no: Optional[str]) -> bool:
    """CLAUDE.md D2: criminal = the case number contains CRIMINAL or CRL."""
    return bool(case_no) and bool(CRIMINAL_RE.search(case_no))


def raw_case_type(case_no: Optional[str]) -> Optional[str]:
    """Case number without its number part: 'CRIMINAL APPEAL No. 387/2019' -> 'CRIMINAL APPEAL'."""
    if not case_no:
        return None
    head = re.split(r"\s+No\b\.?|\s+\d", case_no, maxsplit=1, flags=re.IGNORECASE)[0]
    return _norm(head).upper() or None


def normalize_case_type(case_no: Optional[str]) -> Optional[str]:
    """Readable case-type label for criminal matters; the raw type for anything else."""
    raw = raw_case_type(case_no)
    if not raw:
        return None
    if not is_criminal(case_no):
        return raw
    for pattern, label in CASE_TYPE_RULES:
        if pattern.search(raw):
            return label
    return raw


def iso_date(value: Any) -> Optional[str]:
    """'26-02-2019' -> '2019-02-26'. None if the value is not DD-MM-YYYY."""
    if not isinstance(value, str):
        return None
    m = re.fullmatch(r"\s*(\d{1,2})-(\d{1,2})-(\d{4})\s*", value)
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def bench_type(strength: Optional[int]) -> str:
    """CLAUDE.md D6 labels. '' when the bench is unknown."""
    if not strength or strength < 1:
        return ""
    if strength >= 5:
        return "Constitution Bench"
    if strength >= 3:
        return "Full (3+)"
    if strength == 2:
        return "Division (2)"
    return "Single"


def strip_boilerplate(text: str, title: Optional[str] = None) -> str:
    """Drop the disclaimer before '<title> - [YYYY] N S.C.R.' and the trailing view links."""
    start = -1
    if title and isinstance(title, str) and title.strip():
        start = text.find(title.strip())
    if start < 0:
        m = DISCLAIMER_END_RE.search(text)
        start = m.end() if m else 0
    return VIEW_LINKS_RE.sub("", text[start:]).strip()


def parse_coram(text: str) -> Dict[str, Any]:
    """Judges from 'Coram : A * , B'. Returns judges, author (the '*' name) and the match end."""
    m = CORAM_RE.search(text)
    if not m:
        return {"judges": [], "author_judge": None, "end": None}
    judges: List[str] = []
    author: Optional[str] = None
    for entry in m.group(1).split(","):
        for part in AND_SPLIT_RE.split(entry):
            starred = "*" in part
            name = _norm(part.replace("*", ""))
            if not name:
                continue
            judges.append(name)
            if starred and author is None:
                author = name
    return {"judges": judges, "author_judge": author, "end": m.end()}


def parse_trailer(text: str) -> Dict[str, Any]:
    """Fields from the trailing 'Decision Date : ... | Case No : ... | ... | Bench : N Judges' block."""
    dd = DECISION_DATE_RE.search(text)
    cn = CASE_NO_TEXT_RE.search(text)
    dn = DISPOSAL_RE.search(text)
    bd = BENCH_DECLARED_RE.search(text)
    return {
        "decision_date": iso_date(dd.group(1)) if dd else None,
        "case_no": _norm(cn.group(1)) if cn else None,
        "disposal_nature": _norm(dn.group(1)) if dn else None,
        "bench_declared": int(bd.group(1)) if bd else None,
    }


def subject_hint(text: str) -> Optional[str]:
    """First subject line: up to the Act's year ('Penal Code, 1860'), else up to the first ':' or dash."""
    text = text.strip()
    if not text:
        return None
    m = SUBJECT_YEAR_RE.match(text)
    if m:
        return m.group(1).strip()
    first = SUBJECT_SEP_RE.split(text, maxsplit=1)[0].strip()
    return first[:120] or None


def parse_headnote(after_coram: str) -> Dict[str, Any]:
    """Split the text between Coram and the trailing block into issue / headnote / subject.

    Labelled records (2025): 'Issue for Consideration <issue> Headnotes† <headnote>'.
    Unlabelled records (2019): the text starts with a subject line. raw_html cuts this
    text short, so it is only a snippet; the full headnote comes from the PDF (Phase 3).
    """
    after_coram = after_coram.strip()
    issue_m = ISSUE_RE.search(after_coram)
    head_m = HEADNOTES_RE.search(after_coram)
    labelled = bool(issue_m or head_m)

    issue = None
    if issue_m:
        issue_end = head_m.start() if (head_m and head_m.start() > issue_m.end()) else len(after_coram)
        issue = after_coram[issue_m.end():issue_end].strip() or None

    subject_source = after_coram[head_m.end():] if head_m else ("" if labelled else after_coram)
    return {
        "headnote_labelled": labelled,
        "headnote_snippet": after_coram if labelled else None,
        "issue_for_consideration": issue,
        "subject_hint": subject_hint(subject_source),
    }


def resolve_bench(coram_count: int, declared: Optional[int]) -> Dict[str, Any]:
    """Bench strength rule (Phase 1 finding): max(Coram count, declared 'Bench : N Judges').

    11 of 1,891 criminal records list 1 judge in Coram but declare 'Bench : 2 Judges', so the
    larger number wins. bench_source: 'coram' | 'declared' | '' (unknown; Phase 3 tries the PDF).
    """
    coram_count = coram_count or 0
    declared = declared or 0
    strength = max(coram_count, declared)
    if strength == 0:
        return {"bench_strength": None, "bench_source": ""}
    return {"bench_strength": strength, "bench_source": "coram" if coram_count >= declared else "declared"}


def single_judge_name(judge_column: Any) -> Optional[str]:
    """The parquet `judge` value if it holds exactly one name, else None."""
    if not isinstance(judge_column, str):
        return None
    names = [n.strip() for part in judge_column.split(",") for n in AND_SPLIT_RE.split(part) if n.strip()]
    return _norm(names[0]) if len(names) == 1 else None


def resolve_author(coram_author: Optional[str], judge_column: Any = None) -> Dict[str, Any]:
    """Author rule: the Coram name marked '*'; else the parquet `judge` column if it holds one name."""
    if coram_author:
        return {"author_judge": coram_author, "author_source": "coram_star"}
    single = single_judge_name(judge_column)
    if single:
        return {"author_judge": single, "author_source": "judge_column"}
    return {"author_judge": None, "author_source": "unknown"}


def parse_record(raw_html: Any, title: Optional[str] = None, judge_column: Any = None) -> Dict[str, Any]:
    """Parse one metadata row's raw_html (+ optional parquet `judge`) into selection/manifest fields."""
    text = strip_boilerplate(clean_html(raw_html), title)
    case_no = parse_case_no(raw_html)
    trailer = parse_trailer(text)
    coram = parse_coram(text)

    trailer_m = TRAILER_START_RE.search(text)
    body_end = trailer_m.start() if trailer_m else len(text)
    body_start = coram["end"] if coram["end"] is not None else body_end
    head = parse_headnote(text[body_start:body_end]) if body_start < body_end else parse_headnote("")

    coram_count = len(coram["judges"])
    declared = trailer["bench_declared"]
    if not coram_count:
        bench_check = "no_coram"
    elif declared is None:
        bench_check = "no_declared"
    else:
        bench_check = "match" if declared == coram_count else "mismatch"
    bench = resolve_bench(coram_count, declared)
    strength = bench["bench_strength"]

    scr = SCR_RE.search(text)
    neutral = NEUTRAL_RE.search(text)
    return {
        "case_no": case_no or trailer["case_no"],
        "case_type_raw": raw_case_type(case_no or trailer["case_no"]),
        "case_type": normalize_case_type(case_no or trailer["case_no"]),
        "is_criminal": is_criminal(case_no or trailer["case_no"]),
        "judges": coram["judges"],
        "author_judge_coram": coram["author_judge"],
        **resolve_author(coram["author_judge"], judge_column),
        "bench_coram_count": coram_count,
        "bench_strength": strength,
        "bench_source": bench["bench_source"],
        "bench_type": bench_type(strength),
        "bench_declared": declared,
        "bench_check": bench_check,
        "decision_date_html": trailer["decision_date"],
        "disposal_nature_html": trailer["disposal_nature"],
        "scr_citation": f"[{scr.group(1)}] {scr.group(2)} S.C.R. {scr.group(3)}" if scr else None,
        "neutral_citation": f"{neutral.group(1)} INSC {neutral.group(2)}" if neutral else None,
        **head,
    }
