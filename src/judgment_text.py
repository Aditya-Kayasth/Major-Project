"""Judgment PDF text: extraction, cleaning, page-1 parsing, headnote split, statutes, chunking.

Pure functions (stdlib only; pypdf is imported inside extract_pages). This is the ONLY project
module the spawned PDF-extraction workers import, so it must never import torch, pydantic,
sentence_transformers or other src modules.

Built from real PDFs (server_logs/peek_pdf_pages.log, 26 Nov). Three layouts:
- old_scr (2018-2022): margin letters A..H as separate lines on every page; running header
  "208 SUPREME COURT REPORTS [2019] 4 S.C.R."; bench "[A AND B, JJ.]"; unlabelled headnote
  (subject line ... "HELD: 1. ... [Para 10][214-F, G]"); body pages carry "TITLE v. TITLE" and
  "[NAME, J.]" running headers (the latter names the author); last page "<Reporter> Appeal allowed."
- escr_2023: "CASE DETAILS", "[J. B. PARDIWALA AND MANOJ MISRA, JJ.]", "HEADNOTES",
  "Issue for consideration :", "OTHER CASE DETAILS INCLUDING IMPUGNED ORDER AND APPEARANCES";
  body after "JUDGMENT / ORDER OF THE SUPREME COURT"; ends "Headnotes prepared by: Appeal allowed."
- escr_2024 (2024-25): "[Ahsanuddin Amanullah* and Prashant Kumar Mishra, JJ.]" + "* Author"
  footnote; "Issue for Consideration"; "Headnotes†"; running headers "198 [2025] 11 S.C.R." /
  "Supreme Court Reports"; ends "Result of the case: Appeal allowed." + "†Headnotes prepared by: ...".
"""

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# =========================================================================== normalisation
_LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"  # ff fi fl ffi ffl st st
# pypdf emits "oﬀ ence", "ﬁ led": a ligature followed by a space inside a word
_LIG_SPLIT_RE = re.compile(rf"([{_LIGATURES}])[ \t]+([a-z]+)")
_JOIN_STOP = {"a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
              "or", "the", "to", "was", "were", "with"}
_SPACES_RE = re.compile(r"[ \t ]+")


def _rejoin_ligature(m: "re.Match") -> str:
    return m.group(0) if m.group(2) in _JOIN_STOP else m.group(1) + m.group(2)


def normalize_text(text: str) -> str:
    """Rejoin ligature-split words, apply NFKC (ﬀ -> ff, ﬁ -> fi, nbsp -> space), collapse spaces per line."""
    if not text:
        return ""
    text = _LIG_SPLIT_RE.sub(_rejoin_ligature, text)
    text = unicodedata.normalize("NFKC", text)
    return "\n".join(_SPACES_RE.sub(" ", line).strip() for line in text.splitlines())


def extract_pages(pdf_path: str) -> List[str]:
    """One normalised text string per page (pypdf). A page that fails to extract gives ''."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(normalize_text(page.extract_text() or ""))
        except Exception:
            pages.append("")
    return pages


# =========================================================================== layout
def detect_layout(pages: List[str]) -> str:
    first = pages[0] if pages else ""
    lines = [l.strip() for l in first.splitlines()]
    margin = sum(1 for l in lines[:20] if re.fullmatch(r"[A-H]", l))
    if margin >= 6 or "SUPREME COURT REPORTS" in first:
        return "old_scr"
    head = "\n".join(pages[:3])
    if "CASE DETAILS" in first or re.search(r"JUDGMENT\s*/\s*ORDER OF THE SUPREME COURT", head):
        return "escr_2023"
    return "escr_2024"


# =========================================================================== cleaning
MARGIN_LETTER_RE = re.compile(r"^[A-H]$")
PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
SCR_HEADER_RES = [
    re.compile(r"^\d{1,4}\s+SUPREME COURT REPORTS\s+\[\d{4}\]\s+\d+\s+S\.C\.R\.?$"),
    re.compile(r"^\[\d{4}\]\s+\d+\s+S\.C\.R\.?(?:\s+\d{1,4})?(?:\s*:\s*\d{4}\s+INSC\s+\d+)?$"),
    re.compile(r"^\d{1,4}\s*\[\d{4}\]\s+\d+\s+S\.C\.R\.?$"),
    re.compile(r"^Supreme Court Reports$", re.IGNORECASE),
]
SIGNATURE_RE = re.compile(r"^(?:Signature Not Verified|Digitally signed by|Date:\s*\d{4}[.-]\d{2}[.-]\d{2}|Reason:)",
                          re.IGNORECASE)
AUTHOR_FOOTNOTE_RE = re.compile(r"^\*\s*Author$", re.IGNORECASE)
HEADNOTES_BY_RE = re.compile(r"^†?\s*Headnotes prepared by\s*:?\s*(.*)$", re.IGNORECASE)
RESULT_RE = re.compile(r"^Result of the case\s*:\s*(.+?)\.?$", re.IGNORECASE)
DISPOSITION_RE = re.compile(
    r"((?:Criminal\s+|Special Leave\s+|Writ\s+|Review\s+|Transfer\s+|Contempt\s+|Curative\s+)?"
    r"(?:Appeals?|Petitions?|Applications?|Cases?)\s+(?:(?:partly|partially)\s+)?"
    r"(?:allowed|dismissed|disposed of|disposed off))\.?$",
    re.IGNORECASE,
)
REPORTER_NAME_RE = re.compile(r"^[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]*){0,3}$")
REPORTABLE_RE = re.compile(r"^(NON[- ]?REPORTABLE|REPORTABLE)$", re.IGNORECASE)
RUNNING_AUTHOR_RE = re.compile(r"^\[([A-Z][A-Za-z .'-]{2,60}?),\s*J\.?\]$")


def _line_key(line: str) -> str:
    """Key for repeated-header detection: digits and spaces removed, lower case."""
    return re.sub(r"[\d\s]+", " ", line).strip().lower()


def _disposition(text: str) -> Optional[str]:
    m = DISPOSITION_RE.search(text.strip())
    return _SPACES_RE.sub(" ", m.group(1)).strip().capitalize() if m else None


def clean_pages(pages: List[str], repeat_fraction: float = 0.4) -> Dict[str, Any]:
    """Remove page furniture; return kept lines plus what was learnt on the way.

    Removed: margin letters, page numbers, S.C.R. running headers, lines repeated on >= 40% of
    pages (title running headers), signature blocks, '* Author' footnote, 'Headnotes prepared by'
    (+ the reporter's name), 'Result of the case', the reporter + result line, REPORTABLE markers.
    Learnt: running author ('[NAME, J.]'), disposition from the end text, reportable flag.
    """
    n = len(pages)
    page_lines = [[l.strip() for l in p.splitlines()] for p in pages]
    key_pages: Counter = Counter()
    for lines in page_lines:
        key_pages.update({_line_key(l) for l in lines if l})
    threshold = max(2, math.ceil(repeat_fraction * n)) if n >= 3 else None

    removed: Counter = Counter()
    running_authors: Counter = Counter()
    disposition: Optional[str] = None
    reportable: Optional[bool] = None
    kept: List[str] = []
    page_starts: List[int] = []

    for pi, lines in enumerate(page_lines):
        page_starts.append(len(kept))
        drop_reporter_name = False
        near_end = pi >= n - 2
        for line in lines:
            if not line:
                continue
            if drop_reporter_name:
                drop_reporter_name = False
                if REPORTER_NAME_RE.match(line) and not DISPOSITION_RE.search(line):
                    removed["reporter"] += 1
                    continue
            m = RUNNING_AUTHOR_RE.match(line)
            if m:
                running_authors[m.group(1).strip()] += 1
                removed["running_author"] += 1
                continue
            if MARGIN_LETTER_RE.match(line):
                removed["margin_letter"] += 1
                continue
            if PAGE_NUM_RE.match(line):
                removed["page_number"] += 1
                continue
            if any(r.match(line) for r in SCR_HEADER_RES):
                removed["scr_header"] += 1
                continue
            if SIGNATURE_RE.match(line):
                removed["signature"] += 1
                continue
            if AUTHOR_FOOTNOTE_RE.match(line):
                removed["author_footnote"] += 1
                continue
            m = RESULT_RE.match(line)
            if m:
                disposition = _disposition(m.group(1)) or m.group(1).strip()
                removed["result_line"] += 1
                continue
            m = HEADNOTES_BY_RE.match(line)
            if m:
                disposition = _disposition(m.group(1)) or disposition
                drop_reporter_name = not m.group(1).strip() or _disposition(m.group(1)) is not None
                removed["headnotes_prepared_by"] += 1
                continue
            if near_end:
                m = DISPOSITION_RE.search(line)
                if m:
                    prefix = line[: m.start()].strip()
                    if not prefix or (len(prefix.split()) <= 5 and REPORTER_NAME_RE.match(prefix)):
                        disposition = _disposition(line)
                        removed["reporter_result"] += 1
                        continue
            if REPORTABLE_RE.match(line):
                reportable = not line.upper().startswith("NON")
                removed["reportable_marker"] += 1
                continue
            if threshold is not None:
                key = _line_key(line)
                if len(key) >= 6 and key_pages[key] >= threshold:
                    removed["running_header"] += 1
                    continue
            kept.append(line)

    running_author = running_authors.most_common(1)[0][0] if running_authors else None
    return {
        "lines": kept,
        "page_starts": page_starts,
        "disposition_pdf": disposition,
        "reportable": reportable,
        "running_author": running_author,
        "removed": dict(removed),
    }


# =========================================================================== page 1: bench
_BENCH_LINE_END_RE = re.compile(r"JJ?\.?\s*\]\s*$")
_BENCH_RE = re.compile(r"\[([^\[\]]*?[A-Za-z][^\[\]]*?),?\s*(JJ|J)\.?\s*\]")
_BENCH_SPLIT_RE = re.compile(r"\s*,\s*|\s+and\s+|\s+AND\s+|\s*&\s*")


def parse_bench(page1_text: str) -> Dict[str, Any]:
    """Bench from page 1: '[A AND B, JJ.]', '[A* and B, JJ.]', '[A and B, JJ]', '[A, J.]'."""
    head = " ".join(l.strip() for l in page1_text.splitlines()[:40])
    for m in _BENCH_RE.finditer(head):
        content = m.group(1).strip()
        if re.match(r"^\d", content):  # "[2019] 4 S.C.R." is a citation, not a bench
            continue
        judges: List[str] = []
        author = None
        for part in _BENCH_SPLIT_RE.split(content):
            starred = "*" in part
            name = _SPACES_RE.sub(" ", part.replace("*", "")).strip().upper()
            if len(name) < 2:
                continue
            judges.append(name)
            if starred and author is None:
                author = name
        if judges:
            return {"judges": judges, "author": author}
    return {"judges": [], "author": None}


def name_key(name: Optional[str]) -> str:
    """'J. B. PARDIWALA' and 'J.B. Pardiwala' -> 'JBPARDIWALA' (for matching names across sources)."""
    return re.sub(r"[^A-Z]", "", (name or "").upper())


# =========================================================================== sections
HEADNOTE_LABEL_RE = re.compile(r"^(?:HEADNOTES?†?|Headnotes†?|Issue for [Cc]onsideration\b)")
CASE_DETAILS_LABEL_RE = re.compile(r"^CASE DETAILS$")
DETAILS_START_RE = re.compile(
    r"^(?:OTHER CASE DETAILS|Case Arising From|(?:CRIMINAL\s+|CIVIL\s+)?(?:APPELLATE|ORIGINAL|INHERENT)\s+JURISDICTION\b)",
    re.IGNORECASE,
)
BODY_MARKER_RE = re.compile(r"^Judgment\s*/\s*Order of the Supreme Court$", re.IGNORECASE)
DELIVERED_RE = re.compile(r"^The (Judgment|Order) of the Court was delivered by\s*:?\s*(.*)$", re.IGNORECASE)
JUDGMENT_HEADING_RE = re.compile(r"^(?:J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|O\s*R\s*D\s*E\s*R|JUDGMENT|ORDER|Judgment|Order)$")
ORDER_HEADING_RE = re.compile(r"^(?:O\s*R\s*D\s*E\s*R|ORDER|Order)$")
FIRST_NUMBERED_PARA_RE = re.compile(r"^1\.\s+\S")
JUDGE_SIGNATURE_RE = re.compile(r"^([A-Z][A-Za-z .'-]{2,60}?),\s*J\.?$")


def split_sections(lines: List[str]) -> Dict[str, Any]:
    """Split cleaned lines into front matter, headnote, case details/appearances and body.

    Returns lists of lines plus: headnote_split ('details' | 'body_marker' | 'numbered_para' |
    'none' | 'unsplit'), body_marker, is_order, delivered_by.
    """
    n = len(lines)
    front_end = 0
    for i in range(min(40, n)):
        if _BENCH_LINE_END_RE.search(lines[i]):
            front_end = i + 1
            break

    details_start = next((i for i in range(front_end, n) if DETAILS_START_RE.match(lines[i])), None)

    body_start = None
    body_marker = None
    delivered_by = None
    search_from = details_start if details_start is not None else front_end
    for i in range(search_from, n):
        line = lines[i]
        if BODY_MARKER_RE.match(line):
            body_start, body_marker = i + 1, "judgment_order_marker"
            break
        m = DELIVERED_RE.match(line)
        if m:
            body_start, body_marker = i + 1, "delivered_by"
            delivered_by = m.group(2).strip() or (lines[i + 1] if i + 1 < n else "")
            break
        if details_start is not None and i > details_start and JUDGMENT_HEADING_RE.match(line):
            body_start, body_marker = i, "heading"
            break

    has_label = any(HEADNOTE_LABEL_RE.match(l) for l in lines[front_end : front_end + 6])
    if details_start is not None:
        head_end, split = details_start, "details"
    elif body_start is not None:
        head_end, split = (body_start - 1 if body_marker != "heading" else body_start), "body_marker"
    elif has_label:
        nump = next((i for i in range(front_end, n) if FIRST_NUMBERED_PARA_RE.match(lines[i])), None)
        if nump is None:
            head_end, split = None, "unsplit"
        else:
            head_end, split, body_start = nump, "numbered_para", nump
    else:
        head_end, split = front_end, "none"

    if split == "unsplit":
        headnote, details, body = [], [], lines[front_end:]
    else:
        headnote = lines[front_end:head_end]
        details = lines[details_start:body_start] if (details_start is not None and body_start is not None) else []
        if body_start is None:
            body_start = details_start if details_start is not None else head_end
        body = lines[body_start:]

    headnote = [l for l in headnote if not CASE_DETAILS_LABEL_RE.match(l)]
    first_body = next((l for l in body[:5] if l), "")
    is_order = bool(ORDER_HEADING_RE.match(first_body))
    if body_marker == "delivered_by":
        dm = next((DELIVERED_RE.match(l) for l in lines if DELIVERED_RE.match(l)), None)
        is_order = is_order or bool(dm and dm.group(1).lower() == "order")
    return {
        "front": lines[:front_end],
        "headnote": headnote,
        "details": details,
        "body": body,
        "headnote_split": split,
        "body_marker": body_marker,
        "is_order": is_order,
        "delivered_by": delivered_by,
    }


def find_author(bench_author: Optional[str], running_author: Optional[str], delivered_by: Optional[str],
                body_lines: List[str], is_order: bool) -> Tuple[Optional[str], str]:
    """Author from the PDF: '*' in the bench line; else the old running header '[NAME, J.]'; else
    'delivered by NAME, J.'; else the first 'NAME, J.' line in the body; else 'Per Curiam' for an ORDER."""
    if bench_author:
        return bench_author.upper(), "pdf_star"
    if running_author:
        return running_author.upper(), "pdf_running_header"
    if delivered_by:
        name = re.sub(r",?\s*J\.?\s*$", "", delivered_by).strip()
        if name and not re.search(r"\bJJ\b", delivered_by):
            return name.upper(), "pdf_delivered_by"
    for line in body_lines[:25]:
        m = JUDGE_SIGNATURE_RE.match(line)
        if m:
            return m.group(1).strip().upper(), "pdf_body_signature"
    if is_order:
        return "Per Curiam", "pdf_order"
    return None, "unknown"


# =========================================================================== statutes
_ACTS: List[Tuple[str, str]] = [
    ("IPC", r"I\.?\s?P\.?\s?C\b\.?|(?i:Indian\s+Penal\s+Code|Penal\s+Code)"),
    ("CrPC", r"Cr\.?\s?P\.?\s?C\b\.?|(?i:Code\s+of\s+Criminal\s+Procedure)"),
    ("Evidence Act", r"(?i:(?:Indian\s+)?Evidence\s+Act)"),
    ("BNS", r"BNS(?!S)\b|(?i:Bharatiya\s+Nyaya\s+Sanhita)"),
    ("BNSS", r"BNSS\b|(?i:Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita)"),
    ("BSA", r"BSA\b|(?i:Bharatiya\s+Sakshya\s+Adhiniyam)"),
    ("NDPS", r"NDPS(?:\s+Act)?\b|(?i:Narcotic\s+Drugs\s+and\s+Psychotropic\s+Substances\s+Act)"),
    ("POCSO", r"POCSO(?:\s+Act)?\b|(?i:Protection\s+of\s+Children\s+from\s+Sexual\s+Offences\s+Act)"),
    ("Arms Act", r"(?i:Arms\s+Act)"),
    ("SC/ST (PoA)", r"SC\s*/\s*ST(?:\s*\(PoA\))?(?:\s+Act)?|(?i:Scheduled\s+Castes\s+and\s+(?:the\s+)?Scheduled\s+Tribes\s+\(Prevention\s+of\s+Atrocities\)\s+Act)"),
    ("PC Act", r"(?i:Prevention\s+of\s+Corruption\s+Act)|P\.?\s?C\.?\s+Act"),
]
_ACT_ALT = "|".join(f"(?P<a{i}>{pat})" for i, (_, pat) in enumerate(_ACTS))
_ACT_RE = rf"(?<![A-Za-z])(?:{_ACT_ALT})(?![A-Za-z])"
_NUM = r"\d{1,3}[A-Z]{0,2}(?:\s*\(\s*\w{1,4}\s*\))*"
_SEC_WORD = r"(?<![A-Za-z])(?i:sections?|secs?\.?|ss?\.|u/ss?\.?|u/s\.?)"
_SEP = rf"\s*(?:,|and|&|or|/|r/w|read\s+with|to)\s*(?:(?i:ss?|secs?|sections?)\.?\s*)?"
_NUMLIST = rf"{_NUM}(?:{_SEP}{_NUM})*"
_STAT_RES = [
    re.compile(rf"{_SEC_WORD}\s*(?P<nums>{_NUMLIST})\s*,?\s*(?:(?i:of)\s+(?:(?i:the)\s+)?)?{_ACT_RE}"),
    re.compile(rf"{_ACT_RE}(?:\s*,\s*\d{{4}})?\s*[:–—-]\s*{_SEC_WORD}\s*(?P<nums>{_NUMLIST})"),
]


def statutes_cited(text: str) -> str:
    """Doc-level statute sections: 'IPC 34; IPC 302; CrPC 319; Arms Act 25' (act order, then number)."""
    found = set()
    for rx in _STAT_RES:
        for m in rx.finditer(text or ""):
            act_idx = next(int(k[1:]) for k, v in m.groupdict().items() if k.startswith("a") and v)
            nums = re.sub(r"\(\s*\w{1,4}\s*\)", "", m.group("nums"))
            for num in re.findall(r"\d{1,3}[A-Z]{0,2}", nums):
                found.add((act_idx, int(re.match(r"\d+", num).group()), num))
    return "; ".join(f"{_ACTS[a][0]} {num}" for a, _, num in sorted(found))


# =========================================================================== disposition
def disposition_class(text: Optional[str]) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    if "partly" in t or "partially" in t:
        return "partly"
    if "allowed" in t:
        return "allowed"
    if "dismissed" in t:
        return "dismissed"
    if "disposed" in t:
        return "disposed"
    return None


# =========================================================================== roles + chunking
ROLES = ("Headnote", "Facts", "Issues", "Submissions", "Precedents Cited", "Ratio/Analysis", "Order", "Unclassified")
_HEADING_ROLES = [
    (("SUBMISSION", "CONTENTION", "ARGUMENT"), "Submissions"),
    (("ISSUE", "QUESTION", "POINT FOR"), "Issues"),
    (("PRECEDENT", "CASE LAW", "AUTHORIT"), "Precedents Cited"),
    (("FACT", "BACKGROUND"), "Facts"),
    (("ANALYSIS", "DISCUSSION", "CONSIDERATION", "FINDING", "RATIO", "HELD", "REASON"), "Ratio/Analysis"),
    (("CONCLUSION", "RESULT", "DIRECTION", "DISPOSAL", "RELIEF", "OPERATIVE"), "Order"),
]
_CUE_ROLES = [
    ("Submissions", ("learned counsel", "learned senior counsel", "contended that", "submitted that",
                     "argued that", "it was urged", "per contra", "it is the case of the appellant")),
    ("Issues", ("question that arises", "questions that arise", "issue that arises", "the issue for",
                "point for determination", "question for consideration", "falls for consideration",
                "the questions for", "issues for consideration")),
    ("Ratio/Analysis", ("we hold", "held that", "in our view", "in our considered", "we are of the opinion",
                        "we are of the considered", "it is settled", "it is well settled", "we find that",
                        "we are unable to", "we are not persuaded", "ratio decidendi")),
    ("Order", ("appeal is allowed", "appeal is dismissed", "appeals are allowed", "appeals are dismissed",
               "stands disposed", "disposed of", "we direct", "is hereby allowed", "pending application",
               "is set aside", "are set aside")),
    ("Facts", ("prosecution case", "factual matrix", "fir no", "fir was", "brief facts", "the facts",
               "it is alleged", "complaint was", "the deceased", "charge sheet", "chargesheet", "trial court convicted")),
    ("Precedents Cited", ("in the case of", "this court in", "relied upon", "relied on", " scc ", "(supra)",
                          "air 19", "air 20", " scr ")),
]

PARA_START_RE = re.compile(r"^(?:\d{1,3}\.(?:\d{1,2}\.?)?|\(\d{1,3}\)|\([a-z]{1,3}\)|\([ivx]{1,6}\)|[IVX]{1,6}\.)\s+\S")
HEADING_LINE_RE = re.compile(r"^[A-Z][A-Z0-9 ,.&'()/:\-]{2,59}$")
MD_HEADING_RE = re.compile(r"^#+\s+(.+)$")
ANCHOR_RE = re.compile(
    r"(?:\d{1,3}\.\s+)?(?:Learned\s+(?:Senior\s+)?(?:Counsel|counsel)|Having\s+heard|Having\s+considered|"
    r"We\s+are\s+of\s+the\s+(?:considered\s+)?opinion|In\s+our\s+(?:considered\s+)?view|"
    r"The\s+question\s+(?:that\s+falls|for\s+determination|that\s+arises)|We\s+hold|It\s+is\s+held|"
    r"We\s+accordingly\s+hold|For\s+the\s+reasons|In\s+the\s+result|The\s+appeals?\s+(?:is|are)\s+(?:allowed|dismissed)|"
    r"Per\s+contra|Heard\s+learned)",
    re.IGNORECASE,
)


def detect_role(text: str, heading: Optional[str]) -> str:
    if heading:
        h = heading.upper()
        for keys, role in _HEADING_ROLES:
            if any(k in h for k in keys):
                return role
    sample = " " + text[:400].lower() + " "
    for role, cues in _CUE_ROLES:
        if any(c in sample for c in cues):
            return role
    return "Unclassified"


def _is_heading(line: str) -> Optional[str]:
    m = MD_HEADING_RE.match(line)
    if m:
        return m.group(1).strip()
    if HEADING_LINE_RE.match(line) and sum(c.isalpha() for c in line) >= 4 and not PARA_START_RE.match(line):
        return line.rstrip(":").strip()
    return None


def to_paragraphs(lines: List[str]) -> List[Tuple[Optional[str], str]]:
    """Rebuild paragraphs from PDF lines: a new paragraph at numbered paragraph starts and headings."""
    paras: List[Tuple[Optional[str], str]] = []
    heading: Optional[str] = None
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            paras.append((heading, cur.strip()))
        cur = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if JUDGMENT_HEADING_RE.match(line):  # "JUDGMENT" / "ORDER" names the document type, not a section
            flush()
            heading = None
            continue
        h = _is_heading(line)
        if h is not None:
            flush()
            heading = h
            continue
        if PARA_START_RE.match(line):
            flush()
            cur = line
            continue
        cur = (cur[:-1] + line) if cur.endswith("-") else (f"{cur} {line}" if cur else line)
    flush()
    return paras


def _split_long(text: str, size: int) -> List[str]:
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            cut = cut + 1 if cut > 0 else text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        out.append(text[start:end].strip())
        start = end
    return [t for t in out if t]


def build_parents(paragraphs: List[Tuple[Optional[str], str]], parent_size: int) -> List[Tuple[Optional[str], str]]:
    """Group paragraphs into parents of <= parent_size chars, flushing at rhetorical markers."""
    parents: List[Tuple[Optional[str], str]] = []
    cur: List[str] = []
    cur_len = 0
    cur_heading: Optional[str] = None
    for heading, para in paragraphs:
        pieces = _split_long(para, parent_size) if len(para) > parent_size else [para]
        for piece in pieces:
            anchor = bool(ANCHOR_RE.match(piece))
            if cur and (anchor or cur_len + len(piece) > parent_size or heading != cur_heading):
                parents.append((cur_heading, "\n\n".join(cur)))
                cur, cur_len = [], 0
            if not cur:
                cur_heading = heading
            cur.append(piece)
            cur_len += len(piece) + 2
    if cur:
        parents.append((cur_heading, "\n\n".join(cur)))
    return parents


def child_windows(text: str, size: int, overlap: int) -> List[str]:
    """Sliding windows of ~size chars with overlap, cut at spaces."""
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            sp = text.rfind(" ", start + size - 120, end)
            end = sp if sp > start else end
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        sp = text.find(" ", nxt, end)
        start = sp + 1 if sp != -1 else nxt
    return out


# Section 7 document-level fields copied onto every child chunk
DOC_FIELDS = ("doc_id", "case_title", "petitioner", "respondent", "court", "decision_date", "year", "judges",
              "bench_strength", "bench_type", "bench_source", "author_judge", "author_source", "citation", "case_id",
              "cnr", "disposal_nature", "case_type", "case_no", "reportable", "statutes_cited", "source_path")


def chunk_sections(sections: List[Tuple[str, List[str]]], doc: Dict[str, Any], parent_size: int, child_size: int,
                   overlap: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parents and children for [('headnote', lines), ('body', lines)]. IDs: <doc_id>-p<NNNNN>-c<NNN>."""
    doc_id = doc["doc_id"]
    title = doc.get("case_title") or ""
    base = {k: doc.get(k) for k in DOC_FIELDS}
    parents: List[Dict[str, Any]] = []
    children: List[Dict[str, Any]] = []
    p_idx = 0
    for kind, lines in sections:
        for heading, p_text in build_parents(to_paragraphs(lines), parent_size):
            parent_id = f"{doc_id}-p{p_idx:05d}"
            p_role = "Headnote" if kind == "headnote" else detect_role(p_text, heading)
            parents.append({
                "parent_id": parent_id,
                "doc_id": doc_id,
                "case_title": title,
                "text": p_text,
                "metadata": {**base, "parent_id": parent_id, "parent_index": p_idx, "role": p_role,
                             "section_kind": kind, "structural_heading": heading or "General",
                             "char_length": len(p_text)},
            })
            for c_idx, c_text in enumerate(child_windows(p_text, child_size, overlap)):
                role = "Headnote" if kind == "headnote" else detect_role(c_text, heading)
                vec_text = f"[{role} | {title}]: {c_text}"
                children.append({
                    **base,
                    "parent_id": parent_id,
                    "chunk_id": f"{parent_id}-c{c_idx:03d}",
                    "role": role,
                    "section_kind": kind,
                    "parent_index": p_idx,
                    "text": vec_text,
                    "raw_text": c_text,
                    "char_length": len(vec_text),
                    "raw_char_length": len(c_text),
                })
            p_idx += 1
    return parents, children


# =========================================================================== whole document
def _none(v: Any) -> Any:
    """None for NaN / empty strings (manifest rows come from pandas)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v


def bench_type_label(strength: Optional[int]) -> str:
    """CLAUDE.md D6 labels (duplicated from src.sc_metadata so workers import nothing else)."""
    if not strength or strength < 1:
        return ""
    return "Constitution Bench" if strength >= 5 else "Full (3+)" if strength >= 3 else "Division (2)" if strength == 2 else "Single"


def build_document(pdf_path: str, row: Dict[str, Any], parent_size: int = 2000, child_size: int = 800,
                   overlap: int = 120, pages: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract, clean, split and chunk one judgment. Runs in a spawned worker; returns plain dicts.

    Manifest metadata wins; the PDF only fills a missing bench (bench_source='pdf') and an unknown
    author (author_source='pdf_*'). Any exception is returned as {'error': ...}, never raised.
    """
    doc_id = str(row.get("doc_id") or hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:16])
    try:
        if pages is None:
            pages = extract_pages(pdf_path)
        n_chars_raw = sum(len(p) for p in pages)
        if n_chars_raw < 200:
            return {"doc_id": doc_id, "error": f"insufficient text ({n_chars_raw} chars; scanned PDF?)"}

        layout = detect_layout(pages)
        bench_pdf = parse_bench(pages[0])
        cleaned = clean_pages(pages)
        sec = split_sections(cleaned["lines"])

        judges = _none(row.get("judges"))
        strength = _none(row.get("bench_strength"))
        bench_source = _none(row.get("bench_source")) or ""
        if not strength and bench_pdf["judges"]:
            judges = "; ".join(bench_pdf["judges"])
            strength = len(bench_pdf["judges"])
            bench_source = "pdf"
        strength = int(strength) if strength else None

        pdf_author, pdf_author_source = find_author(bench_pdf["author"], cleaned["running_author"],
                                                    sec["delivered_by"], sec["body"], sec["is_order"])
        author = _none(row.get("author_judge"))
        author_source = _none(row.get("author_source")) or "unknown"
        if not author or author_source == "unknown":
            author, author_source = pdf_author, pdf_author_source
        author_agrees = None
        if _none(row.get("author_judge")) and pdf_author and pdf_author != "Per Curiam":
            author_agrees = name_key(row.get("author_judge")) == name_key(pdf_author)

        full_text = "\n".join(cleaned["lines"])
        disposal = _none(row.get("disposal_nature"))
        d_pdf, d_meta = disposition_class(cleaned["disposition_pdf"]), disposition_class(disposal)
        doc = {
            "doc_id": doc_id,
            "case_title": _none(row.get("title")) or _none(row.get("case_title")) or "",
            "petitioner": _none(row.get("petitioner")),
            "respondent": _none(row.get("respondent")),
            "court": "Supreme Court of India",
            "decision_date": _none(row.get("decision_date")),
            "year": int(row["year"]) if _none(row.get("year")) is not None else None,
            "judges": judges,
            "bench_strength": strength,
            "bench_type": bench_type_label(strength),
            "bench_source": bench_source,
            "author_judge": author,
            "author_source": author_source,
            "citation": _none(row.get("citation")),
            "case_id": _none(row.get("case_id")),
            "cnr": _none(row.get("cnr")),
            "disposal_nature": disposal,
            "case_type": _none(row.get("case_type")),
            "case_no": _none(row.get("case_no")),
            "reportable": cleaned["reportable"],
            "statutes_cited": statutes_cited(full_text),
            "source_path": str(pdf_path),
        }
        sections = [("headnote", sec["headnote"]), ("body", sec["body"])]
        parents, children = chunk_sections(sections, doc, parent_size, child_size, overlap)
        if not children:
            return {"doc_id": doc_id, "error": "zero chunks after cleaning"}

        doc_row = {
            **doc,
            "layout": layout,
            "headnote": "\n".join(sec["headnote"]),
            "headnote_split": sec["headnote_split"],
            "body_marker": sec["body_marker"] or "",
            "disposition_pdf": cleaned["disposition_pdf"],
            "disposition_agrees": (d_pdf == d_meta) if (d_pdf and d_meta) else None,
            "pdf_judges": "; ".join(bench_pdf["judges"]),
            "pdf_author": pdf_author,
            "pdf_author_source": pdf_author_source,
            "author_pdf_agrees": author_agrees,
            "n_pages": len(pages),
            "n_chars": len(full_text),
            "n_parents": len(parents),
            "n_chunks": len(children),
        }
        stats = {
            "layout": layout,
            "headnote_split": sec["headnote_split"],
            "has_headnote": bool(sec["headnote"]),
            "roles": dict(Counter(c["role"] for c in children)),
            "section_kind": dict(Counter(c["section_kind"] for c in children)),
            "removed": cleaned["removed"],
            "author_source": author_source,
            "bench_source": bench_source,
            "author_pdf_agrees": author_agrees,
            "disposition_agrees": doc_row["disposition_agrees"],
            "disposition_found": cleaned["disposition_pdf"] is not None,
            "reportable": cleaned["reportable"],
            "n_statutes": len([s for s in doc["statutes_cited"].split("; ") if s]),
            "statutes": doc["statutes_cited"],
        }
        return {"doc_id": doc_id, "doc": doc_row, "parents": parents, "children": children, "stats": stats}
    except Exception as err:  # never kill the pool
        return {"doc_id": doc_id, "error": f"{type(err).__name__}: {err}"[:300]}
