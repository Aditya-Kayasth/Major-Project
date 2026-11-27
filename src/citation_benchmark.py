"""Benchmark v3 building blocks (pure): citations in an unseen judgment -> precedents in our corpus.

Query   = the court-written "Issue for Consideration" of a 2026 judgment (not in the index), scrubbed
          of case names, citations, case numbers, dates and its own party/judge names.
Targets = earlier judgments that the 2026 judgment itself cites and that ARE in our corpus, matched by
          - SCR citation  "[2019] 4 SCR 208" / "[2019] 4 S.C.R. 208"   (year + volume + start page)
          - neutral        "2019 INSC 386"
          - optional, conservative case-name match (both party keys + year 2018-2025, unique hit) = 'name'
SCC / AIR citations cannot be mapped to our corpus (no SCC/AIR metadata): counted, never used.
"""

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src import benchmark_v2 as bv

SCR_RE = re.compile(r"\[(\d{4})\]\s*(\d{1,2})\s*S\.?\s?C\.?\s?R\.?\s*(\d{1,4})")
INSC_RE = re.compile(r"\b((?:19|20)\d{2})\s*INSC\s*(\d{1,5})\b")
SCC_RE = re.compile(r"\(\d{4}\)\s*\d{1,2}\s*SCC\s*\d{1,4}")
AIR_RE = re.compile(r"\bAIR\s*\d{4}\s*SC\s*\d{1,5}\b")
CASE_NAME_RE = re.compile(
    r"([A-Z][A-Za-z.&'@\-]*(?:\s+[A-Za-z.&'@\-()]+){0,10}?)\s+(?:v\.?|vs\.?|versus)\s+"
    r"([A-Z][A-Za-z.&'@\-]*(?:\s+[A-Za-z.&'@\-()]+){0,10}?)(?=\s*[,;:\[]|\s*\(\d|\s*$|\s+\d|\s+(?:and|where|in|it|the|this|was|is)\b)")
# Suffixes that are not part of a party's identity
_PARTY_NOISE = [
    re.compile(r"\(\s*(?:dead|d)\s*\)", re.IGNORECASE),
    re.compile(r"\b(?:thr(?:ough)?\.?)\s+(?:(?:his|her|their)\s+)?(?:l\.?\s?r\.?s?\.?|legal\s+(?:heirs?|representatives?))", re.IGNORECASE),
    re.compile(r"\s*(?:&|and)\s+(?:ors|anr|others|another|connected\s+matters?)\b\.?", re.IGNORECASE),
    re.compile(r"\betc\.?", re.IGNORECASE),
]
_LEAD_WORDS = {"in", "see", "also", "similarly", "held", "cf", "vide", "and", "as", "of", "the", "judgment", "decision",
               "case", "reliance", "relied", "placed", "on", "upon", "per", "while", "where", "this", "court"}
ISSUE_RE = re.compile(r"Issue\s+for\s+[Cc]onsideration\s*:?\s*(.*?)(?=\n\s*(?:Headnotes?|HEADNOTES?)\b|\Z)", re.DOTALL)
CORPUS_YEARS = range(2018, 2026)


def scr_key(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    m = SCR_RE.search(text or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def insc_key(text: Optional[str]) -> Optional[str]:
    m = INSC_RE.search(text or "")
    return f"{int(m.group(1))} INSC {int(m.group(2))}" if m else None


def normalize_party(name: Optional[str], cited: bool = False) -> str:
    """'Mehboob-ur-Rehman (Dead) through LRs & Ors.' -> 'Mehboob-ur-Rehman'. For a cited first party,
    leading sentence words are dropped ('Reliance was placed on Sugreev Kumar' -> 'Sugreev Kumar')."""
    s = name or ""
    for rx in _PARTY_NOISE:
        s = rx.sub(" ", s)
    toks = s.split()
    if cited:
        last_lower = max((i for i, t in enumerate(toks) if t[:1].islower() and t.lower() not in ("ur", "bin", "binti")),
                         default=-1)
        toks = toks[last_lower + 1:] if last_lower < len(toks) - 1 else toks
        while toks and toks[0].lower().strip(".,") in _LEAD_WORDS:
            toks = toks[1:]
    return " ".join(toks).strip(" ,.&")


def _party_key(name: Optional[str], cited: bool = False) -> frozenset:
    return frozenset(bv.party_tokens(normalize_party(name, cited)))


def build_corpus_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Manifest rows (indexed judgments) -> lookup tables by SCR key, neutral citation and party keys."""
    index: Dict[str, Any] = {"scr": {}, "insc": {}, "names": []}
    for r in rows:
        doc = r.get("doc_id")
        for field in ("scr_citation", "citation"):
            k = scr_key(r.get(field))
            if k:
                index["scr"].setdefault(k, doc)
        for field in ("neutral_citation", "case_id", "citation"):
            k = insc_key(r.get(field))
            if k:
                index["insc"].setdefault(k, doc)
        try:
            year = int(r.get("year"))
        except (TypeError, ValueError):
            year = None
        index["names"].append((_party_key(r.get("petitioner")), _party_key(r.get("respondent")), year, doc))
    return index


def extract_citations(text: str) -> Dict[str, Any]:
    """All citations in a judgment's text, by type (SCC / AIR only counted)."""
    names = []
    for m in CASE_NAME_RE.finditer(text):
        tail = text[m.end(): m.end() + 40]
        y = re.search(r"\b((?:19|20)\d{2})\b", tail)
        names.append((m.group(1), m.group(2), int(y.group(1)) if y else None))
    return {
        "scr": list(dict.fromkeys((int(a), int(b), int(c)) for a, b, c in SCR_RE.findall(text))),
        "insc": list(dict.fromkeys(f"{int(a)} INSC {int(b)}" for a, b in INSC_RE.findall(text))),
        "scc": len(SCC_RE.findall(text)),
        "air": len(AIR_RE.findall(text)),
        "names": names,
    }


def name_match(a: str, b: str, year: Optional[int], index: Dict[str, Any]) -> Optional[str]:
    """Conservative: both party keys contained in the corpus parties, same year (2018-2025), exactly one hit."""
    if year not in CORPUS_YEARS:
        return None
    ka, kb = _party_key(a, cited=True), _party_key(b)
    if not ka:
        return None
    hits = {doc for pk, rk, y, doc in index["names"]
            if y == year and pk and ka <= pk and ((kb and rk and kb <= rk) or (not kb and not rk))}
    return next(iter(hits)) if len(hits) == 1 else None


def unmatched_names(cites: Dict[str, Any], index: Dict[str, Any]) -> List[str]:
    """Cited case names with a corpus year (2018-2025) that did NOT match (for the build log)."""
    return [f"{normalize_party(a, cited=True)} v. {normalize_party(b)} ({y})" for a, b, y in cites["names"]
            if y in CORPUS_YEARS and name_match(a, b, y, index) is None]


def map_targets(cites: Dict[str, Any], index: Dict[str, Any], use_names: bool = True,
                exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Cited judgments that are in the corpus; citation matches win over name matches; deduped by doc_id."""
    found: Dict[str, Dict[str, Any]] = {}
    for k in cites["scr"]:
        doc = index["scr"].get(k)
        if doc and doc not in exclude:
            found.setdefault(doc, {"doc_id": doc, "match_type": "scr", "citation": f"[{k[0]}] {k[1]} SCR {k[2]}"})
    for k in cites["insc"]:
        doc = index["insc"].get(k)
        if doc and doc not in exclude:
            found.setdefault(doc, {"doc_id": doc, "match_type": "neutral", "citation": k})
    if use_names:
        for a, b, y in cites["names"]:
            doc = name_match(a, b, y, index)
            if doc and doc not in exclude and doc not in found:
                found[doc] = {"doc_id": doc, "match_type": "name", "citation": f"{a} v. {b} ({y})"}
    return list(found.values())


def issue_text(headnote: str, fallback_text: str = "") -> Tuple[str, str]:
    """('issue' | 'headnote_para' | 'none', text): the Issue for Consideration, else the first headnote paragraph."""
    for source in (headnote, fallback_text[:6000]):
        m = ISSUE_RE.search(source or "")
        if m and len(m.group(1).split()) >= 5:
            return "issue", " ".join(m.group(1).split())
    for para in re.split(r"\n\s*\n|\n(?=[A-Z][a-z]+ (?:Act|Code|Sanhita|Adhiniyam)\b)", headnote or ""):
        words = para.split()
        if len(words) >= 8 and not re.match(r"^(HEADNOTES?|CASE DETAILS)\b", para.strip()):
            return "headnote_para", " ".join(words)
    return "none", ""


def scrub(text: str, parties: Sequence[str] = (), judges: Sequence[str] = ()) -> Tuple[str, Counter]:
    """Remove case names, citations, case numbers, dates, non-Act years and the case's own party/judge names."""
    removed: Counter = Counter()

    def sub(rx, repl, s, label):
        new, n = rx.subn(repl, s)
        removed[label] += n
        return new

    s = text
    for rx, label in ((SCR_RE, "citation"), (INSC_RE, "citation"), (SCC_RE, "citation"), (AIR_RE, "citation"),
                      (bv.LEAK_PATTERNS["citation"], "citation")):
        s = sub(rx, "", s, label)
    s = sub(CASE_NAME_RE, "a precedent", s, "case_name")
    s = sub(bv.LEAK_PATTERNS["case_number"], "", s, "case_number")
    s = sub(bv.LEAK_PATTERNS["date"], "", s, "date")
    kept = []

    def drop_year(m: "re.Match") -> str:
        if bv.is_act_year(m.string, m):
            kept.append(m.group(0))
            return m.group(0)
        removed["year"] += 1
        return ""

    s = bv.YEAR_RE.sub(drop_year, s)
    for tok in list(parties) + list(judges):
        s = sub(re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE), "the party", s, "own_name")
    s = re.sub(r"\(\s*[,;]?\s*\)|\[\s*\]", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", s)).strip(" ,;:")
    return s, removed
