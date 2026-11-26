"""Citation check for generated answers (no LLM).

The model may cite only [Precedent N] for the N precedents it was given. Anything else counts as
an unknown citation:
- a [Precedent N] with N outside 1..len(precedents)
- a case citation written out in the answer (neutral 'YYYY INSC N', SCR, SCC, AIR, Cri LJ) that is
  not the citation of one of the given precedents.

citation_precision = valid citations / all citations found (None when the answer cites nothing).
"""

import re
from typing import Any, Dict, List, Optional

PRECEDENT_TAG_RE = re.compile(r"\[\s*Precedents?\s+(\d+(?:\s*(?:,|and|&|-)\s*\d+)*)\s*\]", re.IGNORECASE)
EXTERNAL_CITATION_RES = [
    re.compile(r"\b\d{4}\s+INSC\s+\d+\b"),
    re.compile(r"\[\d{4}\]\s+\d+\s+S\.?\s?C\.?\s?R\.?\s+\d+"),
    re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+"),
    re.compile(r"\bAIR\s+\d{4}\s+SC\s+\d+\b"),
    re.compile(r"\b\d{4}\s+Cri\.?\s?L\.?\s?J\.?\s+\d+\b", re.IGNORECASE),
]
REFUSAL_PHRASE = "insufficient precedent in the indexed corpus"


def _norm(c: str) -> str:
    return re.sub(r"[\s.]+", "", (c or "").upper())


def _numbers(tag_body: str) -> List[int]:
    nums: List[int] = []
    for part in re.split(r"\s*(?:,|and|&)\s*", tag_body):
        if "-" in part:
            a, _, b = part.partition("-")
            if a.strip().isdigit() and b.strip().isdigit() and int(b) - int(a) < 50:
                nums.extend(range(int(a), int(b) + 1))
        elif part.strip().isdigit():
            nums.append(int(part))
    return nums


def check_citations(answer: str, precedents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """precedents: the list given to the model, in order; each may carry 'citation' and 'scr_citation'."""
    text = answer or ""
    n = len(precedents)
    cited = []
    for m in PRECEDENT_TAG_RE.finditer(text):
        cited.extend(_numbers(m.group(1)))
    distinct = sorted(set(cited))
    valid = [i for i in distinct if 1 <= i <= n]
    unknown_numbers = [i for i in distinct if not 1 <= i <= n]

    known = {_norm(p.get(k) or "") for p in precedents for k in ("citation", "scr_citation", "case_id") if p.get(k)}
    external = []
    for rx in EXTERNAL_CITATION_RES:
        external.extend(m.group(0) for m in rx.finditer(text))
    external = list(dict.fromkeys(external))
    unknown_external = [c for c in external if _norm(c) not in known]
    known_external = [c for c in external if _norm(c) in known]

    total = len(distinct) + len(external)
    precision: Optional[float] = round((len(valid) + len(known_external)) / total, 4) if total else None
    return {
        "n_precedents": n,
        "cited_precedents": distinct,
        "valid_precedents": valid,
        "unknown_precedent_numbers": unknown_numbers,
        "external_citations": external,
        "unknown_external_citations": unknown_external,
        "citation_precision": precision,
        "uncited_precedents": [i for i in range(1, n + 1) if i not in valid],
        "refused": REFUSAL_PHRASE in text.lower(),
        "ends_with_verify": text.strip().rstrip(".").lower().endswith("verify on the official source"),
    }
