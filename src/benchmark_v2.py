"""Benchmark v2 building blocks (pure, no I/O): year allocation, leakage filter, out-of-scope set,
human-query template, judgment-level metrics and bootstrap confidence intervals.

Used by scripts/generate_benchmark_v2.py and scripts/evaluate_v2.py.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------- query generation
HOLDING_CUES = ("we hold", "held that", "in our view", "in our considered view", "we are of the", "the question",
                "accordingly", "set aside", "it is well settled", "the principle", "we are unable", "we find")
FACT_CUES = ("prosecution case", "fir", "the deceased", "the accused", "complainant", "charge sheet", "chargesheet",
             "the appellant was", "it is alleged", "trial court")

TITLE_STOPWORDS = {
    "the", "of", "and", "vs", "versus", "state", "union", "india", "ors", "anr", "others", "another", "ltd", "pvt",
    "limited", "through", "its", "secretary", "government", "on", "etc", "represented", "by", "station", "house",
    "officer", "director", "central", "bureau", "investigation", "republic", "national", "agency", "public",
    "prosecutor", "pradesh", "uttar", "madhya", "andhra", "himachal", "arunachal", "west", "bengal", "tamil", "nadu",
    "punjab", "haryana", "kerala", "karnataka", "maharashtra", "gujarat", "rajasthan", "bihar", "odisha", "orissa",
    "assam", "delhi", "telangana", "jharkhand", "chhattisgarh", "uttarakhand", "goa", "manipur", "tripura",
    "meghalaya", "mizoram", "nagaland", "sikkim", "jammu", "kashmir", "ladakh", "nct", "territory", "deceased",
    "minor", "legal", "heirs", "thr", "lrs", "father", "mother", "son", "wife", "husband", "accused", "appellant",
    "respondent", "petitioner", "complainant", "court", "high", "supreme", "officer", "police", "inspector",
}
# Act years a query may legitimately mention ("Indian Penal Code, 1860")
ACT_YEARS = {"1860", "1872", "1898", "1908", "1950", "1955", "1959", "1961", "1973", "1985", "1988", "1989",
             "2000", "2002", "2005", "2012", "2013", "2015", "2018", "2023"}
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
LEAK_PATTERNS = {
    # "No. 509", "509/2019", "of 2019" (but "302/34 IPC" is a section pair, not a case number)
    "case_number": re.compile(r"\b(?:no|nos)\.?\s*\d+|\b\d+\s*/\s*(?:19|20)\d{2}\b|\bof\s+(?:19|20)\d{2}\b", re.IGNORECASE),
    "citation": re.compile(r"\bINSC\b|\bSCC\b|S\.\s?C\.\s?R\.|\bSCR\b|\bAIR\s+\d{4}|\bCri\.?\s?L\.?\s?J", re.IGNORECASE),
    "date": re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\b|\b{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}|\b\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}\b",
                       re.IGNORECASE),
}

OUT_OF_SCOPE_QUERIES = [
    "What is the limitation period for filing a suit for specific performance of an agreement to sell immovable property?",
    "Can a landlord evict a tenant for non-payment of rent under the Maharashtra Rent Control Act?",
    "How is long-term capital gains tax computed on the sale of inherited agricultural land?",
    "What are the conditions for a decree of divorce by mutual consent under the Hindu Marriage Act?",
    "Under United States federal law, when does qualified immunity protect a police officer from a civil rights suit?",
]

HUMAN_TEMPLATE_SLOTS = 20


def human_template() -> List[Dict[str, Any]]:
    return [{"query_id": f"HQ-{i:02d}", "type": "legal_issue", "query": "", "target_doc_ids": [],
             "target_citations": [], "author": "", "notes": "type: legal_issue | fact_pattern; give the neutral "
             "citation (e.g. 2019 INSC 386) or doc_id of every judgment that answers the query"}
            for i in range(1, HUMAN_TEMPLATE_SLOTS + 1)]


def allocate_by_year(counts: Dict[int, int], total: int) -> Dict[int, int]:
    """Proportional allocation of `total` queries to years (largest remainder), capped by availability."""
    n = sum(counts.values())
    if n == 0 or total <= 0:
        return {y: 0 for y in counts}
    quotas = {y: total * c / n for y, c in counts.items()}
    alloc = {y: min(counts[y], int(math.floor(q))) for y, q in quotas.items()}
    rest = sorted(counts, key=lambda y: (quotas[y] - math.floor(quotas[y]), counts[y]), reverse=True)
    i = 0
    while sum(alloc.values()) < min(total, n) and i < 10 * len(rest):
        y = rest[i % len(rest)]
        if alloc[y] < counts[y]:
            alloc[y] += 1
        i += 1
    return alloc


def party_tokens(*names: Optional[str]) -> List[str]:
    """Distinctive words of party names / title (>= 4 letters, not generic)."""
    toks = set()
    for name in names:
        for w in re.findall(r"[A-Za-z]{4,}", name or ""):
            if w.lower() not in TITLE_STOPWORDS:
                toks.add(w.lower())
    return sorted(toks)


def judge_tokens(judges: Optional[str]) -> List[str]:
    return sorted({w.lower() for w in re.findall(r"[A-Za-z]{4,}", judges or "") if w.lower() not in TITLE_STOPWORDS})


def leakage_reasons(query: str, parties: Sequence[str] = (), judges: Sequence[str] = ()) -> List[str]:
    """Why a generated query would leak the target judgment (empty list = clean)."""
    q = query or ""
    low = q.lower()
    reasons = []
    hit = [t for t in parties if re.search(rf"\b{re.escape(t)}\b", low)]
    if hit:
        reasons.append("party_name:" + ",".join(hit[:3]))
    hit = [t for t in judges if re.search(rf"\b{re.escape(t)}\b", low)]
    if hit:
        reasons.append("judge_name:" + ",".join(hit[:3]))
    for name, rx in LEAK_PATTERNS.items():
        if rx.search(q):
            reasons.append(name)
    years = [y for y in re.findall(r"\b(?:18|19|20)\d{2}\b", q) if y not in ACT_YEARS]
    if years:
        reasons.append("year:" + ",".join(years[:3]))
    return reasons


# ---------------------------------------------------------------------------- metrics
METRICS = ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10", "recall@10")


def query_metrics(ranking: Sequence[str], relevant: Sequence[str], k: int = 10) -> Dict[str, float]:
    """Judgment-level metrics for one query. ranking: doc_ids best first; relevant: target doc_ids."""
    rel = set(relevant)
    top = list(ranking)[:k]
    ranks = [i for i, d in enumerate(top, start=1) if d in rel]
    first = ranks[0] if ranks else None
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel), k) + 1)) or 1.0
    return {
        "hit@1": float(first is not None and first <= 1),
        "hit@3": float(first is not None and first <= 3),
        "hit@5": float(first is not None and first <= 5),
        "hit@10": float(first is not None and first <= 10),
        "mrr@10": 1.0 / first if first else 0.0,
        "ndcg@10": dcg / idcg,
        "recall@10": len(set(top) & rel) / len(rel) if rel else 0.0,
        "rank": first,
    }


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05) -> List[float]:
    """Percentile bootstrap CI of the mean."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    return [round(float(np.percentile(means, 100 * alpha / 2)), 4), round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4)]


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], n_resamples: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """mean(a - b) with a paired percentile bootstrap CI (same resampled queries for both systems)."""
    import numpy as np

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {"delta": round(float(d.mean()), 4) if d.size else None, "ci": bootstrap_ci(d, n_resamples, seed)}


def latency_stats(ms: Sequence[float]) -> Dict[str, float]:
    import numpy as np

    if not ms:
        return {"mean": None, "p50": None, "p95": None}
    a = np.asarray(ms, dtype=float)
    return {"mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1)}


def summarize(per_query: List[Dict[str, float]], with_ci: bool = True, seed: int = 42) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(per_query)}
    for m in METRICS:
        vals = [q[m] for q in per_query]
        mean = round(sum(vals) / len(vals), 4) if vals else None
        out[m] = {"value": mean, "ci": bootstrap_ci(vals, seed=seed)} if with_ci else mean
    return out


def full_ranking(reranked_doc_ids: Sequence[str], fused_doc_order: Sequence[str], k: int = 10) -> List[str]:
    """Reranked judgments first, then the remaining fused (RRF) judgments in fused order, up to k."""
    seen = []
    for d in list(reranked_doc_ids) + list(fused_doc_order):
        if d and d not in seen:
            seen.append(d)
        if len(seen) >= k:
            break
    return seen
