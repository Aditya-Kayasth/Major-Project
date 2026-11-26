"""Statutory Concordance Mapping and Query Expansion Engine.

Bridges the 2024 Indian Criminal Law transition:
- Indian Penal Code (IPC 1860) <--> Bharatiya Nyaya Sanhita (BNS 2023)
- Code of Criminal Procedure (CrPC 1973) <--> Bharatiya Nagarik Suraksha Sanhita (BNSS 2023)
- Indian Evidence Act (IEA 1872) <--> Bharatiya Sakshya Adhiniyam (BSA 2023)

Provides bidirectional query expansion to ensure BM25 keyword matching
and dense vector search retrieve precedents regardless of whether the query
uses historical or modern statutory nomenclature.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# CrPC (1973) <-> BNSS (2023) Mappings
CRPC_TO_BNSS: Dict[str, str] = {
    "482": "528",  # Inherent powers of High Court
    "439": "483",  # Special powers of High Court / Sessions regarding bail
    "438": "482",  # Anticipatory bail
    "437": "480",  # Bail in non-bailable offences by Magistrate
    "311": "348",  # Power to summon material witness or examine person present
    "167": "187",  # Procedure when investigation cannot be completed in 24 hours (remand/default bail)
    "173": "193",  # Report of police officer on completion of investigation (chargesheet)
    "154": "173",  # Information in cognizable cases (FIR)
    "156": "175",  # Police officer's power to investigate cognizable case
    "200": "223",  # Examination of complainant
    "207": "230",  # Supply of copies of chargesheet and documents to accused
    "227": "250",  # Discharge in Sessions trial
    "228": "251",  # Framing of charge in Sessions trial
    "239": "262",  # Discharge in warrant cases by Magistrate
    "313": "351",  # Power to examine the accused
    "319": "358",  # Power to proceed against other persons appearing to be guilty of offence
    "389": "430",  # Suspension of sentence pending the appeal; release of appellant on bail
    "397": "438",  # Calling for records to exercise powers of revision
    "401": "442",  # High Court's powers of revision
    "144": "163",  # Power to issue order in urgent cases of nuisance
    "107": "126",  # Security for keeping the peace
}

BNSS_TO_CRPC: Dict[str, str] = {v: k for k, v in CRPC_TO_BNSS.items()}

# IPC (1860) <-> BNS (2023) Mappings
IPC_TO_BNS: Dict[str, str] = {
    "302": "103",   # Murder
    "304": "105",   # Culpable homicide not amounting to murder
    "304A": "106",  # Causing death by negligence
    "304B": "80",   # Dowry death
    "307": "109",   # Attempt to murder
    "376": "64",    # Rape
    "376D": "70",   # Gang rape
    "354": "74",    # Assault or criminal force to woman with intent to outrage modesty
    "420": "318",   # Cheating and dishonestly inducing delivery of property
    "406": "316",   # Criminal breach of trust
    "498A": "85",   # Husband or relative of husband subjecting woman to cruelty
    "120B": "61",   # Criminal conspiracy
    "34": "3(5)",   # Common intention
    "149": "190",   # Unlawful assembly member guilty of offence in prosecution of common object
    "467": "338",   # Forgery of valuable security, will, etc.
    "468": "336",   # Forgery for purpose of cheating
    "471": "340",   # Using as genuine a forged document
    "506": "351",   # Criminal intimidation
    "124A": "152",  # Act endangering sovereignty, unity and integrity of India (Sedition replacement)
    "326": "117",   # Voluntarily causing grievous hurt by dangerous weapons
    "379": "303",   # Theft
    "392": "309",   # Robbery
    "395": "310",   # Dacoity
}

BNS_TO_IPC: Dict[str, str] = {v: k for k, v in IPC_TO_BNS.items()}

# IEA (1872) <-> BSA (2023) Mappings
IEA_TO_BSA: Dict[str, str] = {
    "65B": "63",    # Admissibility of electronic records
    "27": "23",     # Discovery of fact pursuant to information received from accused
    "32": "26",     # Cases in which statement of relevant fact by person who is dead is relevant (Dying declaration)
    "45": "39",     # Opinions of experts
    "114A": "119",  # Presumption as to absence of consent in certain prosecutions for rape
    "113A": "117",  # Presumption as to abetment of suicide by a married woman
    "113B": "118",  # Presumption as to dowry death
    "138": "143",  # Order of examinations (Examination-in-chief, cross, re-exam)
    "145": "148",  # Cross-examination as to previous statements in writing
    "154": "157",  # Question by party to his own witness (Hostile witness)
}

BSA_TO_IEA: Dict[str, str] = {v: k for k, v in IEA_TO_BSA.items()}



# Act-name patterns. The Act name is REQUIRED: a bare "Section 107" is never mapped, and a section
# is only mapped within its own code ("Section 107 IPC" must not pick up the CrPC 107 -> BNSS 126 entry).
_ACT_PATTERNS = {
    "CrPC": r"cr\.?\s?p\.?\s?c\.?|code\s+of\s+criminal\s+procedure",
    "BNSS": r"b\.?\s?n\.?\s?s\.?\s?s\.?|bharatiya\s+nagarik\s+suraksha\s+sanhita",
    "IPC": r"i\.?\s?p\.?\s?c\.?|indian\s+penal\s+code|penal\s+code",
    "BNS": r"b\.?\s?n\.?\s?s\.?(?!\s?s)|bharatiya\s+nyaya\s+sanhita",
    "Evidence Act": r"(?:indian\s+)?evidence\s+act|i\.?e\.?a\.?",
    "BSA": r"b\.?\s?s\.?\s?a\.?|bharatiya\s+sakshya\s+adhiniyam",
}
# source code -> (mapping, target code)
_MAPPINGS = {
    "CrPC": (CRPC_TO_BNSS, "BNSS"),
    "BNSS": (BNSS_TO_CRPC, "CrPC"),
    "IPC": (IPC_TO_BNS, "BNS"),
    "BNS": (BNS_TO_IPC, "IPC"),
    "Evidence Act": (IEA_TO_BSA, "BSA"),
    "BSA": (BSA_TO_IEA, "Evidence Act"),
}
_SEC = r"(?<![A-Za-z])(?:sections?|secs?\.?|ss?\.|u/ss?\.?|§)"
_NUMS = r"(\d+[A-Za-z]?(?:\s*(?:,|and|&|/|or|r/w|read\s+with)\s*\d+[A-Za-z]?)*)"


def _section_refs(query: str) -> List[Tuple[str, str]]:
    """(code, section) pairs written with an explicit Act name, e.g. 'Sections 302 and 34 IPC', 'IPC s. 302'."""
    refs: List[Tuple[str, str]] = []
    for code, act in _ACT_PATTERNS.items():
        act_re = rf"(?<![A-Za-z])(?:{act})(?![A-Za-z])"
        patterns = [
            rf"{_SEC}\s*{_NUMS}\s*,?\s*(?:of\s+(?:the\s+)?)?{act_re}",   # Section 302 IPC / s. 302 of the IPC
            rf"{act_re}\s*,?\s*{_SEC}\s*{_NUMS}",                         # IPC Section 302
        ]
        for pat in patterns:
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                for num in re.findall(r"\d+[A-Za-z]?", m.group(1)):
                    refs.append((code, num.upper()))
    return list(dict.fromkeys(refs))


class StatutoryConcordance:
    """Provides bidirectional statutory mapping and query expansion."""

    @classmethod
    def expand_query(cls, query: str) -> str:
        """Append the old/new code equivalents of every section cited WITH its Act name.

        Example:
            'Can proceedings be quashed under Section 482 CrPC?'
            -> 'Can proceedings be quashed under Section 482 CrPC? (statutory concordance: Section 528 BNSS)'
        """
        additions: List[str] = []
        lower = query.lower()
        for code, sec in _section_refs(query):
            mapping, target = _MAPPINGS[code]
            target_sec = mapping.get(sec)
            if not target_sec:
                continue
            addition = f"Section {target_sec} {target}"
            if addition.lower() not in lower:
                additions.append(addition)
        if not additions:
            return query
        return f"{query} (statutory concordance: {', '.join(dict.fromkeys(additions))})"

