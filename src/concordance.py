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


class StatutoryConcordance:
    """Provides bidirectional statutory mapping and query expansion."""

    @classmethod
    def expand_query(cls, query: str) -> str:
        """Analyze query for statutory references and inject concordance equivalents.

        Example:
            'Can stay be extended under Section 482 CrPC?'
            -> 'Can stay be extended under Section 482 CrPC (concordance: Section 528 BNSS)?'
        """
        expanded = query
        additions: List[str] = []

        # 1. Match CrPC -> BNSS
        crpc_matches = re.findall(r"(?:section|sec\.?|§)\s*(\d+[A-Z]?)\s*(?:of\s+the\s+)?(?:cr\.?p\.?c\.?|code of criminal procedure)?", query, flags=re.IGNORECASE)
        for sec in crpc_matches:
            sec_clean = sec.upper()
            if sec_clean in CRPC_TO_BNSS and "bnss" not in query.lower():
                bnss_sec = CRPC_TO_BNSS[sec_clean]
                additions.append(f"Section {bnss_sec} BNSS")

        # 2. Match BNSS -> CrPC
        bnss_matches = re.findall(r"(?:section|sec\.?|§)\s*(\d+[A-Z]?)\s*(?:of\s+the\s+)?(?:b\.?n\.?s\.?s\.?|bharatiya nagarik suraksha sanhita)", query, flags=re.IGNORECASE)
        for sec in bnss_matches:
            sec_clean = sec.upper()
            if sec_clean in BNSS_TO_CRPC and "crpc" not in query.lower():
                crpc_sec = BNSS_TO_CRPC[sec_clean]
                additions.append(f"Section {crpc_sec} CrPC")

        # 3. Match IPC -> BNS
        ipc_matches = re.findall(r"(?:section|sec\.?|§)\s*(\d+[A-Z]?)\s*(?:of\s+the\s+)?(?:i\.?p\.?c\.?|indian penal code)?", query, flags=re.IGNORECASE)
        for sec in ipc_matches:
            sec_clean = sec.upper()
            if sec_clean in IPC_TO_BNS and "bns" not in query.lower():
                bns_sec = IPC_TO_BNS[sec_clean]
                additions.append(f"Section {bns_sec} BNS")

        # 4. Match BNS -> IPC
        bns_matches = re.findall(r"(?:section|sec\.?|§)\s*(\d+[A-Z]?)\s*(?:of\s+the\s+)?(?:b\.?n\.?s\.?|bharatiya nyaya sanhita)", query, flags=re.IGNORECASE)
        for sec in bns_matches:
            sec_clean = sec.upper()
            if sec_clean in BNS_TO_IPC and "ipc" not in query.lower():
                ipc_sec = BNS_TO_IPC[sec_clean]
                additions.append(f"Section {ipc_sec} IPC")

        # 5. Match Evidence Act -> BSA
        iea_matches = re.findall(r"(?:section|sec\.?|§)\s*(\d+[A-Z]?)\s*(?:of\s+the\s+)?(?:evidence act|i\.?e\.?a\.?)", query, flags=re.IGNORECASE)
        for sec in iea_matches:
            sec_clean = sec.upper()
            if sec_clean in IEA_TO_BSA and "bsa" not in query.lower():
                bsa_sec = IEA_TO_BSA[sec_clean]
                additions.append(f"Section {bsa_sec} BSA")

        # Also check common standalone sections like "65B" or "482"
        if re.search(r"\b65B\b", query, flags=re.IGNORECASE) and "63 bsa" not in query.lower():
            additions.append("Section 63 BSA")

        if additions:
            unique_additions = list(dict.fromkeys(additions))
            expanded = f"{query} (statutory concordance: {', '.join(unique_additions)})"

        return expanded

