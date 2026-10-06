import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import chromadb

# Ensure root path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings


def create_indexed_cases_metadata_json():
    settings = get_settings()
    chroma_dir = Path(settings.chroma_db_dir).resolve()
    parent_dir = Path(settings.parent_store_dir).resolve()
    collection_name = settings.collection_name

    print(f"[INFO] Connecting to ChromaDB at {chroma_dir}...")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_collection(collection_name)
    all_data = col.get(include=["metadatas", "documents"])

    # Aggregate by stem
    db_groups = {}
    for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        src = meta.get("source_path", "")
        stem = Path(src).stem if src else "unknown"
        if stem not in db_groups:
            db_groups[stem] = {
                "chunk_ids": [],
                "parent_ids": set(),
                "roles": {},
                "source_path": src,
                "first_chunk_id": cid,
            }
        db_groups[stem]["chunk_ids"].append(cid)
        if meta.get("parent_id"):
            db_groups[stem]["parent_ids"].add(meta.get("parent_id"))
        role = meta.get("role", "Obiter Dicta")
        db_groups[stem]["roles"][role] = db_groups[stem]["roles"].get(role, 0) + 1

    # Definitive metadata dictionary for each case indexed in the retrieval engine
    case_definitions = [
        {
            "case_id": "CASE-001",
            "document_stem": "654b1c982df7e33902d05c9e",
            "pdf_filename": "654b1c982df7e33902d05c9e.pdf",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\PROPERTY\654b1c982df7e33902d05c9e.pdf",
            "cached_text_path": "data/processed/654b1c982df7e33902d05c9e.txt",
            "legal_metadata": {
                "case_title": "Chandrapal Singh vs. State of U.P. and Another",
                "neutral_citation": "2023:AHC:212021-FB",
                "case_number": "APPLICATION U/S 482 No. 28574 of 2019",
                "court": "High Court of Judicature at Allahabad (Full Bench)",
                "bench": [
                    "Hon'ble Pritinker Diwaker, Chief Justice",
                    "Hon'ble Ashwani Kumar Mishra, J.",
                    "Hon'ble Ajay Bhanot, J."
                ],
                "judgment_date": "2023-11-03",
                "year": 2023,
                "petitioner": "Chandrapal Singh",
                "respondent": "State of U.P. and Another",
                "advocates": {
                    "applicant": ["Prashant Shukla", "Rajrshi Gupta", "Sudhanshu Kumar"],
                    "opposite_party": ["G.A.", "Rohit Shukla", "Shashi Kant Shukla"]
                },
                "legal_domain": "Criminal & Civil Procedure (Interim Stay Vacation Jurisprudence)",
                "subject_tags": [
                    "Asian Resurfacing Doctrine",
                    "Automatic Vacation of Stay Orders",
                    "Execution Proceedings",
                    "Inherent Powers (§ 482 CrPC)",
                    "Article 226 Constitutional Discretion"
                ],
                "core_legal_issue": "Whether the mandatory 6-month automatic stay vacation rule in Asian Resurfacing applies mechanically to execution proceedings or where hearings are pending through no fault of the litigant.",
                "ratio_decidendi": "The Full Bench held that automatic vacation of stay orders under Asian Resurfacing cannot apply mechanically to execution proceedings or where proceedings are delayed without fault of the litigant. The High Court's inherent power under Section 482 CrPC and plenary constitutional jurisdiction under Article 226 remain uncurtailed by administrative lapse.",
                "statutes_and_sections": [
                    "Constitution of India: Article 226 (Writ Jurisdiction)",
                    "Constitution of India: Article 141 (Law declared by Supreme Court)",
                    "Constitution of India: Article 142 (Complete Justice)",
                    "Code of Criminal Procedure (CrPC): Section 482 (Inherent Powers)",
                    "Code of Civil Procedure (CPC): Order 39 (Temporary Injunctions)",
                    "Code of Civil Procedure (CPC): Section 151 (Inherent Powers)"
                ],
                "precedents_cited": [
                    "Asian Resurfacing of Road Agency Pvt. Ltd. v. CBI ((2018) 16 SCC 299)",
                    "Dharam Vir Sood v. Savitri Devi",
                    "Atma Ram Properties (P) Ltd. v. Federal Motors (P) Ltd.",
                    "Fazaladdin v. State of U.P."
                ],
                "final_disposition": "Reference answered: automatic 6-month stay lapse does not apply mechanically to execution proceedings."
            }
        },
        {
            "case_id": "CASE-002",
            "document_stem": "2026041142",
            "pdf_filename": "2026041142.pdf",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\CIVIL\2026041142.pdf",
            "cached_text_path": "data/processed/2026041142.txt",
            "legal_metadata": {
                "case_title": "Minor Child K & Ors. vs. State (NCT of Delhi) & Anr.",
                "neutral_citation": "2025:DHC:1142",
                "case_number": "CRL.M.C. 3880/2025 & CRL.M.A. 16947/2025",
                "court": "High Court of Delhi at New Delhi",
                "bench": [
                    "Hon'ble Ms. Justice Swarana Kanta Sharma"
                ],
                "judgment_date": "2026-03-11 (Reserved on 16.12.2025)",
                "year": 2025,
                "petitioner": "Minor Child K & Ors.",
                "respondent": "State (NCT of Delhi) & Anr.",
                "advocates": {
                    "petitioners": ["Mr. Ashish Kumar", "Ms. Ritu Sharma"],
                    "respondents": ["Mr. Manoj Pant, APP for State", "Mr. D.S. Kohli for R-2"]
                },
                "legal_domain": "Criminal Law (POCSO Act & Child Witness Protection)",
                "subject_tags": [
                    "POCSO Act Child-Friendly Procedures",
                    "Vulnerable Witness Deposition Complexes (VWDC)",
                    "Cross-Examination of Minor Victims",
                    "Prohibition of Coercive Warrants on Victims",
                    "Secondary Victimization Prevention"
                ],
                "core_legal_issue": "Whether Special Courts can issue coercive processes (bailable warrants) against minor sexual assault victims or compel their repeated physical presence on bail hearings under the POCSO Act.",
                "ratio_decidendi": "Special Courts under the POCSO Act must strictly adhere to child-friendly examination protocols under Sections 33 and 36. Issuing coercive processes like warrants against minor victims or compelling repeated physical attendance on bail dates violates victim dignity and causes secondary victimization. Deposition must occur through Vulnerable Witness Deposition Complexes with child psychologists.",
                "statutes_and_sections": [
                    "Protection of Children from Sexual Offences (POCSO) Act: Section 6 (Aggravated Penetrative Sexual Assault)",
                    "POCSO Act: Section 33 (Procedure and powers of Special Court)",
                    "POCSO Act: Section 35 (Period for recording evidence and disposal)",
                    "POCSO Act: Section 36 (Child not to see accused at time of testifying)",
                    "POCSO Act: Section 40 (Right of child to legal representation)",
                    "Indian Penal Code (IPC): Section 376 (Rape)",
                    "Indian Penal Code (IPC): Section 363 (Kidnapping)",
                    "Code of Criminal Procedure (CrPC): Section 161 & 164 (Statements to Police/Magistrate)",
                    "Code of Criminal Procedure (CrPC): Section 439 (Special powers regarding bail)"
                ],
                "precedents_cited": [
                    "Smruti Tukaram Badade v. State of Maharashtra ((2022) SCC OnLine SC 78)",
                    "State of Karnataka v. Shivanna ((2014) 8 SCC 913)",
                    "Alakh Alok Srivastava v. Union of India ((2018) 11 SCC 615)"
                ],
                "final_disposition": "Petition allowed; bailable warrant quashed; nationwide child-friendly guidelines reaffirmed."
            }
        },
        {
            "case_id": "CASE-003",
            "document_stem": "2026071765",
            "pdf_filename": "2026071765.pdf",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\CIVIL\2026071765.pdf",
            "cached_text_path": "data/processed/2026071765.txt",
            "legal_metadata": {
                "case_title": "Pila Pahan @ Peela Pahan and others vs. State of Jharkhand and another",
                "neutral_citation": "2026 INSC 604",
                "case_number": "Writ Petition (Crl.) No. 169 of 2025 with WP (C) No. 489 of 2025",
                "court": "Supreme Court of India",
                "bench": [
                    "Hon'ble Mr. Justice J.B. Pardiwala",
                    "Hon'ble Mr. Justice R. Mahadevan"
                ],
                "judgment_date": "2026",
                "year": 2026,
                "petitioner": "Pila Pahan @ Peela Pahan and others",
                "respondent": "State of Jharkhand and another",
                "advocates": {
                    "petitioners": ["Senior Counsel for Petitioners"],
                    "respondents": ["Standing Counsel for State of Jharkhand"]
                },
                "legal_domain": "Constitutional Law & Judicial Administration (Reserved Judgments Delay)",
                "subject_tags": [
                    "Delay in Pronouncing Reserved Judgments",
                    "Right to Speedy Justice (Article 21)",
                    "Order XX Rule 1 CPC Mandates",
                    "De Novo Re-hearing Procedures",
                    "Judicial Accountability"
                ],
                "core_legal_issue": "What constitutional remedy lies when a High Court bench keeps a judgment reserved for over six months without pronouncement, infringing Article 21.",
                "ratio_decidendi": "Inordinate and unexplained delay in pronouncing reserved judgments violates the fundamental right to speedy justice under Article 21. Reserved judgments must ordinarily be pronounced within 2 to 3 months. If a judgment remains reserved beyond six months without pronouncement, the case must be withdrawn and listed before another bench for de novo hearing.",
                "statutes_and_sections": [
                    "Constitution of India: Article 21 (Right to Life & Speedy Justice)",
                    "Constitution of India: Article 14 (Equality & Non-arbitrariness)",
                    "Constitution of India: Article 32 (Supreme Court Writ Remedies)",
                    "Code of Civil Procedure (CPC): Order XX Rule 1 (Judgment when pronounced)",
                    "High Court of Jharkhand Rules: Practice and Procedure"
                ],
                "precedents_cited": [
                    "Anil Rai v. State of Bihar ((2001) 7 SCC 318)",
                    "R.C. Sharma v. Union of India ((1976) 3 SCC 574)",
                    "Kanhaiyalal v. Dinesh (AIR 1991 MP 101)"
                ],
                "final_disposition": "Directions issued; registry ordered to withdraw pending matter and list before fresh bench."
            }
        },
        {
            "case_id": "CASE-004",
            "document_stem": "2026092958",
            "pdf_filename": "2026092958.pdf",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\CIVIL\2026092958.pdf",
            "cached_text_path": "data/processed/2026092958.txt",
            "legal_metadata": {
                "case_title": "Basudev & Ors. vs. Sanjay Kumar & Ors.",
                "neutral_citation": "2026 INSC 831",
                "case_number": "Civil Appeal No. ___ of 2026 (Arising out of SLP (C) No. 4338 of 2025)",
                "court": "Supreme Court of India",
                "bench": [
                    "Hon'ble Mr. Justice K. Vinod Chandran",
                    "Hon'ble Mr. Justice Hrishikesh Roy"
                ],
                "judgment_date": "2026",
                "year": 2026,
                "petitioner": "Basudev & Ors.",
                "respondent": "Sanjay Kumar & Ors.",
                "advocates": {
                    "appellants": ["Learned Counsel for Appellants"],
                    "respondents": ["Learned Counsel for Respondents"]
                },
                "legal_domain": "Civil Procedure & Appellate Law (Composite Decrees & Counter-Claims)",
                "subject_tags": [
                    "Composite Appeal Maintainability",
                    "Order VIII Rule 6A CPC Counter-Claim",
                    "Order XLI Rule 1 CPC Appeal from Decree",
                    "Res Judicata (§ 11 CPC)",
                    "Dismissal of Counter-Claim in Common Judgment"
                ],
                "core_legal_issue": "Whether an appellant must file two separate appeals when a trial court decrees a suit and simultaneously dismisses a counter-claim in a single judgment, or if a single composite appeal is maintainable.",
                "ratio_decidendi": "When a suit and counter-claim are decided together in a common judgment and a composite decree is drawn up, a single composite appeal challenging both the suit decree and counter-claim dismissal is maintainable under Order XLI Rule 1 CPC, provided the grounds of appeal challenge both aspects and appropriate court fee is tendered.",
                "statutes_and_sections": [
                    "Code of Civil Procedure (CPC): Section 96 (Appeals from Original Decrees)",
                    "Code of Civil Procedure (CPC): Section 100 (Second Appeals)",
                    "Code of Civil Procedure (CPC): Section 11 (Res Judicata)",
                    "Code of Civil Procedure (CPC): Order VIII Rule 6A (Counter-claim by defendant)",
                    "Code of Civil Procedure (CPC): Order XX Rule 19 (Decree when counter-claim allowed)",
                    "Code of Civil Procedure (CPC): Order XLI Rule 1 (Form of appeal and contents)"
                ],
                "precedents_cited": [
                    "Lakshmi Ram Bhuyan v. Hari Prasad Bhuyan ((2003) 1 SCC 197)",
                    "Premier Tyres Ltd. v. K.S.R.T.C. (1993 Supp (2) SCC 146)",
                    "Ramagya Prasad Gupta v. Murli Prasad ((1974) 2 SCC 266)"
                ],
                "final_disposition": "Appeal allowed; High Court order rejecting composite appeal set aside; appeal restored."
            }
        },
        {
            "case_id": "CASE-005",
            "document_stem": "V_R_Sanal_Kumar_vs_Union_Of_India_on_12_May_2023",
            "pdf_filename": "V_R_Sanal_Kumar_vs_Union_Of_India_on_12_May_2023.PDF",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\CIVIL\V_R_Sanal_Kumar_vs_Union_Of_India_on_12_May_2023.PDF",
            "cached_text_path": "data/processed/V_R_Sanal_Kumar_vs_Union_Of_India_on_12_May_2023.txt",
            "legal_metadata": {
                "case_title": "Dr. V.R. Sanal Kumar vs. Union of India & Ors.",
                "neutral_citation": "2023 INSC 524 / Civil Appeal No. 6301 of 2013",
                "court": "Supreme Court of India",
                "bench": [
                    "Hon'ble Mr. Justice M.R. Shah",
                    "Hon'ble Mr. Justice C.T. Ravikumar"
                ],
                "judgment_date": "2023-05-12",
                "year": 2023,
                "petitioner": "Dr. V.R. Sanal Kumar (Scientist/Engineer-SF, VSSC/ISRO)",
                "respondent": "Union of India & Ors.",
                "advocates": {
                    "appellant": ["Senior Counsel for Appellant"],
                    "respondents": ["Additional Solicitor General for Union of India / ISRO"]
                },
                "legal_domain": "Constitutional Law & Civil Service Jurisprudence (State Security)",
                "subject_tags": [
                    "Article 311(2) Second Proviso Clause (c)",
                    "Dispensation of Departmental Inquiry",
                    "National Security & ISRO Propulsion Research",
                    "Pleasure Doctrine (Article 310)",
                    "Judicial Review Limits of Subjective Satisfaction"
                ],
                "core_legal_issue": "Whether the President's subjective satisfaction dispensing with an inquiry in the interest of the security of the State under Article 311(2)(c) is open to judicial review on sufficiency of materials.",
                "ratio_decidendi": "The second proviso clause (c) to Article 311(2) gives the President absolute constitutional discretion to dispense with an inquiry if satisfied that it is not expedient in the interest of the security of the State. The subjective satisfaction is non-justiciable as to adequacy or sufficiency of material; courts can only interfere if mala fides or complete absence of relevant material is proven.",
                "statutes_and_sections": [
                    "Constitution of India: Article 311(2) second proviso Clause (c) (State Security Exception)",
                    "Constitution of India: Article 310 (Tenure of office of persons serving the Union)",
                    "Constitution of India: Article 309 (Recruitment and conditions of service)",
                    "Constitution of India: Article 14 (Equality before Law)",
                    "ISRO Service Rules / Central Civil Services (Classification, Control & Appeal) Rules"
                ],
                "precedents_cited": [
                    "Union of India v. Tulsiram Patel ((1985) 3 SCC 398)",
                    "A.K. Kaul v. Union of India ((1995) 4 SCC 73)",
                    "Satyavir Singh v. Union of India ((1985) 4 SCC 252)"
                ],
                "final_disposition": "Appeal dismissed; dismissal of scientist by President under Article 311(2)(c) upheld."
            }
        },
        {
            "case_id": "CASE-006",
            "document_stem": "2026020987",
            "pdf_filename": "2026020987.pdf",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\CIVIL\2026020987.pdf",
            "cached_text_path": "data/processed/2026020987.txt",
            "legal_metadata": {
                "case_title": "Ram Swaroop Gupta & Ors. vs. State (NCT of Delhi) & Anr.",
                "neutral_citation": "2026:DHC:987",
                "case_number": "CRL.M.C. 537/2026, CRL.M.A. 2162/2026 & CRL.M.A. 2161/2026",
                "court": "High Court of Delhi at New Delhi",
                "bench": [
                    "Hon'ble Mr. Justice Anoop Kumar Mendiratta"
                ],
                "judgment_date": "2026-01-21",
                "year": 2026,
                "petitioner": "Ram Swaroop Gupta & Ors.",
                "respondent": "State (NCT of Delhi) & Anr.",
                "advocates": {
                    "petitioners": ["Mr. Ajatshatru Singh Rawat", "Ms. Naimishi Verma"],
                    "respondents": ["Mr. Sunil Kumar Gautam, APP for State"]
                },
                "legal_domain": "Criminal Procedure (Witness Examination & Fair Trial)",
                "subject_tags": [
                    "Section 311 CrPC Powers",
                    "Section 348 BNSS Correspondence",
                    "Recall of Material Witnesses",
                    "Prohibition on Filling Prosecution Lacunae",
                    "Fair Trial Rights of Accused"
                ],
                "core_legal_issue": "Whether the trial court can permit the recall of a witness under Section 311 CrPC when it has the effect of filling up prosecution lacunae after defense disclosure.",
                "ratio_decidendi": "The power under Section 311 CrPC is discretionary and must be exercised ex debito justitiae only to arrive at the truth. It cannot be used as a disguise or subterfuge to enable the prosecution to fill up gaps or lacunae in evidence to the irreparable prejudice of the accused.",
                "statutes_and_sections": [
                    "Code of Criminal Procedure (CrPC): Section 311 (Power to summon material witness)",
                    "Bharatiya Nagarik Suraksha Sanhita (BNSS): Section 348 (Power to summon witness)",
                    "Indian Evidence Act: Section 138 (Order of examinations)",
                    "Code of Criminal Procedure (CrPC): Section 397 (Calling for records in revision)"
                ],
                "precedents_cited": [
                    "Zahira Habibullah Sheikh v. State of Gujarat ((2004) 4 SCC 158)",
                    "Natasha Singh v. CBI ((2013) 5 SCC 741)",
                    "State of Haryana v. Ram Mehar ((2016) 8 SCC 320)"
                ],
                "final_disposition": "Petition allowed; trial court order allowing recall of witness set aside."
            }
        },
        {
            "case_id": "CASE-007",
            "document_stem": "CyberCrime",
            "pdf_filename": "CyberCrime.pdf",
            "pdf_source_path": r"E:\GIT_HUB\MAJOR PROJECT\Major Project Data\CRIMINAL\CyberCrime.pdf",
            "cached_text_path": "data/processed/CyberCrime.txt",
            "legal_metadata": {
                "case_title": "Shri Samir Raina vs. PIO, Ministry of Home Affairs (In Re: Cyber Crime Portal & Electronic Records)",
                "neutral_citation": "2024:CIC:634921 / Second Appeal No. CIC/MHOME/A/2024/634921",
                "court": "Central Information Commission (New Delhi)",
                "bench": [
                    "Shri Heeralal Samariya, Chief Information Commissioner"
                ],
                "judgment_date": "2025-08-19",
                "year": 2024,
                "petitioner": "Shri Samir Raina",
                "respondent": "Central Public Information Officer (CPIO), Ministry of Home Affairs (Cyber & Information Security Division)",
                "advocates": {
                    "appellant": ["In-person Appellant"],
                    "respondent": ["CPIO representative, MHA"]
                },
                "legal_domain": "Cyber Law, Electronic Records & Transparency",
                "subject_tags": [
                    "National Cybercrime Reporting Portal (NCRP)",
                    "Electronic Evidence Chain of Custody",
                    "Section 65B Electronic Records Certification",
                    "Citizen Financial Cyber Fraud Management System (CFCFRMS)",
                    "RTI Security Exemptions (§ 8(1)(a))"
                ],
                "core_legal_issue": "Disclosure standards for standard operating procedures governing cyber forensic tools and citizen complaint tracking on the National Cybercrime Portal.",
                "ratio_decidendi": "Electronic records and cyber forensic data must maintain contemporaneous cryptographic hash verification and compliance with Section 65B standards. System SOPs are disclosed to ensure citizen transparency while exempting underlying algorithmic source security under Section 8(1)(a) of the RTI Act.",
                "statutes_and_sections": [
                    "Information Technology Act: Section 65B (Electronic Records)",
                    "Indian Evidence Act: Section 65B (Admissibility of electronic records)",
                    "Bharatiya Sakshya Adhiniyam (BSA): Section 63 (Admissibility of electronic records)",
                    "Right to Information Act: Section 8(1)(a) (Exemption on security grounds)",
                    "Right to Information Act: Section 19 (Second Appeal to Commission)"
                ],
                "precedents_cited": [
                    "Arjun Panditrao Khotkar v. Kailash Kushanrao Gorantyal ((2020) 7 SCC 1)",
                    "Shafhi Mohammad v. State of Himachal Pradesh ((2018) 2 SCC 801)"
                ],
                "final_disposition": "Second appeal disposed of with directions to PIO to provide sanitized SOP details."
            }
        }
    ]

    total_chunks = len(all_data["ids"])
    total_parents = len(set(meta.get("parent_id") for meta in all_data["metadatas"] if meta.get("parent_id")))

    cases_payload = []
    for cdef in case_definitions:
        stem = cdef["document_stem"]
        db_stat = db_groups.get(stem, {
            "chunk_ids": [],
            "parent_ids": set(),
            "roles": {},
            "source_path": cdef["pdf_source_path"],
            "first_chunk_id": "N/A"
        })

        # File stats
        pdf_path = Path(cdef["pdf_source_path"])
        file_size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0

        cached_text_file = Path(cdef["cached_text_path"])
        char_count = 0
        word_count = 0
        if cached_text_file.exists():
            text_data = cached_text_file.read_text(encoding="utf-8", errors="replace")
            char_count = len(text_data)
            word_count = len(text_data.split())

        case_entry = {
            "case_id": cdef["case_id"],
            "document_stem": stem,
            "pdf_filename": cdef["pdf_filename"],
            "pdf_source_path": str(cdef["pdf_source_path"]),
            "pdf_size_bytes": file_size_bytes,
            "pdf_size_kb": round(file_size_bytes / 1024, 2),
            "cached_text_path": str(cdef["cached_text_path"]),
            "text_metrics": {
                "character_count": char_count,
                "word_count": word_count,
                "parent_chunk_count": len(db_stat["parent_ids"]),
                "child_chunk_count": len(db_stat["chunk_ids"]),
            },
            "legal_metadata": cdef["legal_metadata"],
            "vector_store_metadata": {
                "collection_name": collection_name,
                "indexed_child_chunks": len(db_stat["chunk_ids"]),
                "parent_store_sections": len(db_stat["parent_ids"]),
                "chunk_roles_distribution": db_stat["roles"],
                "embedding_model": settings.gemini_embedding_model,
                "embedding_dimension": 768,
                "vector_metric": "cosine",
                "sample_chunk_id": db_stat["first_chunk_id"] if db_stat["first_chunk_id"] != "N/A" else None,
            }
        }
        cases_payload.append(case_entry)

    master_metadata = {
        "system_info": {
            "engine_name": "Indian Legal Precedents RAG Retrieval Engine",
            "chromadb_collection": collection_name,
            "chromadb_store_path": str(chroma_dir),
            "parent_store_path": str(parent_dir),
            "total_indexed_cases": len(cases_payload),
            "total_indexed_chunks": total_chunks,
            "total_parent_sections": total_parents,
            "embedding_model": settings.gemini_embedding_model,
            "embedding_dimension": 768,
            "vector_distance_metric": "cosine",
            "bm25_index_file": str(chroma_dir / f"{collection_name}_bm25.pkl"),
            "reranker_model": "BAAI/bge-reranker-v2-m3 (FP16 half-precision on NVIDIA RTX 2050)",
            "export_timestamp": datetime.now().isoformat(),
        },
        "cases": cases_payload
    }

    # Save to data/indexed_cases_metadata.json
    out_file1 = Path("data/indexed_cases_metadata.json")
    out_file1.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file1, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=2, ensure_ascii=False)

    # Also save to data/processed/indexed_cases_metadata.json
    out_file2 = Path("data/processed/indexed_cases_metadata.json")
    out_file2.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file2, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Exported indexed cases metadata to:")
    print(f"  1. {out_file1}")
    print(f"  2. {out_file2}")
    print(f"Total Cases Documented: {len(cases_payload)}")
    print(f"Total Chunks Verified: {total_chunks:,}")


if __name__ == "__main__":
    create_indexed_cases_metadata_json()
