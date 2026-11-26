"""Tests for src/judgment_text.py on REAL PDF text (server_logs/peek_pdf_pages.log, 26 Nov).

Fixtures are pypdf extract_text() output as printed by scripts/peek_pdf_pages.py: full page 1,
the first 1,500 chars of page 2 and the last 600 chars of the final page. No PDFs, no network.
"""

import subprocess
import sys
import unittest
from pathlib import Path

from src import judgment_text as jt

# --------------------------------------------------------------------------- 2019 (old SCR)
P2019_1 = """A
B
C
D
E
F
G
H
208 SUPREME COURT REPORTS [2019] 4 S.C.R.
SUGREEV KUMAR
v.
STATE OF PUNJAB & ORS.
(Criminal Appeal No. 509 of 2019)
MARCH 15, 2019
[ABHAY MANOHAR SAPRE AND
DINESH MAHESHWARI, JJ.]
Code of Criminal Procedure, 1973:
s. 319 – Application under – In a criminal proceeding trying
three accused for offences u/ss. 302, 307, 341 r/w s.34 IPC and u/
ss. 25, 54 and 59 of Arms Act, 1959 – For summoning eight persons
to face trial, on the basis of testimony of a witness whereby he had
deposed that eleven persons were involved in the crime – Trial court
summoned one person and dismissed the application qua  seven
persons – High Court affirmed the order of trial court – On appeal,
held: Section 319 empowers summoning of any person on the basis
of any relevant evidence – However, the power being discretionary,
is to be exercised sparingly – The test to be applied is the one which
is more than a prima facie  case as examined at the time of framing
charge, but not of satisfaction to the extent that the evidence if goes
uncontroverted, would lead to the conviction of the accused – In
the present case, consideration of the application had been as if
the existence of a case beyond reasonable doubt was being examined
against the proposed accused – This was clearly an erroneous
approach – Impugned order is set aside and application is restored
to be re-examined by the trial court.
Partly allowing the appeal, the Court
HELD: 1. The provisions contained in Section 319 CrPC
are to achieve the objective that the real culprit should not get
away unpunished. By virtue of these provisions, the Court is
empowered to proceed against any person not shown as an
accused, if it appears from evidence that such person has
committed any offence for which, he could be tried together with
the other accused persons. [Para 10][214-F, G]
[2019] 4 S.C.R. 208
208"""

P2019_2 = """A
B
C
D
E
F
G
H
209
2. The provisions contained in Section 319 CrPC sanction
the summoning of any person on the basis of any relevant evidence
as available on record. However, it being a discretionary power
and an extraordinary one, is to be exercised sparingly and only
when cogent evidence is available. The prime facie  opinion which
is to be formed for exercise of this power requires stronger
evidence than mere probability of complicity of a person. The
test to be applied is the one which is more than a prime facie  case
as examined at the time of framing charge but not of satisfaction
to the extent that the evidence, if goes uncontroverted, would
lead to the conviction of the accused. [Para 12][216-F, G]"""

P2019_LAST = """made by the prosecution under Section 319 CrPC are restored for
reconsideration of the Trial Court. In the interest of justice, it is made
clear that we have not pronounced on the merits of the case either way
and it would be expected of the Trial Court to reconsider the prayer of
prosecution for proceeding against the proposed accused persons totally
uninfluenced by any observation herein regarding facts of the case but
with due regard to the evidence on record and to the law applicable.
Kalpana K. Tripathy Appeal partly allowed.
SUGREEV KUMAR v. STATE OF PUNJAB & ORS.
[DINESH MAHESHWARI, J.]"""

# --------------------------------------------------------------------------- 2023 (e-SCR, CASE DETAILS)
P2023_1 = """[2023] 11 S.C.R. 34 : 2023 INSC 675
34
CASE DETAILS
KISHORE BALKRISHNA NAND
v.
STATE OF MAHARASHTRA & ANR.
(Criminal Appeal No. 2291 of 2011)
AUGUST 02, 2023
[J. B. PARDIWALA AND MANOJ MISRA, JJ.]
HEADNOTES
Issue for consideration :  Whether the allegations made in the
complaint in the good faith addressed to the lawful authority-Sub Divisional
Magistrate, make out an oﬀ ence of defamation under Section 500 IPC or not.
Penal Code, 1860 – s.499, eighth exception, s. 500 – Accusation
preferred in good faith – When not defamation:
Held :  Accusation in the complaint preferred in good faith against
any person, to a person in lawful authority with regard to the subject-matter
of accusation, is not a defamation – By perusing the allegations made in the
complaint, no case for defamation made out – Impugned order passed by
the High Court set aside – Original order passed by the Magistrate issuing
summons, quashed and set aside. [Paras 16 and 17]
OTHER CASE DETAILS INCLUDING IMPUGNED
ORDER AND APPEARANCES
CRIMINAL APPELLATE JURISDICTION: Criminal Appeal No.
2291 of 2011.
From the Judgment and Order dated 03.02.2010 of the High Court
of Bombay at Nagpur in CRLWP No. 676 of 2009.
Appearances:
Satyajit A Desai, Anshuman Ashok, Siddharth Gautam, Abhinav K.
Mutyalwar, Gajanan N Tirthkar, Vijay Raj Singh Chouhan, Ms. Anagha S.
Desai, Advs. for the Appellant."""

P2023_2 = """35KISHORE BALKRISHNA NAND v.
STATE OF MAHARASHTRA & ANR.
Nishant Ramakantrao Katneshwarkar, Aaditya Aniruddha Pande,
Siddharth Dharmadhikari, Bharat Bagla, Sourav Singh, Aditya Krishna,
Advs. for the Respondents.
JUDGMENT / ORDER OF THE SUPREME COURT
ORDER
1. The respondent No.2 (original complainant) although served with
the notice issued by this Court, yet has chosen not to remain present before
this Court, either in-person or through an advocate, and oppose this appeal.
2. This is an appeal at the instance of the original accused summoned
for the oﬀ ence of defamation punishable under Section 500 of the Indian
Penal Code (for short, “the IPC”) and is directed against the order passed by
the High Court of Judicature at Bombay, Nagpur Bench, dated 03.02.2010 in
Criminal Writ Petition No.676 of 2009, by which the High Court rejected the
writ petition ﬁ led by the appellant – Kishore Balkrishna Nand and thereby
declined to quash the order of issue of process by the Magistrate for the
oﬀ ence of defamation.
3. It appears from the materials on record that the appellant herein
lodged a complaint in writing addressed to the Sub-Divisional Magistrate
(for short, “the SDM”) stating that the respondent no.2 herein (original
complainant) had put up a shop by encroaching upon some land.   In the
complaint. the appellant is said to have further stated that such shop put up
by the complainant was creating nuisance, as many anti-social elements
and road romeos had start"""

P2023_LAST = """17. In the overall view of the matter, we are convinced that the appeal
deserves to be allowed and is hereby allowed. The impugned order passed
by the High Court is hereby set aside. As a consequence of the same, the
original order passed by the Magistrate issuing summons, is also hereby
quashed and set aside. The criminal proceedings in the form of Criminal
Case No.247 of 2002 pending in the Court of Judicial Magistrate First Class,
Worora, Chandrapur, Maharashtra stand terminated.
18. Pending application(s), if any, stands disposed of.
Headnotes prepared by:  Appeal allowed.
Nidhi Jain"""

# --------------------------------------------------------------------------- 2025 (e-SCR 2024-25)
P2025_1 = """[2025] 11 S.C.R. 197 : 2025 INSC 1210
Sankar Padam Thapa
v.
Vijaykumar Dineshchandra Agarwal
(Criminal Appeal No. 4402 of 2025)
09 October 2025
[Ahsanuddin Amanullah* and
Prashant Kumar Mishra, JJ.]
Issue for Consideration
Whether in the absence of a Trust being made an accused in a
complaint under the Negotiable Instruments Act, 1881, when a
Cheque has been issued on behalf of a Trust, the said complaint
would be maintainable against the Chairman/a Trustee of the
said Trust.
Headnotes†
Negotiable Instruments Act, 1881 – Trusts Act, 1882 – Cheque
issued in favour of the Appellant under the signature of the
Respondent-Chairman of the Trust as authorized signatory
of the Trust, was dishonoured – Appellant filed complaint
against the Respondent for offences u/ss.138 and 142, NI Act
as well as u/s.420, IPC – Respondent sought quashing of the
complaint on the ground that the Trust being a juristic entity
and a necessary party not having been added as a party, the
complaint was not maintainable – Complaint quashed by High
Court – Sustainability:
* Author"""

P2025_2 = """198 [2025] 11 S.C.R.
Supreme Court Reports
through the Trustee(s) and the objects thereof were for charitable
purposes – The Deed of Trust also provides for permitting one or
more Trustees to operate a bank account – Thus, it is the Trustees
alone, through whom the Trust funds/property(ies) are managed
and dealt with – The Trust itself is without any independent legal
status. [Paras 26, 29, 17, 40]"""

P2025_LAST = """[2025] 11 S.C.R.  225
Sankar Padam Thapa v. Vijaykumar Dineshchandra Agarwal
MISCELLANEOUS:
45. The Registry may seek suitable orders from Hon’ble the Chief Justice
of India apropos constitution of an appropriate Bench to decide the
pending reference in Special Leave Petition (Civil) No.18636/2019.
We clarify that the instant appeal is disposed of and not tagged with
the reference.
Result of the case: Appeal allowed.
†Headnotes prepared by: Divya Pandey"""

# 2025_1_592_609, author_source=unknown in the manifest (no '*')
P2025U_1 = """[2025] 1 S.C.R. 592 : 2025 INSC 93
Sadashiv Dhondiram Patil
v.
The State of Maharashtra
(Criminal Appeal No. 1718 of 2017)
09 January 2025
[J.B. Pardiwala and R. Mahadevan, JJ]
Issue for Consideration
Issue arose whether Village Police Patil is a Police Officer in terms
of s. 25 of the Evidence Act; and whether the High Court erred in
holding the appellant guilty of the offence of murder.
Headnotes†
Evidence Act, 1872 – ss.25, 106 – Extra judicial confession –
Admissibility – Extra judicial confession to Police Patil,
if  admissible – Appellant-husband charged for the murder
of his wife – Extra judicial confession allegedly made by the
appellant to the village Police Patil – Trial court acquitted
the accused for the offence punishable u/ss.302 and 201
IPC holding that extra judicial confession allegedly made by
the appellant to Village Police Patil was inadmissible as per
s.25 – High Court set aside the acquittal and held the appellant
guilty of the offence of murder – Correctness: """


def pages(*raw):
    return [jt.normalize_text(p) for p in raw]


class TestNormalisation(unittest.TestCase):
    def test_ligature_split_words_are_rejoined(self):
        self.assertEqual(jt.normalize_text("an oﬀ ence of defamation"), "an offence of defamation")
        self.assertEqual(jt.normalize_text("writ petition ﬁ led by"), "writ petition filed by")
        self.assertEqual(jt.normalize_text("switched oﬀ the light"), "switched off the light")  # stop word: no join
        self.assertNotIn("ﬀ", jt.normalize_text(P2023_2))

    def test_layouts(self):
        self.assertEqual(jt.detect_layout(pages(P2019_1, P2019_2)), "old_scr")
        self.assertEqual(jt.detect_layout(pages(P2023_1, P2023_2)), "escr_2023")
        self.assertEqual(jt.detect_layout(pages(P2025_1, P2025_2)), "escr_2024")


class TestBench(unittest.TestCase):
    def test_bench_lines_of_all_layouts(self):
        self.assertEqual(jt.parse_bench(P2019_1), {"judges": ["ABHAY MANOHAR SAPRE", "DINESH MAHESHWARI"], "author": None})
        self.assertEqual(jt.parse_bench(P2023_1), {"judges": ["J. B. PARDIWALA", "MANOJ MISRA"], "author": None})
        self.assertEqual(jt.parse_bench(P2025_1),
                         {"judges": ["AHSANUDDIN AMANULLAH", "PRASHANT KUMAR MISHRA"], "author": "AHSANUDDIN AMANULLAH"})
        self.assertEqual(jt.parse_bench(P2025U_1), {"judges": ["J.B. PARDIWALA", "R. MAHADEVAN"], "author": None})

    def test_name_key(self):
        self.assertEqual(jt.name_key("J. B. PARDIWALA"), jt.name_key("J.B. Pardiwala"))


class TestCleaning(unittest.TestCase):
    def test_old_scr_furniture_removed(self):
        c = jt.clean_pages(pages(P2019_1, P2019_2, P2019_LAST))
        lines = c["lines"]
        for junk in ("A", "H", "208 SUPREME COURT REPORTS [2019] 4 S.C.R.", "[2019] 4 S.C.R. 208", "208", "209",
                     "Kalpana K. Tripathy Appeal partly allowed.", "[DINESH MAHESHWARI, J.]"):
            self.assertNotIn(junk, lines)
        self.assertIn("HELD: 1. The provisions contained in Section 319 CrPC", lines)
        self.assertEqual(c["running_author"], "DINESH MAHESHWARI")
        self.assertEqual(c["disposition_pdf"], "Appeal partly allowed")

    def test_escr_2023_end_matter(self):
        c = jt.clean_pages(pages(P2023_1, P2023_2, P2023_LAST))
        self.assertEqual(c["disposition_pdf"], "Appeal allowed")
        for junk in ("Nidhi Jain", "[2023] 11 S.C.R. 34 : 2023 INSC 675", "34", "STATE OF MAHARASHTRA & ANR."):
            self.assertNotIn(junk, c["lines"])
        self.assertIn("18. Pending application(s), if any, stands disposed of.", c["lines"])

    def test_escr_2024_headers_and_footnotes(self):
        c = jt.clean_pages(pages(P2025_1, P2025_2, P2025_LAST))
        for junk in ("* Author", "198 [2025] 11 S.C.R.", "Supreme Court Reports", "[2025] 11 S.C.R. 225",
                     "Result of the case: Appeal allowed.", "†Headnotes prepared by: Divya Pandey"):
            self.assertNotIn(junk, c["lines"])
        self.assertEqual(c["disposition_pdf"], "Appeal allowed")


class TestSectionsAndAuthor(unittest.TestCase):
    def test_escr_2023_split_and_per_curiam(self):
        c = jt.clean_pages(pages(P2023_1, P2023_2, P2023_LAST))
        s = jt.split_sections(c["lines"])
        head, body = "\n".join(s["headnote"]), "\n".join(s["body"])
        self.assertEqual(s["headnote_split"], "details")
        self.assertEqual(s["body_marker"], "judgment_order_marker")
        self.assertTrue(head.startswith("HEADNOTES"))
        self.assertIn("Issue for consideration", head)
        self.assertIn("[Paras 16 and 17]", head)
        self.assertNotIn("Appearances", head)
        self.assertNotIn("Advs. for the Respondents", body)
        self.assertTrue(s["body"][0] == "ORDER" and s["is_order"])
        self.assertIn("offence of defamation", body)
        self.assertEqual(jt.find_author(None, c["running_author"], s["delivered_by"], s["body"], s["is_order"]),
                         ("Per Curiam", "pdf_order"))

    def test_escr_2024_without_end_marker_is_unsplit(self):
        # Only pages 1, 2 and the last page are in the fixture: no body marker, no "1." paragraph
        c = jt.clean_pages(pages(P2025_1, P2025_2, P2025_LAST))
        s = jt.split_sections(c["lines"])
        self.assertEqual(s["headnote_split"], "unsplit")
        self.assertEqual(s["headnote"], [])
        self.assertEqual(s["front"][-1], "Prashant Kumar Mishra, JJ.]")

    def test_author_fallback_order(self):
        self.assertEqual(jt.find_author("AHSANUDDIN AMANULLAH", None, None, [], False), ("AHSANUDDIN AMANULLAH", "pdf_star"))
        self.assertEqual(jt.find_author(None, "DINESH MAHESHWARI", None, [], False), ("DINESH MAHESHWARI", "pdf_running_header"))
        self.assertEqual(jt.find_author(None, None, "DINESH MAHESHWARI, J.", [], False), ("DINESH MAHESHWARI", "pdf_delivered_by"))
        self.assertEqual(jt.find_author(None, None, None, ["Judgment", "Ahsanuddin Amanullah, J."], False),
                         ("AHSANUDDIN AMANULLAH", "pdf_body_signature"))
        self.assertEqual(jt.find_author(None, None, None, ["1. Leave granted."], False), (None, "unknown"))


class TestStatutes(unittest.TestCase):
    def test_old_scr_headnote(self):
        text = "\n".join(jt.clean_pages(pages(P2019_1, P2019_2, P2019_LAST))["lines"])
        self.assertEqual(jt.statutes_cited(text),
                         "IPC 34; IPC 302; IPC 307; IPC 341; CrPC 319; Arms Act 25; Arms Act 54; Arms Act 59")

    def test_escr_texts(self):
        t2023 = "\n".join(jt.clean_pages(pages(P2023_1, P2023_2, P2023_LAST))["lines"])
        self.assertEqual(jt.statutes_cited(t2023), "IPC 499; IPC 500")
        self.assertEqual(jt.statutes_cited(jt.normalize_text(P2025U_1)),
                         "IPC 201; IPC 302; Evidence Act 25; Evidence Act 106")
        self.assertIn("IPC 420", jt.statutes_cited(jt.normalize_text(P2025_1)))
        self.assertEqual(jt.statutes_cited("Section 138 of the NI Act"), "")


class TestChunkingAndDocument(unittest.TestCase):
    ROW_2023 = {  # manifest row of 2023_11_34_38 (judges after the HEADNOTES fix)
        "doc_id": "997f9533e894a526", "year": 2023, "path": "2023_11_34_38",
        "title": "KISHORE BALKRISHNA NAND versus STATE OF MAHARASHTRA & ANR.",
        "case_no": "CRIMINAL APPEAL No. 2291/2011", "judges": "J.B. PARDIWALA; MANOJ MISRA", "bench_strength": 2,
        "bench_source": "coram", "author_judge": None, "author_source": "unknown",
        "decision_date": "2023-08-02", "citation": "2023 INSC 675", "case_type": "Criminal Appeal",
    }

    def test_build_document_2023(self):
        res = jt.build_document("/x/2023_11_34_38_EN.pdf", dict(self.ROW_2023),
                                pages=pages(P2023_1, P2023_2, P2023_LAST))
        self.assertNotIn("error", res)
        doc = res["doc"]
        self.assertEqual((doc["author_judge"], doc["author_source"]), ("Per Curiam", "pdf_order"))
        self.assertEqual(doc["bench_type"], "Division (2)")
        self.assertEqual(doc["layout"], "escr_2023")
        self.assertEqual(doc["headnote_split"], "details")
        self.assertEqual(doc["statutes_cited"], "IPC 499; IPC 500")
        self.assertEqual(doc["disposition_pdf"], "Appeal allowed")
        kinds = {c["section_kind"] for c in res["children"]}
        self.assertEqual(kinds, {"headnote", "body"})
        head = [c for c in res["children"] if c["section_kind"] == "headnote"]
        self.assertTrue(all(c["role"] == "Headnote" for c in head))
        self.assertTrue(head[0]["text"].startswith("[Headnote | KISHORE BALKRISHNA NAND"))
        first = res["children"][0]
        self.assertEqual(first["chunk_id"], "997f9533e894a526-p00000-c000")
        for key in ("doc_id", "parent_id", "case_title", "court", "decision_date", "year", "judges", "bench_strength",
                    "bench_type", "author_judge", "citation", "case_type", "statutes_cited", "role", "section_kind",
                    "source_path", "raw_text"):
            self.assertIn(key, first)
        self.assertTrue(all(len(c["raw_text"]) <= 800 for c in res["children"]))
        # deterministic IDs
        again = jt.build_document("/x/2023_11_34_38_EN.pdf", dict(self.ROW_2023), pages=pages(P2023_1, P2023_2, P2023_LAST))
        self.assertEqual([c["chunk_id"] for c in res["children"]], [c["chunk_id"] for c in again["children"]])

    def test_manifest_values_win_and_pdf_fills_missing_bench(self):
        row = dict(self.ROW_2023, bench_strength=None, judges=None, bench_source="",
                   author_judge="J.B. PARDIWALA", author_source="judge_column")
        doc = jt.build_document("/x/a.pdf", row, pages=pages(P2023_1, P2023_2, P2023_LAST))["doc"]
        self.assertEqual((doc["bench_strength"], doc["bench_source"]), (2, "pdf"))
        self.assertEqual(doc["judges"], "J. B. PARDIWALA; MANOJ MISRA")
        self.assertEqual((doc["author_judge"], doc["author_source"]), ("J.B. PARDIWALA", "judge_column"))

    def test_scanned_pdf_is_an_error_not_an_exception(self):
        res = jt.build_document("/x/b.pdf", {"doc_id": "d"}, pages=["", " "])
        self.assertIn("error", res)

    def test_child_windows_overlap_and_size(self):
        text = " ".join(f"word{i}" for i in range(600))
        wins = jt.child_windows(text, 800, 120)
        self.assertTrue(all(len(w) <= 800 for w in wins))
        self.assertGreater(len(wins), 3)
        self.assertTrue(wins[1].split()[0] in wins[0])  # overlap


class TestWorkerImports(unittest.TestCase):
    def test_worker_modules_do_not_import_torch(self):
        """Spawned workers re-import index_corpus (as __mp_main__) and src.judgment_text: no torch/chromadb."""
        root = Path(__file__).resolve().parent.parent
        code = ("import sys; import index_corpus, src.judgment_text; "
                "bad=[m for m in ('torch','chromadb','sentence_transformers','pydantic') if m in sys.modules]; "
                "print(','.join(bad) or 'clean')")
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip(), "clean", out.stderr[-500:])


if __name__ == "__main__":
    unittest.main()
