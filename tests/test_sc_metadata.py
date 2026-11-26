"""Unit tests for src/sc_metadata.py, built on REAL cleaned raw_html text.

The fixtures were copied from scripts/peek_raw_html.py output on the server (26 Nov):
- 2019 records: first 2,000 cleaned characters as printed (disclaimer + header). The
  trailing 'Decision Date | Case No | ...' block lies beyond that cut-off, so it is absent here.
- 2025 records: the full cleaned text (835 / 823 characters).
No I/O, no models, no network.
"""

import unittest

from src.sc_metadata import (
    bench_type,
    clean_html,
    iso_date,
    is_criminal,
    normalize_case_type,
    parse_case_no,
    parse_coram,
    parse_record,
    raw_case_type,
    strip_boilerplate,
)

DISCLAIMER = (
    "English हिन्दी - Hindi ਪੰਜਾਬੀ - Punjabi Disclaimer Due care and caution has been taken by the Editorial "
    "Section, Supreme Court of India to provide complete and accurate information in the English version. "
    "Judgments in regional languages are also made available alongside the English version. The translations "
    "are being done with the help of various human agencies and software tools. Reasonable efforts have been "
    "made to provide an accurate translation. Hindi Judgment as published in Ucchtam Nyayalya Nirnay Patrika "
    "are reproduced under approval of Vidhi Sahitya Prakashan, Department of legislative affairs, Govt. Of "
    "India. Supreme Court Registry will not responsible for incorrect or inaccurate translation and for any "
    "error, omission or discrepancy in the content of translated text. The translation of Judgements is "
    "provided for general information and shall have no legal effect for compliance or enforcement. If any "
    "questions arise related to the accuracy of the information/statement contained in the translated "
    "judgment, users are advised to verify from the original judgments and also to refer to correct position "
    "of law while referring to old judgments. Visitors to the site are requested to cross check the "
    "correctness of the information on this site with the authorities concerned or consult the relevant "
    "record. The information made available here is not meant for legal evidence. Neither the Courts "
    "concerned nor the National Informatics Centre (NIC) nor the e-Committee is responsible for any data "
    "inaccuracy or delay in the updation of the data on this website. We do not accept any responsibility or "
    "liability for any damage or loss arising from the direct/indirect use of the information provided on the "
    "site. However, we shall be obliged if errors/omissions are brought to our notice for carrying out the "
    "corrections."
)

# path=2019_3_400_410, judge=R.F. NARIMAN, decision_date=26-02-2019
TITLE_2019_1 = "AARISH ASGAR QURESHI versus FAREED AHMED QURESHI & ANR."
TEXT_2019_1 = (
    DISCLAIMER + " AARISH ASGAR QURESHI versus FAREED AHMED QURESHI & ANR. - [2019] 3 S.C.R. 400 "
    "2019 INSC 270 Coram : R.F. NARIMAN * , VINEET SARAN Code of Criminal Procedure, 1973"
)

# path=2019_2_185_198, judge=D.Y. CHANDRACHUD, decision_date=15-02-2019 (cut off mid-word)
TEXT_2019_2 = (
    DISCLAIMER + " MEHBOOB-UR-REHMAN (D) THR. LRS. versus SUDHA SEETHARAM & ANR - [2019] 2 S.C.R. 185 "
    "2019 INSC 216 Coram : D.Y. CHANDRACHUD * , HEMANT GUPTA Code of Criminal Proced"
)

# path=2019_17_405_421, judge=L. NAGESWARA RAO, HEMANT GUPTA (no '*' author mark)
TITLE_2019_3 = "UNION OF INDIA & ORS. versus P. S. GILL"
TEXT_2019_3 = (
    DISCLAIMER + " UNION OF INDIA & ORS. versus P. S. GILL - [2019] 17 S.C.R. 405 2019 INSC 1290 "
    "Coram : L. NAGESWARA RAO, HEMANT GUPTA Armed Forces Tribunal Act, 2007: ss.14 and 15"
)

# path=2025_7_2012_2052, judge=B.V. NAGARATHNA, decision_date=29-07-2025 (full cleaned text)
TITLE_2025_1 = "SUKHDEV YADAV @ PEHALWAN versus STATE OF (NCT OF DELHI) & OTHERS"
TEXT_2025_1 = (
    "SUKHDEV YADAV @ PEHALWAN versus STATE OF (NCT OF DELHI) & OTHERS - [2025] 7 S.C.R. 2012 2025 INSC 969 "
    "Coram : B.V. NAGARATHNA * , K.V. VISWANATHAN Issue for Consideration Whether, an accused/convict who has "
    "completed his “life imprisonment for a fixed term” such as twenty years of actual sentence without "
    "remission, as in the instant case, is entitled to be released from prison on completion of such a "
    "sentence or in other of the fixed term of sentence as aforesaid, should the accused/convict seek "
    "remission of his sentence of “life imprisonment” by making an application to the competent authority "
    "for seeking “reduction of his sentence”. Headnotes† Penal Code, 1860 – ss.53, Decision Date : "
    "29-07-2025 | Case No : CRIMINAL APPEAL No. 3271/2025 | Disposal Nature : Disposed off | Bench : 2 "
    "Judges Split view HTML view Flip view PDF"
)

# path=2025_1_1009_1040, judge=SURYA KANT, decision_date=28-01-2025 (full cleaned text)
TITLE_2025_2 = "IVAN RATHINAM versus MILAN JOSEPH"
TEXT_2025_2 = (
    "IVAN RATHINAM versus MILAN JOSEPH - [2025] 1 S.C.R. 1009 2025 INSC 115 Coram : SURYA KANT * , UJJAL "
    "BHUYAN Issue for Consideration Whether the presumption of legitimacy, if not displaced, determines "
    "paternity in law; whether the Civil Court had the jurisdiction to entertain the Original Suit and "
    "whether the Family Court was entitled to reopen the Maintenance Petition; whether the second round of "
    "by the principle of res judicata. Headnotes† Evidence Act, 1872 – s.112 – Birth during marriage, "
    "conclusive proof of legitimacy – Respondent and his mother claimed the appellant to be the respondent’s "
    "biological father despite the fact that the respondent was born Decision Date : 28-01-2025 | Case No : "
    "CRIMINAL APPEAL No. 413/2025 | Disposal Nature : Appeal(s) allowed | Bench : 2 Judges Split view HTML "
    "view Flip view PDF"
)

# Real HTML fragment (CLAUDE.md D1)
CASE_NO_HTML = "| Case No :</span><font color='green'> CRIMINAL APPEAL No. 645/2019</font>"


class TestCaseNumber(unittest.TestCase):
    def test_case_no_from_html_fragment(self):
        case_no = parse_case_no(CASE_NO_HTML)
        self.assertEqual(case_no, "CRIMINAL APPEAL No. 645/2019")
        self.assertTrue(is_criminal(case_no))
        self.assertEqual(raw_case_type(case_no), "CRIMINAL APPEAL")
        self.assertEqual(normalize_case_type(case_no), "Criminal Appeal")

    def test_case_no_from_trailing_block_text(self):
        self.assertEqual(parse_case_no(TEXT_2025_1), "CRIMINAL APPEAL No. 3271/2025")
        self.assertEqual(parse_case_no(TEXT_2025_2), "CRIMINAL APPEAL No. 413/2025")

    def test_case_no_missing(self):
        self.assertIsNone(parse_case_no(TEXT_2019_1))  # trailing block cut off in the fixture
        self.assertIsNone(parse_case_no(None))
        self.assertIsNone(parse_case_no(float("nan")))

    def test_case_type_rules(self):
        # Type names from the CLAUDE.md D2 table; numbers are placeholders.
        cases = {
            "SLP (CRIMINAL) No. 1/2020": "SLP (Crl)",
            "WRIT PETITION (CRIMINAL) No. 1/2020": "Writ Petition (Crl)",
            "REVIEW PETITION (CRIMINAL) No. 1/2020": "Review Petition (Crl)",
            "TRANSFER PETITION (CRIMINAL) No. 1/2020": "Transfer Petition (Crl)",
            "TRANSFERRED CASE (CRIMINAL) No. 1/2020": "Transferred Case (Crl)",
            "SUO MOTO WRIT (CRIMINAL) No. 1/2020": "Suo Motu Writ (Crl)",
            "CONTEMPT (CRIMINAL) No. 1/2020": "Contempt (Crl)",
            "CRIMINAL MISC. PETITION No. 1/2020": "Crl Misc. Petition",
            "CURATIVE (CRL) No. 1/2020": "Curative (Crl)",
        }
        for case_no, label in cases.items():
            with self.subTest(case_no=case_no):
                self.assertTrue(is_criminal(case_no))
                self.assertEqual(normalize_case_type(case_no), label)
        self.assertFalse(is_criminal("CIVIL APPEAL No. 1/2020"))
        self.assertEqual(normalize_case_type("CIVIL APPEAL No. 1/2020"), "CIVIL APPEAL")


class TestBoilerplateAndCoram(unittest.TestCase):
    def test_strip_disclaimer_with_and_without_title(self):
        self.assertTrue(strip_boilerplate(TEXT_2019_1, TITLE_2019_1).startswith(TITLE_2019_1))
        self.assertTrue(strip_boilerplate(TEXT_2019_1).startswith(TITLE_2019_1))
        self.assertTrue(strip_boilerplate(TEXT_2025_1).endswith("Bench : 2 Judges"))

    def test_coram_with_author_star(self):
        c = parse_coram(TEXT_2019_1)
        self.assertEqual(c["judges"], ["R.F. NARIMAN", "VINEET SARAN"])
        self.assertEqual(c["author_judge"], "R.F. NARIMAN")
        c = parse_coram(TEXT_2019_2)
        self.assertEqual(c["judges"], ["D.Y. CHANDRACHUD", "HEMANT GUPTA"])
        self.assertEqual(c["author_judge"], "D.Y. CHANDRACHUD")

    def test_coram_without_author_star(self):
        c = parse_coram(TEXT_2019_3)
        self.assertEqual(c["judges"], ["L. NAGESWARA RAO", "HEMANT GUPTA"])
        self.assertIsNone(c["author_judge"])

    def test_square_brackets_are_not_the_bench(self):
        scr_header = (
            "COURT REPORTS [2019] 13 S.C.R. MISS XYZ v. STATE OF GUJARAT & ANR (Criminal Appeal No.1619 of "
            "2019) OCTOBER 25, 2019 [UDAY UMESH LALIT, INDU MALHOTRA AND R..."
        )
        self.assertEqual(parse_coram(scr_header)["judges"], [])


class TestParseRecord(unittest.TestCase):
    def test_2019_unlabelled_record(self):
        r = parse_record(TEXT_2019_1, TITLE_2019_1)
        self.assertEqual(r["judges"], ["R.F. NARIMAN", "VINEET SARAN"])
        self.assertEqual(r["bench_strength"], 2)
        self.assertEqual(r["bench_type"], "Division (2)")
        self.assertEqual(r["bench_check"], "no_declared")
        self.assertEqual(r["scr_citation"], "[2019] 3 S.C.R. 400")
        self.assertEqual(r["neutral_citation"], "2019 INSC 270")
        self.assertFalse(r["headnote_labelled"])
        self.assertIsNone(r["headnote_snippet"])
        self.assertEqual(r["subject_hint"], "Code of Criminal Procedure, 1973")

        r3 = parse_record(TEXT_2019_3, TITLE_2019_3)
        self.assertIsNone(r3["author_judge_coram"])
        self.assertEqual(r3["subject_hint"], "Armed Forces Tribunal Act, 2007")
        self.assertEqual(r3["neutral_citation"], "2019 INSC 1290")

    def test_2025_labelled_record(self):
        r = parse_record(TEXT_2025_1, TITLE_2025_1)
        self.assertEqual(r["case_no"], "CRIMINAL APPEAL No. 3271/2025")
        self.assertEqual(r["case_type"], "Criminal Appeal")
        self.assertTrue(r["is_criminal"])
        self.assertEqual(r["judges"], ["B.V. NAGARATHNA", "K.V. VISWANATHAN"])
        self.assertEqual(r["author_judge_coram"], "B.V. NAGARATHNA")
        self.assertEqual(r["bench_declared"], 2)
        self.assertEqual(r["bench_check"], "match")
        self.assertEqual(r["decision_date_html"], "2025-07-29")
        self.assertEqual(r["disposal_nature_html"], "Disposed off")
        self.assertEqual(r["scr_citation"], "[2025] 7 S.C.R. 2012")
        self.assertTrue(r["headnote_labelled"])
        self.assertTrue(r["issue_for_consideration"].startswith("Whether, an accused/convict"))
        self.assertTrue(r["issue_for_consideration"].endswith("“reduction of his sentence”."))
        self.assertEqual(r["subject_hint"], "Penal Code, 1860")
        self.assertNotIn("Decision Date", r["headnote_snippet"])

    def test_2025_second_record(self):
        r = parse_record(TEXT_2025_2, TITLE_2025_2)
        self.assertEqual(r["judges"], ["SURYA KANT", "UJJAL BHUYAN"])
        self.assertEqual(r["author_judge_coram"], "SURYA KANT")
        self.assertEqual(r["decision_date_html"], "2025-01-28")
        self.assertEqual(r["disposal_nature_html"], "Appeal(s) allowed")
        self.assertTrue(r["issue_for_consideration"].startswith("Whether the presumption of legitimacy"))
        self.assertEqual(r["subject_hint"], "Evidence Act, 1872")

    def test_non_string_input(self):
        r = parse_record(None)
        self.assertEqual(r["judges"], [])
        self.assertIsNone(r["bench_strength"])
        self.assertEqual(r["bench_check"], "no_coram")
        self.assertFalse(r["is_criminal"])


class TestSmallHelpers(unittest.TestCase):
    def test_iso_date(self):
        self.assertEqual(iso_date("26-02-2019"), "2019-02-26")
        self.assertEqual(iso_date("29-07-2025"), "2025-07-29")
        self.assertIsNone(iso_date("2019-02-26"))
        self.assertIsNone(iso_date("32-01-2019"))
        self.assertIsNone(iso_date(None))

    def test_bench_type(self):
        self.assertEqual(bench_type(1), "Single")
        self.assertEqual(bench_type(2), "Division (2)")
        self.assertEqual(bench_type(3), "Full (3+)")
        self.assertEqual(bench_type(4), "Full (3+)")
        self.assertEqual(bench_type(5), "Constitution Bench")
        self.assertEqual(bench_type(7), "Constitution Bench")
        self.assertEqual(bench_type(None), "")

    def test_clean_html(self):
        self.assertEqual(clean_html("<b>A &amp; B</b>\n\n <i>C</i>"), "A & B C")
        self.assertEqual(clean_html(None), "")


if __name__ == "__main__":
    unittest.main()
