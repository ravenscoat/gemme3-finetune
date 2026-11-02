"""Guard narrow code screens against rejecting legitimate prefixes or overstating privacy."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from advanced_exploratory_audits import contacts,icd_prefix_issue


class AdvancedAuditTests(unittest.TestCase):
    def record(self,code,version):return {'primary_diagnosis_code':code,'primary_diagnosis_version':version}

    def test_legitimate_icd_prefixes_include_modern_u_and_icd9_e_v(self):
        for code,version in [('41401',9),('E8490',9),('V5861',9),('I25110',10),('U071',10),('e119',10),(None,None)]:
            with self.subTest(code=code):self.assertIsNone(icd_prefix_issue(self.record(code,version)))

    def test_obvious_cross_version_pairs_and_missing_code_are_not_accepted(self):
        for code,version in [('41401',10),('S0622A',9),('K543',9),('',9),(None,10)]:
            with self.subTest(code=code):self.assertIsNotNone(icd_prefix_issue(self.record(code,version)))
        # Deliberate limitation: this screen does not validate code-list membership.
        self.assertIsNone(icd_prefix_issue(self.record('V999999',9)))

    def test_contact_normalization_does_not_confuse_a_year_with_a_phone(self):
        values=contacts('In1978, email Example+Demo@Example.ORG or call +1 (202) 555-0147. Also202-555-0147.')
        self.assertEqual(values,{('email','example+demo@example.org'),('candidate_US_phone','2025550147')})
        self.assertFalse(contacts('The price is $30 per month; a plan lasts30 years.'))


if __name__=='__main__':unittest.main()
