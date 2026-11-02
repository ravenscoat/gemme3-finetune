"""Quota planning must retain source proportions and missing-value meaning."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mimic_fidelity_pilot import allocate,missing_pattern


class FidelityPilotTests(unittest.TestCase):
    def test_small_batch_quota_rounding_preserves_total_and_source_proportions(self):
        counts={(9,'000'):112,(10,'000'):92,(10,'111'):14,(9,'111'):6,(9,'011'):2,(10,'001'):1,(9,'010'):1,(9,'001'):1,(10,'110'):1}
        planned=allocate(counts,64,random.Random(2026110117))
        self.assertEqual(sum(planned.values()),64)
        self.assertEqual(set(planned),set(counts))
        for key,count in planned.items():self.assertLess(abs(count-64*counts[key]/230),1)
        self.assertEqual(planned,allocate(counts,64,random.Random(2026110117)))

    def test_zero_is_observed_and_joint_null_patterns_remain_distinct(self):
        self.assertEqual(missing_pattern({'creatinine_mg_dl':0,'glucose_mg_dl':None,'hemoglobin_g_dl':None}),'011')
        self.assertEqual(missing_pattern({'creatinine_mg_dl':None,'glucose_mg_dl':None,'hemoglobin_g_dl':None}),'111')
        self.assertEqual(missing_pattern({'creatinine_mg_dl':.7,'glucose_mg_dl':90,'hemoglobin_g_dl':12}),'000')


if __name__=='__main__':unittest.main()
