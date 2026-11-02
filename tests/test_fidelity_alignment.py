"""Controlled batches must use source-only plans without leaking complete records."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import core,data,read_json
from fidelity_alignment import OUT,SIZES,age_band,bin_index,interval,features,numeric_values


class AlignmentTests(unittest.TestCase):
    def test_zero_missing_and_topcoded_age_remain_distinct(self):
        self.assertEqual(bin_index(None,[0,1,2]),'missing')
        self.assertEqual(bin_index(0,[0,1,2]),'0')
        self.assertEqual(interval(0,[0,1,2]),[0,1])
        self.assertEqual(age_band(90),[90,90]);self.assertEqual(age_band(89),[80,89])

    def test_plans_preserve_primary_quotas_and_provenance(self):
        for key,n in SIZES.items():
            p=read_json(OUT/key/'protocol.json')
            self.assertEqual(len(p['conditions']),n);self.assertEqual(len(set(p['selected_source_rows'])),n)
            self.assertEqual(sum(p['balance']['primary_quotas'].values()),n)
            self.assertLessEqual(p['balance']['final_L1'],p['balance']['initial_L1'])
            for prompt in p['prompts']:
                self.assertNotIn('source_id',prompt);self.assertNotIn('group_id',prompt)
            if key=='bitext':self.assertEqual(set(p['balance']['primary_quotas'].values()),{4})

    def test_numeric_profile_intervals_contain_their_source_values(self):
        for key in ['mimic_demo','insuranceqa']:
            p=read_json(OUT/key/'protocol.json');source=data(key)['train']
            lookup={r.get('source_id',r.get('source_line')):r for r in source}
            for ident,c in zip(p['selected_source_rows'],p['conditions']):
                values=numeric_values(key,core(key,lookup[ident]))
                if key=='mimic_demo':values['age']=lookup[ident]['age']
                for field,limits in c['ranges'].items():
                    if limits is None:self.assertIsNone(values[field])
                    else:self.assertTrue(limits[0]-1e-8<=values[field]<=limits[1]+1e-8)


if __name__=='__main__':unittest.main()
