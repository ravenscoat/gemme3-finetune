import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reliability_consistency_trial import OUT,IDS,ISSUES,prefix_for,GUARD
from dataset_catalog import read_json,read_rows


class ConsistencyTests(unittest.TestCase):
    def test_scenarios_fixed_without_supplied_target_actions(self):
        cases=read_rows(OUT/'cases.jsonl');p=read_json(OUT/'protocol.json')
        self.assertEqual([c['source_sample_id'] for c in cases],IDS)
        self.assertEqual(len(set(IDS)-set(ISSUES)),8)
        ending=', "expected_output": '
        for case,prefix in zip(cases,p['assistant_prefixes']):
            self.assertEqual(prefix,prefix_for(case['record']))
            supplied=json.loads(prefix[:-len(ending)]+'}')
            self.assertNotIn('expected_output',supplied)
            self.assertEqual(supplied['input'],case['record']['input'])

    def test_single_guard_generalizes_without_encoding_case_ids_or_answers(self):
        p=read_json(OUT/'protocol.json')
        self.assertEqual(p['prompts']['candidate'],p['prompts']['baseline']+GUARD)
        for value in ['process_payment','lookup_policy','3000','sample 42']:
            self.assertNotIn(value,GUARD)


if __name__=='__main__':unittest.main()
