"""Mixed prompt views retain source targets and patient separation."""
from collections import defaultdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import core,data,read_rows
from mimic_conditioned_data import PREPARED,conditioned_prompt,condition_from_record


class ConditionedDataTests(unittest.TestCase):
    def test_two_views_do_not_change_targets_or_add_patients(self):
        rows=read_rows(PREPARED/'train.jsonl');source={r['source_id']:r for r in data('mimic_demo')['train']}
        grouped=defaultdict(list)
        for r in rows:
            grouped[r['source_id']].append(r)
            self.assertEqual(r['completion'],core('mimic_demo',source[r['source_id']]))
            self.assertNotIn('source_id',r['completion']);self.assertNotIn('group_id',r['completion'])
            self.assertNotIn('source_id',r['prompt']);self.assertNotIn('group_id',r['prompt'])
        self.assertEqual(len(rows),460);self.assertEqual(len(grouped),230)
        self.assertTrue(all({r['view'] for r in pair}=={'generic','conditioned'} for pair in grouped.values()))
        validation=read_rows(PREPARED/'validation_generic.jsonl')
        self.assertFalse({r['group_id'] for r in rows}&{r['group_id'] for r in validation})

    def test_conditions_preserve_observed_zero_and_explicit_null(self):
        record={'primary_diagnosis_code':'N179','primary_diagnosis_version':10,
            'creatinine_mg_dl':0,'glucose_mg_dl':None,'hemoglobin_g_dl':12}
        c=condition_from_record(record);self.assertEqual(c['lab_missing_pattern'],'010')
        prompt=conditioned_prompt('Generate synthetic data.',c)
        self.assertIn('creatinine_mg_dl must be a non-null numeric value.',prompt)
        self.assertIn('glucose_mg_dl must be null.',prompt)
        self.assertIn('diagnosis code "N179", diagnosis version10.'.replace('version10','version 10'),prompt)


if __name__=='__main__':unittest.main()
