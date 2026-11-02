"""Failures, unseen values and clinical types must remain visible in assessment."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import core,data,signature,validate_record,source_warnings
from dataset_metrics import assess_label,tv


class AssessmentTest(unittest.TestCase):
    def test_invalid_json_and_unseen_categories_do_not_disappear(self):
        original=core('mimic_demo',data('mimic_demo')['train'][0])
        unseen=copy.deepcopy(original);unseen['gender']='UNSEEN'
        self.assertEqual(validate_record('mimic_demo',unseen),[])
        self.assertTrue(source_warnings('mimic_demo',unseen))
        raw=[{'sample_id':0,'text':'```json\n'+json.dumps(original)+'\n```'},
             {'sample_id':1,'text':json.dumps(unseen)}, {'sample_id':2,'text':'{"age":'}]
        with tempfile.TemporaryDirectory() as directory:
            result,rows=assess_label('mimic_demo',raw,Path(directory)/'parsed.jsonl')
        self.assertEqual(result['attempted'],3)
        self.assertEqual(result['structure_valid'],2)
        self.assertEqual(len(rows),2)
        self.assertEqual(len(result['failures']),1)
        self.assertEqual(result['exact_training_record_match_rate'],.5)
        self.assertIn('UNSEEN',result['fidelity_vs_training']['categorical']['gender']['generated_counts'])

    def test_boolean_is_not_an_age_number(self):
        row=core('mimic_demo',data('mimic_demo')['train'][0]);row['age']=True
        self.assertTrue(validate_record('mimic_demo',row))
        self.assertNotEqual(signature(True),signature(1))

    def test_tv_counts_categories_outside_training_support(self):
        self.assertEqual(tv(['new'],['old']),1.)
        self.assertEqual(tv(['a','b'],['b','a','a','b']),0.)


if __name__=='__main__':unittest.main()
