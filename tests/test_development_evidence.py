import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'demo'))
from development_evidence import development_metrics,development_saved
from dataset_catalog import read_json,read_rows,sha


class EvidenceTests(unittest.TestCase):
    def test_cohorts_and_review_ids_are_consistent(self):
        for key,n in [('bitext',156),('reliability',64),('insuranceqa',128),('mimic_demo',64)]:
            m=development_metrics(key);example=development_saved(key,n-1)
            self.assertEqual(example['total'],n);self.assertFalse(m['live_default_replaced'])
            self.assertEqual(sha(ROOT/m['records_path']),m['records_sha256'])
            if key=='reliability':self.assertEqual(len(m['review']['judgments']),64)
        example=development_saved('reliability',42)
        self.assertEqual(example['review']['verdict'],'Issue')
        self.assertIn('already been processed',example['review']['reason'])

    def test_ai_review_reproduces_every_frozen_record(self):
        directory=ROOT/'reports/reliability_workflow_audit/v1'
        protocol=read_json(directory/'protocol.json')
        source=ROOT/'outputs/prefix-alignment-v1/reliability/controlled_records.jsonl'
        self.assertEqual(sha(source),protocol['controlled_parsed_sha256'])
        judgments=read_rows(directory/'ai_judgments.jsonl');rows=read_rows(source)
        self.assertEqual([j['sample_id'] for j in judgments],list(range(64)))
        self.assertEqual([j['record'] for j in judgments],[r['record'] for r in rows])


if __name__=='__main__':unittest.main()
