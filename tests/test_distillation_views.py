"""A teacher comparison must not silently change its control labels or contexts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import ROOT,data,read_json,read_rows,sha


class DistillationViewsTests(unittest.TestCase):
    def test_teacher_and_replay_views_match_on_source_conditions_and_budget(self):
        directory=ROOT/'outputs/reliability-distillation-v2'
        p=read_json(directory/'student_protocol.json')
        control=read_rows(directory/'replay_train.jsonl');student=read_rows(directory/'distilled_train.jsonl')
        self.assertEqual(len(control),len(student));self.assertEqual(len(control),p['rows_per_arm'])
        self.assertEqual(p['max_steps'],18)
        for a,b in zip(control,student):
            self.assertEqual(a['prompt'],b['prompt'])
            for field in ['category','subcategory','difficulty']:
                self.assertEqual(a['completion'][field],b['completion'][field])
            self.assertEqual(a['completion']['input']['context'],b['completion']['input']['context'])
            self.assertEqual(set(a['completion']),set(data('reliability')['schema']['fields']))
            self.assertEqual(set(b['completion']),set(data('reliability')['schema']['fields']))

    def test_only_bound_reviewed_teacher_records_are_eligible_for_training(self):
        directory=ROOT/'outputs/reliability-distillation-v2'
        review=read_rows(directory/'teacher_review.jsonl');accepted={r['sample_id'] for r in review if r['verdict']=='Pass'}
        used={r['teacher_sample_id'] for r in read_rows(directory/'distilled_train.jsonl') if r['source']=='teacher'}
        self.assertEqual(used,accepted)
        p=read_json(directory/'student_protocol.json')
        self.assertEqual(sha(directory/'teacher_review.jsonl'),p['review_sha256'])
        self.assertEqual(sha(directory/'teacher_valid_records.jsonl'),p['teacher_records_sha256'])
        self.assertEqual(sha(directory/'distilled_train.jsonl'),p['view_hashes']['distilled'])


if __name__=='__main__':unittest.main()
