"""Teacher requests must remain train-only, credential-free and auditable."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import data,read_json,read_rows
from distillation_pilot import OUT


class DistillationPilotTests(unittest.TestCase):
    def test_teacher_requests_use_only_training_anchors_without_source_answers(self):
        source={r['source_id']:r for r in data('reliability')['train']}
        cases=read_rows(OUT/'requests.jsonl')
        self.assertEqual(len(cases),12)
        self.assertEqual(len({c['training_source_id'] for c in cases}),12)
        for c in cases:
            anchor=c['request']['source_anchor']
            self.assertNotIn('expected_output',anchor)
            self.assertEqual(anchor['input'],source[c['training_source_id']]['input'])
            self.assertEqual(c['request']['fixed_labels'],{f:source[c['training_source_id']][f] for f in ['category','subcategory','difficulty']})
        self.assertEqual({c['family'] for c in cases},{'history_consistency','policy_context','request_completeness','ordinary_control'})

    def test_teacher_protocol_does_not_store_credentials_or_claim_ground_truth(self):
        p=read_json(OUT/'protocol.json')
        self.assertNotIn('api_key',p)
        self.assertEqual(p['n'],12)
        self.assertIn('not insurance-expert ground truth',p['scope'])


if __name__=='__main__':unittest.main()
