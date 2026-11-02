import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_reliability_workflows import analyze


def row(actions,subcategory='known'):
    return {'category':'a','subcategory':subcategory,'difficulty':'easy','expected_output':{'required_actions':actions,'intent':'lookup','routing_decision':'ai_handle'}}


class WorkflowAuditTests(unittest.TestCase):
    def test_conditioned_counts_do_not_treat_novel_words_as_semantic_errors(self):
        source=[row(['lookup','answer']),row(['lookup','answer','confirm'])]
        result=analyze([{'sample_id':9,'record':row(['find','answer'])}],source)
        self.assertEqual(result['expected_counts_given_generated_groups'],{2:.5,3:.5})
        self.assertEqual(result['count_outside_group_support'],0)
        self.assertEqual(result['records_with_unseen_actions'],1)
        self.assertNotIn('semantic_failure_rate',result)

    def test_unknown_group_does_not_invent_reference_support(self):
        result=analyze([{'sample_id':8,'record':row(['lookup'],'unknown')}],[row(['lookup','answer'])])
        self.assertEqual(result['unknown_group_records'],1)
        self.assertEqual(result['expected_counts_given_generated_groups'],{})
        self.assertFalse(result['findings'][0]['count_outside_observed_group_support'])


if __name__=='__main__':unittest.main()
