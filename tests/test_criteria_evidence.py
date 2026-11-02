import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_criteria_evidence import categorical_sample_distances, distance, numeric_values
from dataset_metrics import tv


class CriteriaEvidenceTests(unittest.TestCase):
    def test_cached_category_distances_match_original_normalization(self):
        reference=['A!', 'a', None, 'b', 'B?', 'c']
        samples=[[0,1,2],[3,4,5],[0,3,5]]
        actual=categorical_sample_distances(reference,samples)
        for indices,value in zip(samples,actual):
            self.assertAlmostEqual(value,tv([reference[i] for i in indices],reference))

    def test_numeric_distance_ignores_nulls_and_booleans(self):
        self.assertEqual(distance([None,True,2,float('nan')],[2,2]),0)
        self.assertIsNone(distance([None],[2]))
        rows=[{'expected_output':{'required_actions':['a','b']}}]
        self.assertEqual(numeric_values('reliability',rows,'required_action_count'),[2])


if __name__=='__main__':unittest.main()
