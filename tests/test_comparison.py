import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generate_comparison import completed_rows
from analyze_comparison import nearest, per_intent, style, write_blinded_review


class ComparisonSafetyTest(unittest.TestCase):
    def test_per_intent_preserves_missing_groups_and_conditions_lengths(self):
        train = [{'intent': 'a', 'instruction': 'aa', 'response': 'aaa'},
                 {'intent': 'b', 'instruction': 'x' * 100, 'response': 'y' * 100}]
        result = per_intent([train[0]], train)
        self.assertTrue(result['b']['missing'])
        self.assertEqual(result['b']['valid_count'], 0)
        self.assertEqual(result['a']['length_distances']['instruction']['wasserstein'], 0.)
        self.assertEqual(result['a']['training_share'], .5)

    def test_resume_rejects_missing_duplicate_or_excessive_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            for ids in ([0, 2], [0, 0], [0, 1, 2]):
                path.write_text(''.join(json.dumps({'sample_id': i}) + '\n' for i in ids))
                with self.assertRaises(RuntimeError):
                    completed_rows(path, 2)
            path.write_text('{"sample_id": 0}\n')
            self.assertEqual(len(completed_rows(path, 2)), 1)

    def test_sparse_nearest_neighbor_and_zero_overlap(self):
        from scipy.sparse import csr_matrix
        scores, indices = nearest(csr_matrix([[1., 0.], [0., 1.]]),
                                  csr_matrix([[0., 1.], [0., 0.]]))
        self.assertEqual(scores, [1., 0.])
        self.assertEqual(indices[0], 1)

    def test_review_preserves_existing_annotations(self):
        row = {'sample_id': 0, 'instruction': 'hello', 'response': 'hello',
               'intent': 'agent', 'category': 'CONTACT', 'tags': 'B'}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_blinded_review(out, {'base': [row], 'tuned': [row]}, count=1)
            path = out / 'blinded_review.jsonl'
            original = path.read_text()
            self.assertNotIn('model_label', original)
            path.write_text('Human annotation retained\n')
            write_blinded_review(out, {'base': [row], 'tuned': [row]}, count=1)
            self.assertEqual(path.read_text(), 'Human annotation retained\n')

    def test_style_markers_are_literal_not_sentiment(self):
        row = {'instruction': 'i do not want fucking insurance?', 'response': '{{WEBSITE_URL}}'}
        result = style([row])
        self.assertEqual(result['request_contains_negation_marker'], 1.)
        self.assertEqual(result['request_contains_profanity_marker'], 1.)
        self.assertEqual(result['response_contains_placeholder'], 1.)


if __name__ == '__main__':
    unittest.main()
