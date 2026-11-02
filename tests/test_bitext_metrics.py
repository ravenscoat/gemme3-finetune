import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/evaluate_bitext.py"
spec = importlib.util.spec_from_file_location("metrics", path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class DistributionMetricsTest(unittest.TestCase):
    def test_outer_fence_only_normalization(self):
        self.assertEqual(metrics.strip_outer_json_fence('```json\n{"a":1}\n```'), '{"a":1}')
        self.assertEqual(metrics.strip_outer_json_fence('Comment\n```json\n{}\n```'), 'Comment\n```json\n{}\n```')

    def test_identical_distributions_have_zero_distance(self):
        self.assertEqual(metrics.tv(["a", "b"], ["a", "b"]), 0)
        self.assertEqual(metrics.numeric_distance([1, 2], [1, 2]), {"ks_statistic": 0, "wasserstein": 0})

    def test_disjoint_categories_and_numeric_shift(self):
        self.assertEqual(metrics.tv(["a"], ["b"]), 1)
        self.assertEqual(metrics.numeric_distance([1, 1], [3, 3]), {"ks_statistic": 1, "wasserstein": 2})

    def test_unequal_sample_sizes_represent_same_distribution(self):
        self.assertEqual(metrics.numeric_distance([1, 2], [1, 1, 2, 2]), {"ks_statistic": 0, "wasserstein": 0})


if __name__ == "__main__":
    unittest.main()
