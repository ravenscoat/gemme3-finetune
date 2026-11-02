from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from training_schedule import continuation_scale


class ContinuationScheduleTest(unittest.TestCase):
    def test_restarts_from_zero_and_warms_up_then_decays(self):
        self.assertEqual(continuation_scale(200, 200, 3900), 0)
        self.assertEqual(continuation_scale(225, 200, 3900), .5)
        self.assertEqual(continuation_scale(250, 200, 3900), 1)
        self.assertGreater(continuation_scale(1000, 200, 3900), 0)
        self.assertLess(continuation_scale(1000, 200, 3900), 1)
        self.assertEqual(continuation_scale(3900, 200, 3900), 0)


if __name__ == '__main__':
    unittest.main()
