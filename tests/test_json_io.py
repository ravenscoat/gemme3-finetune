"""Progress file contention must not interrupt GPU training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import json_io


class JsonWriteTest(unittest.TestCase):
    def test_transient_windows_lock_retries_and_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'progress.json'
            target.write_text('{"step": 18}')
            real_replace = json_io.os.replace
            failures = [PermissionError('Windows sharing violation'), PermissionError('Windows access denied')]
            def replace(source, destination):
                if failures:
                    raise failures.pop()
                real_replace(source, destination)
            with patch.object(json_io.os, 'replace', side_effect=replace) as replacement, patch.object(json_io.time, 'sleep'):
                self.assertTrue(json_io.write_json(target, {'step': 19}))
                self.assertEqual(replacement.call_count, 3)
            self.assertEqual(json.loads(target.read_text()), {'step': 19})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_persistently_locked_display_keeps_old_json_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'progress.json'
            target.write_text('{"step": 18}')
            with patch.object(json_io.os, 'replace', side_effect=PermissionError('locked')), patch.object(json_io.time, 'sleep'), warnings.catch_warnings(record=True) as messages:
                self.assertFalse(json_io.write_json(target, {'step': 19}, best_effort=True, attempts=2))
                self.assertEqual(len(messages), 1)
            self.assertEqual(json.loads(target.read_text()), {'step': 18})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_critical_artifact_failure_still_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(json_io.os, 'replace', side_effect=PermissionError('locked')), patch.object(json_io.time, 'sleep'):
                with self.assertRaises(PermissionError):
                    json_io.write_json(Path(directory) / 'selection.json', {'selected': True}, attempts=2)


if __name__ == '__main__':
    unittest.main()
