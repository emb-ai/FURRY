"""Synthetic, dependency-free recording corruption regressions."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.inspect_recording import FRAME_FIELDS, INPUT_FIELDS, inspect


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'manifest.json').write_text(json.dumps(
            dict(schema_version=1, nq=1, nv=1, nu=1)))
        self.input = dict.fromkeys(INPUT_FIELDS, '0')
        self.input.update(sequence='1', xr_time_ns='100', receive_ns='200', valid='1',
                          head_flags='15', left_flags='15', right_flags='15',
                          left_active='1', right_active='1')
        self.frame_fields = FRAME_FIELDS + [f'reference_{i}' for i in range(29)] + ['qpos_0', 'qvel_0', 'ctrl_0']
        self.frame = dict.fromkeys(self.frame_fields, '0')
        self.frame.update(receive_ns='201', sequence='1', calibrate='1', valid='1', apply_reference='1')
        self.write('input.csv', INPUT_FIELDS, [self.input])
        self.write('frames.csv', self.frame_fields, [self.frame])
        self.write('events.csv', ['receive_ns', 'sequence', 'event'],
                   [dict(receive_ns='200', sequence='1', event='calibrate')])
        (self.path / 'complete.json').write_text(json.dumps(dict(
            schema_version=1, reason='user_stop', input_rows=1, frame_rows=1)))

    def write(self, name, fields, rows):
        with (self.path / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_finalized_episode_is_replayable(self):
        result = inspect(self.path)
        self.assertTrue(result['finalized'])
        self.assertTrue(result['replay_ready'])
        self.assertEqual(result['replayable_state_rows'], 1)

    def test_interrupted_episode_is_diagnostic_only(self):
        (self.path / 'complete.json').unlink()
        result = inspect(self.path)
        self.assertFalse(result['finalized'])
        self.assertFalse(result['replay_ready'])

    def test_wrong_and_duplicate_headers_are_rejected(self):
        for fields in (['wrong'] + INPUT_FIELDS[1:], [INPUT_FIELDS[1]] + INPUT_FIELDS[1:]):
            self.write('input.csv', fields, [dict.fromkeys(fields, '0')])
            with self.assertRaisesRegex(ValueError, 'schema columns'):
                inspect(self.path)

    def test_truncated_rows_are_rejected(self):
        with (self.path / 'frames.csv').open('a') as f:
            f.write('202,2\n')
        with self.assertRaisesRegex(ValueError, 'incomplete row'):
            inspect(self.path)

    def test_validity_flags_are_enforced(self):
        self.input['left_flags'] = '0'
        self.write('input.csv', INPUT_FIELDS, [self.input])
        with self.assertRaisesRegex(ValueError, 'valid poses'):
            inspect(self.path)

    def test_invalid_inputs_cannot_drive_references(self):
        self.input['valid'] = '0'
        self.write('input.csv', INPUT_FIELDS, [self.input])
        with self.assertRaisesRegex(ValueError, 'invalid input'):
            inspect(self.path)

    def test_stale_inputs_cannot_drive_references(self):
        self.frame['valid'] = '0'
        self.write('frames.csv', self.frame_fields, [self.frame])
        with self.assertRaisesRegex(ValueError, 'stale tracking'):
            inspect(self.path)

    def test_source_time_cannot_go_backwards(self):
        second = dict(self.input, sequence='2', xr_time_ns='99', receive_ns='202')
        self.write('input.csv', INPUT_FIELDS, [self.input, second])
        (self.path / 'complete.json').unlink()
        with self.assertRaisesRegex(ValueError, 'source timestamps'):
            inspect(self.path)

    def test_completion_counts_are_checked(self):
        self.write('frames.csv', self.frame_fields, [])
        with self.assertRaisesRegex(ValueError, 'Completion counts'):
            inspect(self.path)

    def test_calibration_without_applied_frames_is_not_replayable(self):
        self.frame['apply_reference'] = '0'
        self.write('frames.csv', self.frame_fields, [self.frame])
        self.assertFalse(inspect(self.path)['replay_ready'])

    def test_reset_invalidates_calibration_for_replay(self):
        self.frame.update(reset='1', calibrate='0')
        self.write('frames.csv', self.frame_fields, [self.frame])
        self.assertFalse(inspect(self.path)['replay_ready'])


if __name__ == '__main__':
    unittest.main()
