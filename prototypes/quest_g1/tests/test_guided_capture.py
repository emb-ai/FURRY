import csv
import json
import tempfile
import unittest
from pathlib import Path
from scripts.inspect_capture import inspect_capture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.plan = dict(protocol='quest_virtual_legs_v1', split='test', target_seconds=300,
                         stages=[dict(id=str(i), seconds=60) for i in range(5)])
        self.summary = dict(split='test', completed=False, target_seconds=300,
                            accepted_seconds=.05, stage_seconds=[.05,0,0,0,0])
        self.json('capture_plan.json', self.plan); self.json('capture_summary.json', self.summary)
        self.json('complete.json', dict(schema_version=1,reason='user_stop',input_rows=1,frame_rows=1))
        self.rows('input.csv', [dict(sequence=1, head_flags=3, xr_time_ns=100)])
        self.rows('body.csv', [dict(sequence=1, valid=1, time_ns=100)])
        self.timeline = dict(receive_ns=200, sequence=1, stage=0, status=5, focused=1,
                             source_valid=1, calibrated=1, accepted_seconds=.05, stage_seconds=.05)
        self.rows('capture_timeline.csv', [self.timeline])

    def json(self, name, value): (self.path/name).write_text(json.dumps(value))
    def rows(self, name, rows):
        with (self.path/name).open('w') as f:
            w=csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

    def test_partial_test_recording_stays_held_out(self):
        r=inspect_capture(self.path)
        self.assertEqual(r['use'], 'held-out evaluation only')
        self.assertFalse(r['protocol_completed'])
        self.assertAlmostEqual(r['accepted_seconds'],.05)

    def test_v2_carrying_plan_is_readable_without_relabelling_v1(self):
        self.assertEqual(inspect_capture(self.path)['protocol'],'quest_virtual_legs_v1')
        self.json('capture_plan.json',dict(self.plan,protocol='quest_virtual_legs_v2'))
        self.assertEqual(inspect_capture(self.path)['protocol'],'quest_virtual_legs_v2')

    def test_pauses_and_invalid_sources_cannot_count(self):
        for field in ('focused','source_valid','calibrated'):
            with self.subTest(field=field):
                self.rows('capture_timeline.csv', [dict(self.timeline, **{field:0})])
                with self.assertRaisesRegex(ValueError,'ineligible'):inspect_capture(self.path)

    def test_missing_input_and_false_validity_are_rejected(self):
        self.rows('capture_timeline.csv', [dict(self.timeline, sequence=2)])
        with self.assertRaisesRegex(ValueError,'identity'):inspect_capture(self.path)
        self.rows('capture_timeline.csv', [self.timeline])
        self.rows('body.csv', [dict(sequence=1, valid=0, time_ns=100)])
        with self.assertRaisesRegex(ValueError,'invalid source'):inspect_capture(self.path)

    def test_short_capture_cannot_claim_completion(self):
        self.json('capture_summary.json',dict(self.summary,completed=True))
        with self.assertRaisesRegex(ValueError,'Incomplete'):inspect_capture(self.path)

    def test_stall_cannot_be_credited(self):
        self.rows('capture_timeline.csv',[dict(self.timeline,accepted_seconds=3,stage_seconds=3)])
        with self.assertRaisesRegex(ValueError,'stall'):inspect_capture(self.path)

    def test_summary_must_match_timeline(self):
        self.json('capture_summary.json',dict(self.summary,accepted_seconds=.09))
        with self.assertRaisesRegex(ValueError,'duration'):inspect_capture(self.path)

    def test_interrupted_capture_is_not_complete(self):
        (self.path/'complete.json').unlink();(self.path/'capture_summary.json').unlink()
        r=inspect_capture(self.path)
        self.assertFalse(r['finalized']);self.assertFalse(r['protocol_completed'])

    def test_completion_counts_detect_truncation(self):
        self.json('complete.json',dict(schema_version=1,reason='user_stop',input_rows=2,frame_rows=1))
        with self.assertRaisesRegex(ValueError,'counts'):inspect_capture(self.path)
