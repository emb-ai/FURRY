"""Reject broken full-body records; controller validity is independent of body validity."""
import json
import unittest
import test_recording
from scripts.inspect_recording import BODY_FIELDS, MIMIC_FIELDS, INPUT_FIELDS, inspect


class GmrRecordingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_recording.RecordingTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.path=self.fixture.path
        p=self.path/'manifest.json';d=json.loads(p.read_text());d['retargeting']='native_gmr_meta_full_body';p.write_text(json.dumps(d))
        self.body=dict.fromkeys(BODY_FIELDS,'0');self.body.update(sequence='1',time_ns='100',supported='1',valid='1',confidence='1')
        for j in range(14):self.body[f'flags_{j}']='15'
        self.command=dict.fromkeys(MIMIC_FIELDS,'0');self.command['sequence']='1'
        self.write()

    def write(self):
        self.fixture.write('body.csv',BODY_FIELDS,[self.body])
        self.fixture.write('mimic.csv',MIMIC_FIELDS,[self.command])

    def test_full_body_does_not_require_active_controllers(self):
        row=dict(self.fixture.input,valid='0',left_active='0',right_active='0',left_flags='0',right_flags='0')
        self.fixture.write('input.csv',INPUT_FIELDS,[row])
        self.assertTrue(inspect(self.path)['replay_ready'])

    def test_invalid_body_cannot_drive_a_reference(self):
        self.body['valid']='0';self.write()
        with self.assertRaisesRegex(ValueError,'invalid input'):inspect(self.path)

    def test_stale_body_is_rejected(self):
        self.body['time_ns']='200000101';self.write()
        with self.assertRaisesRegex(ValueError,'invalid input'):inspect(self.path)

    def test_missing_joint_is_rejected(self):
        self.body['flags_6']='0';self.write()
        with self.assertRaisesRegex(ValueError,'valid skeleton'):inspect(self.path)

    def test_mismatched_command_is_rejected(self):
        self.command['step']='10';self.write()
        with self.assertRaisesRegex(ValueError,'Mimic stream'):inspect(self.path)

    def test_missing_body_row_is_rejected(self):
        self.fixture.write('body.csv',BODY_FIELDS,[])
        with self.assertRaisesRegex(ValueError,'Body stream'):inspect(self.path)
