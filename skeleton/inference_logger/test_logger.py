import json
from pathlib import Path
import tempfile
import time
import unittest

from inference_logger.storage import Session, read_jsonl, save_json
from inference_logger.offline import fit_clocks, mapped, validate, export, import_episode
from inference_logger.quest_patch import patched_source


def probes(offset=4_000_000_000, drift=1.00005, segment='a', origin=10**18):
    result = []
    for i in range(10):
        q = origin+i*1_000_000_000
        h = origin+offset+round(i*1_000_000_000*drift)
        result.append({'clock_segment': segment, 't1_ns': str(h-1_000_000),
                       't2_ns': str(q), 't3_ns': str(q), 't4_ns': str(h+1_000_000)})
    return result


class LoggerTests(unittest.TestCase):
    def test_clock_precision_drift_segments_and_gaps(self):
        fitted = fit_clocks(probes()+probes(-9_000_000_000, .9999, 'b'))
        self.assertAlmostEqual(fitted['a']['drift_ppm'], 50, places=5)
        self.assertEqual(mapped(10**18+2_000_000_000, fitted['a']), 10**18+6_000_100_000)
        self.assertEqual(mapped(10**18, fitted['b']), 10**18-9_000_000_000)
        self.assertIsNone(mapped(10**18+20_000_000_000, fitted['a']))
        sparse = probes()[:3]+probes()[8:]
        self.assertIsNone(mapped(10**18+5_000_000_000, fit_clocks(sparse)['a'], 1_000_000_000))

    def test_bad_clock_rejected(self):
        values = probes(drift=1.1)
        self.assertFalse(fit_clocks(values)['a']['valid'])
        values[2]['t3_ns'] = str(int(values[2]['t2_ns'])-1)
        self.assertEqual(fit_clocks(values)['a']['samples'], 9)

    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(Path(tmp)/'session')
            s.emit('frames', {'frame_id': 'r:1', 'acquisition_monotonic_ns': str(10**18+1)})
            s.emit('skeleton', {'frame_id': 'r:1'})
            with self.assertRaises(ValueError):
                s.emit('bad', {'x': float('nan')})
            s.close()
            self.assertTrue(validate(s.path)['valid'])
            self.assertEqual(list(read_jsonl(s.path/'frames.jsonl'))[0]['acquisition_monotonic_ns'], str(10**18+1))
            with (s.path/'frames.jsonl').open('a') as f:
                f.write('{')
            self.assertFalse(validate(s.path)['valid'])
            with self.assertRaises(ValueError):
                list(read_jsonl(s.path/'frames.jsonl'))

    def test_failure_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(Path(tmp)/'s')
            s.close('disk full')
            self.assertFalse((s.path/'complete.json').exists())
            self.assertFalse(validate(s.path)['valid'])

    def test_export_links_and_no_false_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s = Session(root/'s', {'test_only': True})
            host = 10**18+6_000_100_000
            for fid, t in [('r:1', host), ('r:3', host+500_000_000)]:
                s.emit('frames', {'frame_id': fid, 'acquisition_monotonic_ns': str(t)})
                s.emit('skeleton', {'frame_id': fid, 'acquisition_monotonic_ns': str(t), 'joints': {}})
            for probe in probes():
                s.emit('sync', probe)
            s.emit('quest', {'clock_segment': 'a', 'payload': {'input_sequence': '7',
                'quest_monotonic_ns': str(10**18+2_000_000_000)}})
            s.close()
            episode = root/'episode-1'
            episode.mkdir()
            save_json(episode/'complete.json', {})
            (episode/'camera.jsonl').write_text(json.dumps({'input_sequence': 7,
                'payload': {'session_id': s.id, 'frame_id': 'r:1'}})+'\n')
            (episode/'frames.csv').write_text('sequence,step,qpos_0,ctrl_0\n7,1,0.1,0.2\n7,2,0.2,0.3\n8,3,0.3,0.4\n')
            import_episode(s.path, episode)
            report = export(s.path, root/'derived')
            self.assertEqual(report['paired_frames'], 1)
            self.assertEqual(report['episode_rows_with_camera_link'], 2)
            links = list(read_jsonl(root/'derived'/'episode_links.jsonl'))
            self.assertEqual(len(links), 3)
            self.assertFalse(links[-1]['runtime_link_valid'])
            self.assertEqual(links[0]['values']['ctrl_0'], '0.2')
            empty = Session(root/'empty')
            empty.close()
            with self.assertRaises(ValueError):
                export(empty.path, root/'bad')

    def test_websocket_roundtrip_and_reconnect(self):
        from inference_logger.transport import QuestServer
        from websockets.sync.client import connect
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(Path(tmp)/'s')
            server = QuestServer(s, '127.0.0.1', 0)
            try:
                for _ in range(2):
                    with connect(f'ws://127.0.0.1:{server.port}') as ws:
                        p = json.loads(ws.recv(timeout=2))
                        q = time.monotonic_ns()+123_000_000
                        ws.send(json.dumps({'type': 'sync_reply', 'request_id': p['request_id'],
                            't2_ns': str(q), 't3_ns': str(q)}))
                        ws.send(json.dumps({'t': 1, 'hmd': {}, 'ctrl_l': {}, 'ctrl_r': {},
                                            'input_sequence': '42', 'quest_monotonic_ns': str(q)}))
                        ws.send(json.dumps({'type': 'ping', 't': 2}))
                        self.assertEqual(json.loads(ws.recv(timeout=2))['type'], 'pong')
            finally:
                server.close()
                s.close()
            self.assertEqual(len(list(read_jsonl(s.path/'quest.jsonl'))), 2)
            sync = list(read_jsonl(s.path/'sync.jsonl'))
            self.assertEqual(len(sync), 2)
            self.assertNotEqual(sync[0]['clock_segment'], sync[1]['clock_segment'])

    def test_patch_guard(self):
        with self.assertRaises(ValueError):
            patched_source('wrong revision')


if __name__ == '__main__':
    unittest.main()
