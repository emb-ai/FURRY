import json
import tempfile
import time
import unittest
from pathlib import Path

from websockets.sync.client import connect
from skeleton_server import SkeletonServer, packet


class TransportTest(unittest.TestCase):
    def test_missing_and_camera_coordinates(self):
        p = packet({11: [1, 2, 3], 12: [3, 2, 3], 9: [99, 99, 99]},
                   {11: .9, 12: .8}, {11: "line", 12: "window"}, 123, 1, 30, "test")
        self.assertEqual(p["frame"], "camera")
        self.assertEqual(p["pelvis"]["p"], [2, 2, 3])
        self.assertEqual(p["joints"]["lknee"]["conf"], 0)
        self.assertEqual(p["joints"]["lhip"]["src"], "line")
        self.assertEqual(len(p["joints"]), 9)
        json.dumps(p, allow_nan=False)

    def test_live_transport_uplink_and_no_stale_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "frames.jsonl"
            server = SkeletonServer("127.0.0.1", 0, str(log))
            port = server.server.socket.getsockname()[1]
            try:
                server.publish({"seq": 1})
                time.sleep(.2)
                with connect(f"ws://127.0.0.1:{port}") as client:
                    with self.assertRaises(TimeoutError):
                        client.recv(timeout=.1)
                    server.publish({"seq": 2})
                    self.assertEqual(json.loads(client.recv(timeout=2)), {"seq": 2})
                    client.send(json.dumps({"type": "ping", "t": 42}))
                    self.assertEqual(json.loads(client.recv(timeout=2))["t"], 42)
                    client.send(json.dumps({"t": 1, "hmd": {}, "ctrl_l": {}, "ctrl_r": {}}))
                    deadline = time.monotonic() + 2
                    while 'uplink' not in log.read_text() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertIn('uplink', log.read_text())
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
