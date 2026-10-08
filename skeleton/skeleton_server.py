"""Bounded, latest-frame WebSocket transport. Camera axes until calibrated."""
import json
import math
import threading
import time

from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

JOINTS = {"lhip": 11, "lknee": 13, "lankle": 15, "rhip": 12,
          "rknee": 14, "rankle": 16, "nose": 0, "lshoulder": 5,
          "rshoulder": 6}


def packet(points, confidence, sources, timestamp, seq, fps, backend,
           frame="camera", pelvis=None):
    def joint(i):
        p = points.get(i)
        if p is None or not all(math.isfinite(float(x)) for x in p):
            return {"p": [0., 0., 0.], "conf": 0., "src": "missing"}
        return {"p": [float(x) for x in p], "conf": float(confidence.get(i, 0)),
                "src": sources.get(i, "window")}
    if pelvis is None:
        pelvis = {"p": [0., 0., 0.], "conf": 0.}
        left, right = joint(11), joint(12)
        if left["conf"] > 0 and right["conf"] > 0:
            pelvis = {"p": [(a + b) / 2 for a, b in zip(left["p"], right["p"])],
                      "conf": min(left["conf"], right["conf"])}
    else:
        pelvis = {"p": [float(x) for x in pelvis["p"]], "conf": float(pelvis.get("conf", 0))}
    return {"t": timestamp, "seq": seq, "fps": fps, "backend": backend,
            "frame": frame, "pelvis": pelvis,
            "joints": {name: joint(i) for name, i in JOINTS.items()}}


class SkeletonServer:
    def __init__(self, host="0.0.0.0", port=8765, log=None):
        self.condition = threading.Condition()
        self.latest = None
        self.version = 0
        self.stopped = False
        self.log_lock = threading.Lock()
        self.log = open(log, "a", encoding="utf-8") if log else None
        self.uplink = None
        self.uplink_at = 0.0
        self.server = serve(self.handle, host, port, max_size=16384,
                            max_queue=4, compression=None)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"WebSocket: ws://{host}:{port} (frame=camera)", flush=True)

    def record(self, direction, payload):
        if self.log:
            with self.log_lock:
                self.log.write(json.dumps({"direction": direction,
                    "received_t": time.time_ns() // 1_000_000,
                    "payload": payload}, allow_nan=False) + "\n")
                self.log.flush()

    def latest_uplink(self, max_age_s=0.15):
        with self.condition:
            if self.uplink is None or time.monotonic() - self.uplink_at > max_age_s:
                return None
            return self.uplink

    def publish(self, payload):
        encoded = json.dumps(payload, allow_nan=False)
        self.record("downlink", payload)
        with self.condition:
            self.latest = (encoded, time.monotonic())
            self.version += 1
            self.condition.notify_all()

    def handle(self, ws):
        done = threading.Event()
        def receive():
            try:
                for message in ws:
                    try:
                        value = json.loads(message)
                        if not isinstance(value, dict):
                            continue
                        if value.get("type") == "ping":
                            ws.send(json.dumps({"type": "pong", "t": value.get("t"), "server_t": time.time_ns() // 1_000_000}))
                        elif all(k in value for k in ("t", "hmd", "ctrl_l", "ctrl_r")):
                            self.record("uplink", value)
                            with self.condition:
                                self.uplink = value
                                self.uplink_at = time.monotonic()
                    except (ValueError, TypeError):
                        continue
            except ConnectionClosed:
                pass
            finally:
                done.set()
                with self.condition:
                    self.condition.notify_all()
        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        seen = 0
        try:
            while not done.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or done.is_set()
                                            or self.version != seen, timeout=.5)
                    if self.stopped or done.is_set():
                        break
                    if self.version == seen:
                        continue
                    seen = self.version
                    encoded, created = self.latest
                if time.monotonic() - created < .15:
                    ws.send(encoded)
        except ConnectionClosed:
            pass
        finally:
            ws.close()
            reader.join(timeout=1)

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.server.shutdown()
        self.thread.join(timeout=3)
        if self.log:
            with self.log_lock:
                self.log.close()
