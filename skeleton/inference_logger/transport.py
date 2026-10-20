"""Compatible skeleton downlink, lossless Quest uplink, four-clock probes."""
import json
import threading
import time
import uuid

from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed


class QuestServer:
    def __init__(self, session, host='0.0.0.0', port=8765):
        self.session = session
        self.lock = threading.Lock()
        self.clients = set()
        self.clients_done = threading.Condition(self.lock)
        self.latest = None
        self.stop = threading.Event()
        self.server = serve(self.handle, host, port, max_size=1024*1024,
                            max_queue=4, compression=None)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def publish(self, payload):
        with self.lock:
            self.latest = (time.monotonic_ns(), dict(payload))
        self.session.emit('camera_tx', {'status': 'published', 'payload': payload})

    def handle(self, ws):
        with self.clients_done:
            if self.stop.is_set():
                return
            self.clients.add(ws)
        segment = str(uuid.uuid4())
        pending = {}
        last_probe = 0
        seen = None
        self.session.emit('events', {'event': 'quest_connected', 'clock_segment': segment})
        try:
            while not self.stop.is_set():
                now = time.monotonic_ns()
                if now - last_probe >= 1_000_000_000:
                    request = str(uuid.uuid4())
                    pending = {k: v for k, v in pending.items() if now-v < 5_000_000_000}
                    pending[request] = now
                    ws.send(json.dumps({'type': 'sync_probe', 'request_id': request,
                                        'session_id': self.session.id, 't1_ns': str(now)}))
                    last_probe = now
                with self.lock:
                    latest = self.latest
                if latest and latest[1]['seq'] != seen:
                    created, payload = latest
                    seen = payload['seq']
                    if now-created < 150_000_000:
                        ws.send(json.dumps(payload, allow_nan=False))
                        self.session.emit('camera_tx', {'status': 'sent', 'clock_segment': segment,
                                                       'packet_seq': seen, 'frame_id': payload['frame_id']})
                try:
                    raw = ws.recv(timeout=.01)
                except TimeoutError:
                    continue
                received = time.monotonic_ns()
                self.session.emit('quest_rx', {'clock_segment': segment,
                    'receive_monotonic_ns': str(received),
                    'raw': raw if isinstance(raw, str) else raw.hex(),
                    'binary': isinstance(raw, bytes)})
                if not isinstance(raw, str):
                    continue
                try:
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        continue
                    if value.get('type') == 'ping':
                        ws.send(json.dumps({'type': 'pong', 't': value.get('t'),
                                            'server_t': time.time_ns() // 1_000_000}))
                    elif value.get('type') == 'sync_reply':
                        t1 = pending.pop(value.get('request_id'), None)
                        if t1 is None:
                            continue
                        t2, t3 = int(value['t2_ns']), int(value['t3_ns'])
                        if not (0 <= t2 <= t3 and 0 <= received-t1-(t3-t2) <= 500_000_000):
                            continue
                        self.session.emit('sync', {'clock_segment': segment,
                            'request_id': value['request_id'], 't1_ns': str(t1),
                            't2_ns': str(t2), 't3_ns': str(t3), 't4_ns': str(received),
                            'quest_clock': 'std::chrono::steady_clock'})
                    elif all(k in value for k in ('hmd', 'ctrl_l', 'ctrl_r', 't')):
                        self.session.emit('quest', {'clock_segment': segment,
                            'receive_monotonic_ns': str(received), 'payload': value})
                except (ValueError, TypeError, KeyError):
                    self.session.emit('events', {'event': 'invalid_quest_message', 'clock_segment': segment})
        except ConnectionClosed:
            pass
        except Exception as error:
            self.session.error = f'Quest transport failed: {error!r}'
        finally:
            try:
                if not self.session.error:
                    self.session.emit('events', {'event': 'quest_disconnected', 'clock_segment': segment})
            finally:
                with self.clients_done:
                    self.clients.discard(ws)
                    self.clients_done.notify_all()

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.thread.join()
        with self.clients_done:
            clients = list(self.clients)
        for ws in clients:
            ws.close()
        with self.clients_done:
            if not self.clients_done.wait_for(lambda: not self.clients, timeout=10):
                self.session.error = 'Quest handlers did not finish at shutdown'
