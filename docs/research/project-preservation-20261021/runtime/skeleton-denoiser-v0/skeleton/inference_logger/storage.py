"""Strict, append-only JSONL with bounded asynchronous disk I/O."""
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(dumps(value) + '\n', encoding='utf-8')
    temp.replace(path)


def read_jsonl(path):
    """An incomplete tail is an error; raw files are never silently repaired."""
    with Path(path).open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                yield json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except ValueError as e:
                raise ValueError(f'{path}:{line_number}: corrupt/incomplete JSONL') from e


class Session:
    def __init__(self, directory, metadata=None, capacity=2048):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.id = str(uuid.uuid4())
        self.queue = queue.Queue(capacity)
        self.error = None
        self.closed = False
        self.counts = {}
        save_json(self.path / 'session.json', {
            'schema_version': 1, 'session_id': self.id,
            'created_epoch_ns': str(time.time_ns()),
            'clock': 'host_monotonic_ns', 'timestamp_encoding': 'decimal strings',
            'units': {'position': 'm', 'time': 'ns', 'image': 'px'},
            'quaternion_order': 'wxyz', 'metadata': metadata or {},
        })
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def emit(self, stream, value):
        if self.closed or self.error:
            raise RuntimeError(f'Recorder unavailable: {self.error or "closed"}')
        if not stream.replace('_', '').isalnum():
            raise ValueError('Invalid stream name')
        row = {'schema_version': 1, 'session_id': self.id,
               'host_monotonic_ns': str(time.monotonic_ns()), **value}
        encoded = dumps(row)
        try:
            self.queue.put_nowait((stream, encoded))
        except queue.Full as e:
            self.error = 'telemetry queue overflow'
            raise RuntimeError(self.error) from e

    def _write(self):
        files = {}
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                stream, encoded = item
                if stream not in files:
                    files[stream] = (self.path / (stream + '.jsonl')).open('x', encoding='utf-8')
                files[stream].write(encoded + '\n')
                files[stream].flush()
                self.counts[stream] = self.counts.get(stream, 0) + 1
        except Exception as e:
            self.error = repr(e)
        finally:
            for f in files.values():
                f.close()

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        while self.worker.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                pass
        self.worker.join()
        failure = error or self.error
        save_json(self.path / 'checksums.json', {
            p.relative_to(self.path).as_posix(): sha256(p)
            for p in self.path.rglob('*') if p.is_file() and p.name != 'checksums.json'
        })
        save_json(self.path / ('failed.json' if failure else 'complete.json'),
                  {'session_id': self.id, 'counts': self.counts, 'error': failure})
