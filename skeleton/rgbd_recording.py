"""Bounded raw aligned RGB-D recorder, independent of detector and wire format."""
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np


class RGBDRecorder:
    def __init__(self, directory, seconds=60, max_hz=10):
        if seconds<=0 or max_hz<=0:raise ValueError('Recording duration/rate must be positive')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.seconds,self.period=seconds,1/max_hz
        self.started=None;self.last=-1e30;self.finished=False
        self.frames=0;self.dropped=0;self.error=None
        self.queue=queue.Queue(maxsize=4)
        self.thread=threading.Thread(target=self._write,daemon=True)
        self.thread.start()
        print(f'RGB-D recording armed: {self.directory} ({seconds}s, <= {max_hz} Hz)',flush=True)

    def submit(self, color, depth_m, intrinsics, epoch_ms, sequence, metadata=None):
        now=time.monotonic()
        if self.finished:return
        if self.started is None:
            self.started=now
            print('RGB-D REC START: stand still for 10 seconds, then move',flush=True)
        if now-self.started>=self.seconds:
            # Non-blocking stop. Writer drains existing frames and writes manifest.
            self.finished=True
            return
        if now-self.last<self.period or self.error:return
        self.last=now
        try:self.queue.put_nowait((color.copy(),depth_m.copy(),np.array(intrinsics),epoch_ms,sequence,dict(metadata or {})))
        except queue.Full:self.dropped+=1

    def _write(self):
        try:
            while not self.finished or not self.queue.empty():
                try:item=self.queue.get(timeout=.1)
                except queue.Empty:continue
                color,depth,intr,t,seq,metadata=item
                dest=self.directory/f'frame-{seq:09d}.npz';partial=dest.with_suffix('.partial')
                with partial.open('wb') as f:
                    np.savez_compressed(f,color=color,depth_m=depth,intrinsics=intr,
                                        epoch_ms=np.int64(t),sequence=np.int64(seq),**metadata)
                partial.replace(dest);self.frames+=1
        except Exception as exc:
            self.error=str(exc);self.finished=True
            print('RGB-D RECORDING ERROR: '+self.error,flush=True)
        finally:
            report={'schema_version':1,'frames':self.frames,'dropped_queue':self.dropped,
                    'error':self.error,'complete':self.error is None,
                    'format':'BGR uint8; unfiltered depth metres aligned to color; pinhole [fx,fy,cx,cy]',
                    'requested_seconds':self.seconds,'max_hz':1/self.period}
            try:(self.directory/'manifest.json').write_text(json.dumps(report,indent=2))
            except OSError as exc:print('RGB-D MANIFEST ERROR: '+str(exc),flush=True)
            print(f'RGB-D REC STOP: {self.frames} frames, {self.dropped} dropped; {self.directory}',flush=True)

    def close(self):
        self.finished=True
        self.thread.join()
