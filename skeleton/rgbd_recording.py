"""Bounded raw aligned RGB-D recorder, independent of detector and wire format."""
import json
from pathlib import Path
import queue
import threading
import time
import cv2
import numpy as np

# Depth is stored as integer millimetres (the D435 z16 unit at its default
# 0.001 scale) and color as JPEG: float32 depth + raw BGR through zlib took
# ~4-5 MB per 720p frame, more than a minute of capture leaves free on disk.
DEPTH_UNIT_M=0.001
JPEG_QUALITY=95


def load_frame(path):
    """Frame dict with color (BGR uint8) and depth_m (float32 metres); reads old and new files."""
    with np.load(path,allow_pickle=False) as f:
        frame={k:f[k] for k in f.files}
    if 'color_jpeg' in frame:
        frame['color']=cv2.imdecode(frame.pop('color_jpeg'),cv2.IMREAD_COLOR)
    if 'depth_mm' in frame:
        frame['depth_m']=frame.pop('depth_mm').astype(np.float32)*float(frame.get('depth_unit_m',DEPTH_UNIT_M))
    return frame


class RGBDRecorder:
    def __init__(self, directory, seconds=60, max_hz=15, delay=0, session_id=None):
        if seconds<=0 or max_hz<=0:raise ValueError('Recording duration/rate must be positive')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.seconds,self.period,self.delay,self.session_id=seconds,1/max_hz,delay,session_id
        self.armed=None
        self.started=None;self.last=-1e30;self.finished=False
        self.frames=0;self.dropped=0;self.error=None
        self.queue=queue.Queue(maxsize=4)
        self.thread=threading.Thread(target=self._write,daemon=True)
        self.thread.start()
        print(f'RGB-D recording armed: {self.directory} ({seconds}s, <= {max_hz} Hz, starts {delay}s after the first submitted frame)',flush=True)

    def submit(self, color, depth_m, intrinsics, epoch_ms, sequence, metadata=None):
        now=time.monotonic()
        if self.finished:return
        if self.armed is None:self.armed=now
        # Time to put the headset on and start the Quest episode first.
        if now-self.armed<self.delay:return
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
                ok,jpeg=cv2.imencode('.jpg',color,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
                if not ok:raise RuntimeError('JPEG encoding failed')
                depth_mm=np.clip(np.round(depth/DEPTH_UNIT_M),0,65535).astype(np.uint16)
                with partial.open('wb') as f:
                    np.savez_compressed(f,color_jpeg=jpeg.ravel(),depth_mm=depth_mm,depth_unit_m=np.float64(DEPTH_UNIT_M),
                                        intrinsics=intr,epoch_ms=np.int64(t),sequence=np.int64(seq),**metadata)
                partial.replace(dest);self.frames+=1
        except Exception as exc:
            self.error=str(exc);self.finished=True
            print('RGB-D RECORDING ERROR: '+self.error,flush=True)
        finally:
            report={'schema_version':1,'frames':self.frames,'dropped_queue':self.dropped,
                    'error':self.error,'complete':self.error is None,
                    'format':'color_jpeg BGR q95; depth_mm uint16 unfiltered, aligned to color; pinhole [fx,fy,cx,cy]; read with load_frame',
                    'requested_seconds':self.seconds,'max_hz':1/self.period,'session_id':self.session_id}
            try:(self.directory/'manifest.json').write_text(json.dumps(report,indent=2))
            except OSError as exc:print('RGB-D MANIFEST ERROR: '+str(exc),flush=True)
            print(f'RGB-D REC STOP: {self.frames} frames, {self.dropped} dropped; {self.directory}',flush=True)

    def close(self):
        self.finished=True
        self.thread.join()
