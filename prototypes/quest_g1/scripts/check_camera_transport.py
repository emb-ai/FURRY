"""Test the native WebSocket client against a synthetic local Python server.

Needs scripts/fetch_camera_transport.sh, the pinned OpenXR source and
websockets==15.0.1 in the test environment. No participant data is used.
"""
from pathlib import Path
import json
import os
import subprocess
import threading
import time
import mujoco
from websockets.sync.server import serve
ROOT = Path(__file__).resolve().parents[1]
build = ROOT/'android/build/camera-transport-host'
subprocess.run(['cmake', '-S', str(ROOT/'vendor/IXWebSocket'), '-B', str(build),
    '-DUSE_TLS=OFF', '-DUSE_ZLIB=OFF', '-DBUILD_SHARED_LIBS=OFF',
    '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'], check=True, stdout=subprocess.DEVNULL)
subprocess.run(['cmake', '--build', str(build), '-j', '4'], check=True, stdout=subprocess.DEVNULL)
jsonroot = ROOT/'vendor/OpenXR-SDK-Source/src/external/jsoncpp'
binary = build/'check-camera-stream'
subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17', '-pthread',
    '-I'+str(ROOT/'android/native'), '-I'+str(Path(mujoco.__file__).parent/'include'),
    '-I'+str(ROOT/'vendor/IXWebSocket'), '-I'+str(jsonroot/'include'),
    str(ROOT/'tests/check_camera_stream.cpp'), str(ROOT/'android/native/camera_stream.cpp'),
    *[str(jsonroot/'src/lib_json'/f) for f in ['json_reader.cpp', 'json_value.cpp', 'json_writer.cpp']],
    str(build/'libixwebsocket.a'), '-o', str(binary)], check=True)
uplink = threading.Event()
errors = []
def handler(ws):
    try:
        joint = {'p': [0, 0, 0], 'conf': 0, 'src': 'missing'}
        value = {'t': time.time_ns()/1e6, 'seq': 2, 'frame': 'camera',
            'pelvis': joint, 'joints': {k: joint for k in
            ['lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle', 'nose', 'lshoulder', 'rshoulder']}}
        ws.send(json.dumps(value))
        ws.send('malformed')
        value['seq'] = 1
        ws.send(json.dumps(value))
        data = json.loads(ws.recv(timeout=3))
        assert all(k in data for k in ['t', 'hmd', 'ctrl_l', 'ctrl_r'])
        uplink.set()
        for _ in ws:
            pass
    except Exception as error:
        errors.append(error)
with serve(handler, '127.0.0.1', 0, compression=None) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.socket.getsockname()[1]
    try:
        subprocess.run([str(binary), f'ws://127.0.0.1:{port}'], check=True, timeout=10)
        assert uplink.wait(2), 'Missing headset uplink'
        assert not errors, errors
    finally:
        server.shutdown()
        thread.join(timeout=3)
