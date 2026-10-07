"""Add a backwards-compatible server timestamp to the camera ping response.

Usage: python scripts/enable_camera_clock.py /path/to/skeleton_server.py
Restart that camera server afterwards. Existing pong clients stay compatible.
"""
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = 'ws.send(json.dumps({"type": "pong", "t": value.get("t")}))'
new = 'ws.send(json.dumps({"type": "pong", "t": value.get("t"), "server_t": time.time_ns() // 1_000_000}))'
source = path.read_text()
if new in source:
    print('Server clock response already enabled')
elif source.count(old) == 1:
    path.write_text(source.replace(old, new))
    print('Server clock response enabled; restart the camera server')
else:
    raise SystemExit('Unknown server layout; source was not modified')
