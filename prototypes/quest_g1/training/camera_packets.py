"""camera.jsonl -> plain-text packets for camera_commands (no JSON in C++).

One line per recorded packet, in observation order:
  observed_monotonic_ns observed_epoch_ms received_epoch_ms source_ms sequence camera_frame
  pelvis(x y z conf measured) 13 joints(x y z conf measured)
source_ms carries the clock offset the headset applied (CameraStream).
Optional joints that the packet omits are written unmeasured.
"""
import json
import sys

NAMES = ['lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle', 'nose',
         'lshoulder', 'rshoulder', 'lwrist', 'rwrist', 'lelbow', 'relbow']


def joint(v):
    if v is None:
        return [0, 0, 0, 0, 0]
    return [*v['p'], v['conf'], int(v.get('src', 'window') == 'window')]


def main(src, dst):
    with open(src) as f, open(dst, 'w') as out:
        for line in f:
            r = json.loads(line)
            p = r['payload']
            source = p['t'] + (r['clock_offset_ms'] if r['clock_synced'] else 0)
            row = [r['observed_monotonic_ns'], r['observed_epoch_ms'], r['received_epoch_ms'], source, p['seq'], int(p['frame'] == 'camera')]
            row += joint(p['pelvis'])
            for n in NAMES:
                row += joint(p['joints'].get(n))
            out.write(' '.join(repr(x) if isinstance(x, float) else str(x) for x in row) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
