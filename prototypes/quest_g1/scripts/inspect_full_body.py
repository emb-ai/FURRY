"""Validate the raw 84-joint stream; optionally export long-form CSV.

Usage: python scripts/inspect_full_body.py EPISODE [--csv output.csv]
Poses are unmodified OpenXR STAGE metres and wxyz quaternions. Flags are
preserved; a numerical pose with invalid flags is not a tracked observation.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

HEADER = struct.Struct('<8sIII')
FRAME = struct.Struct('<QqqqIfBBBBI')
JOINT = struct.Struct('<Q7f')
BIND = struct.Struct('<QII')
REST = struct.Struct('<i7f')


def records(path, magic, size):
    with Path(path).open('rb') as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size or HEADER.unpack(header) != (magic, 1, size, 84):
            raise ValueError(f'Invalid full-body header: {path}')
        while raw := stream.read(size):
            if len(raw) != size:
                raise ValueError(f'Truncated full-body record: {path}')
            yield raw


def inspect(folder, csv_path=None):
    folder = Path(folder)
    skeletons = []
    for raw in records(folder/'body_skeleton.bin', b'G1SKEL84', 2704):
        seq, version, reserved = BIND.unpack_from(raw)
        joints = [REST.unpack_from(raw, 16+j*32) for j in range(84)]
        if reserved or any(j[0] < -1 or j[0] >= 84 for j in joints):
            raise ValueError('Invalid bind topology')
        if skeletons and seq <= skeletons[-1]['sequence']:
            raise ValueError('Unordered bind skeletons')
        skeletons.append(dict(sequence=seq, version=version, joints=joints))
    count = available = active = 0
    valid = [0]*84
    previous = -1
    bind_index = -1
    out = Path(csv_path).open('w', newline='') if csv_path else None
    try:
        writer = csv.writer(out) if out else None
        if writer:
            writer.writerow(['sequence','received_ns','xr_time_ns','body_time_ns','skeleton_version','confidence','available','active','joint','flags','x','y','z','qw','qx','qy','qz'])
        for raw in records(folder/'body_full.bin', b'G1BODY84', 3072):
            seq, received, xr_time, body_time, version, confidence, present, enabled, has_bind, supported, n = FRAME.unpack_from(raw)
            if seq <= previous or present not in (0,1) or enabled not in (0,1) or has_bind not in (0,1) or supported not in (0,1) or n != (84 if present else 0):
                raise ValueError('Invalid full-body frame metadata')
            previous = seq
            while bind_index+1 < len(skeletons) and skeletons[bind_index+1]['sequence'] <= seq:
                bind_index += 1
            if has_bind and (bind_index < 0 or skeletons[bind_index]['version'] != version):
                raise ValueError('Missing matching bind skeleton')
            count += 1
            available += present
            active += present and enabled
            for j in range(84):
                values = JOINT.unpack_from(raw, 48+j*36)
                valid[j] += bool(present and enabled and values[0]&3 == 3)
                if writer:
                    writer.writerow([seq,received,xr_time,body_time,version,confidence,present,enabled,j,*values])
    finally:
        if out:
            out.close()
    return dict(schema=1, frames=count, available=available, active=active,
                skeletons=len(skeletons), valid_position_orientation_frames=valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.episode, args.csv), indent=2))
