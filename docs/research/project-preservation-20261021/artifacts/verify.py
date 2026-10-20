#!/usr/bin/env python3
"""Check archived bytes; optionally recompute the eight-policy tracking metrics."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', action='store_true', help='also check trajectory metrics (requires numpy)')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / 'manifest.json').read_text())
    for entry in manifest['files']:
        path = base / entry['path']
        assert path.stat().st_size == entry['bytes'], str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], str(path)
    result = {'verified_originals': len(manifest['files']), 'bytes': sum(e['bytes'] for e in manifest['files'])}
    if args.metrics:
        import numpy as np
        folder = base / 'policy-comparison-20261016'
        with np.load(folder / 'references.npz', allow_pickle=False) as data:
            times, reference = data['time'], data['base']
        assert times.shape == (2001,) and np.allclose(times, np.arange(2001) / 100, atol=1e-12, rtol=0)
        expected = json.loads((folder / 'metrics.json').read_text())['results']
        def yaw(q):
            return np.arctan2(2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]),
                              1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2))
        errors = []
        for entry in expected:
            raw = np.loadtxt(folder / (entry['id'] + '.csv'), delimiter=',')
            assert raw.shape[0] == len(times) and np.isfinite(raw).all(), entry['id']
            assert np.allclose(raw[:, 0], times, atol=1e-8, rtol=0), entry['id']
            q = raw[:, 1:]
            xy = np.linalg.norm(q[1:, :2] - reference[1:, :2], axis=1).mean() * 100
            dy = yaw(q[1:]) - yaw(reference[1:])
            angle = np.rad2deg(np.abs(np.arctan2(np.sin(dy), np.cos(dy)))).mean()
            error = max(abs(float(xy) - entry['mean_xy_cm']), abs(float(angle) - entry['mean_yaw_deg']))
            assert error < 1e-10, (entry['id'], error)
            errors.append(error)
        published = json.loads((base.parents[2] / 'demo/2026-10-16/policy-metrics.json').read_text())
        assert published['results'] == expected
        result['metric_check'] = {'policies': len(errors), 'maximum_absolute_error': max(errors)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
