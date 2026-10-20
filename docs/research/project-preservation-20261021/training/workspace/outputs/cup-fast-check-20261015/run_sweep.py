from pathlib import Path
import json, os, subprocess, sys, time

O = Path(__file__).resolve().parent
OLD = O.parent / 'cup-pipeline-check-20261015'
status = {'pid': os.getpid(), 'started': time.time(), 'phase': 'sweep', 'completed': []}

def save():
    temp = O / 'status.tmp'
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(O / 'status.json')

def evaluate(h, d, seeds, phase):
    label = f'{phase}-d{d}'
    output = O / f'results-{label}-policy-h{h}-valid.json'
    if output.exists():
        return json.loads(output.read_text())
    status['active'] = {'execution': h, 'denoise': d, 'seeds': seeds}
    save()
    cmd = [sys.executable, str(O/'evaluate.py'), '--checkpoint', str(O/'latest-ema.pt'),
           '--execution-steps', str(h), '--denoise-steps', str(d), '--label', label,
           '--seeds', *map(str, seeds), '--limit', '90']
    with (O/f'{phase}-h{h}-d{d}.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = json.loads(output.read_text())
    status['completed'].append({'phase': phase, 'execution': h, 'denoise': d,
                                'successes': sum(r['success'] for r in rows), 'n': len(rows)})
    save()
    return rows

try:
    save()
    baseline = json.loads((OLD/'results-latest-policy-h8-valid.json').read_text())
    variants = [(8, 10, baseline)]
    for h, d in [(4, 10), (16, 10), (8, 20)]:
        variants.append((h, d, evaluate(h, d, [0], 'screen')))
    # Rank solely by complete task success; prefer established baseline on a tie.
    winner = max(variants, key=lambda v: sum(r['success'] for r in v[2]))
    status['phase'] = 'confirmation'
    status['selected'] = {'execution': winner[0], 'denoise': winner[1]}
    save()
    evaluate(8, 10, [1, 2], 'confirm')
    if winner[:2] != (8, 10):
        evaluate(*winner[:2], [1, 2], 'confirm')
    status['phase'] = 'completed'
    status.pop('active', None)
    save()
except Exception as exc:
    status.update(phase='failed', error=repr(exc))
    save()
    raise
