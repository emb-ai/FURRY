"""Aggregate all final-window checkpoints, with paired survival-prefix errors."""
import argparse,json
from pathlib import Path
import numpy as np

METRICS=('root_xy_error_m','body_aligned_error_m','wrist_world_error_m','contact_slip_m_s')
def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None
def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    baseline=json.loads((a.directory/'twist2.json').read_text())
    bt=json.loads((a.directory/'twist2.traces.json').read_text())
    candidates=[];expected=[a.directory/f'checkpoint-{i:05d}.json' for i in range(7100,8001,100)]
    for path in expected:
        r=json.loads(path.read_text());ct=json.loads(path.with_suffix('.traces.json').read_text())
        for key in ('model_sha256','control_hz','physics_hz','horizon_steps','seeds','initialization'):
            if r[key]!=baseline[key]:raise ValueError('Mismatched protocol: '+key)
        if r['status']!='completed' or baseline['status']!='completed':raise ValueError('Incomplete evaluation')
        btrials={(x['clip'],x['seed']):x for x in baseline['trials']}
        if {(x['clip'],x['seed']) for x in r['trials']}!=set(btrials):raise ValueError('Different trial cohorts')
        paired=[]
        for t in r['trials']:
            b=btrials[(t['clip'],t['seed'])]
            if t['reference_sha256']!=b['reference_sha256']:raise ValueError('Different references')
            key=f"{t['clip']}__seed{t['seed']}";n=min(len(bt[key]),len(ct[key]));record={'seed':t['seed'],'steps':n}
            for metric in METRICS:
                record[metric]={'baseline':mean([v[metric] for v in bt[key][:n]]),'candidate':mean([v[metric] for v in ct[key][:n]])}
            paired.append(record)
        candidates.append({'checkpoint':path.stem,'completion_fraction':r['completion_fraction'],'falls':r['falls'],
                           'tracking_failures':sum(not x['completed'] for x in r['trials']),
                           'metrics':{k:mean([x[k] for x in r['trials']]) for k in METRICS},'paired_prefix':paired})
    out={'status':'completed','checkpoint_count':10,'seeds_per_checkpoint':5,'selection':'all checkpoints7100–8000, no best-checkpoint selection',
         'baseline':{'completion_fraction':baseline['completion_fraction'],'falls':baseline['falls'],
                     'tracking_failures':sum(not x['completed'] for x in baseline['trials']),
                     'metrics':{k:mean([x[k] for x in baseline['trials']]) for k in METRICS}},
         'candidate':{'completion_fraction':mean([x['completion_fraction'] for x in candidates]),'falls_total_50_trials':sum(x['falls'] for x in candidates),
                      'tracking_failures_total_50_trials':sum(x['tracking_failures'] for x in candidates),
                      'metrics':{k:mean([x['metrics'][k] for x in candidates]) for k in METRICS}},'checkpoints':candidates,
         'paired_prefix_metrics':{k:{side:mean([mean([p[k][side] for p in x['paired_prefix']]) for x in candidates]) for side in ('baseline','candidate')} for k in METRICS},
         'interpretation':'per-reference skill execution on a seen reference; controller-stack comparison, no generalization or isolated architecture claim'}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
