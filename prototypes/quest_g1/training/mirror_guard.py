"""Admit mirrored TRAIN clips after deterministic baseline physics rollouts.

Does not inspect validation. Original references remain intact, even when the
baseline cannot track them; only optional synthetic copies are filtered.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from env import Motions
from evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--initial',required=True);p.add_argument('--source',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    path=a.dataset/'manifest.json';manifest=json.loads(path.read_text());motions=Motions(a.dataset,'train')
    indices=[i for i,m in enumerate(motions.items) if m['meta'].get('augmentation')=='sagittal_reflection']
    baseline=json.loads(a.baseline.read_text());before=np.load(a.baseline.with_suffix('.npz'))
    assert baseline['policy_sha256']==hashlib.sha256(Path(a.initial).read_bytes()).hexdigest()
    results=evaluate(a.assets,a.dataset,a.initial,a.source,'train',(0,),a.output/'mirrored-baseline.json',clip_indices=indices)
    trials={t['id']:t for t in baseline['trials']};parents={c['id']:c['parent_id'] for c in manifest['clips'] if c.get('augmentation')}
    checks=[];accepted=set()
    for row in results['trials']:
        original=trials[parents[row['clip']]];reasons=[]
        if original['fell']:reasons.append('original_baseline_failed')
        if row['fell']:reasons.append('mirrored_baseline_failed')
        b=before[original['id']];c=results['_traces'][row['trace_key']];n=min(len(b),len(c));start=min(30,n-1)
        means={}
        for key,tolerance,absolute in [('joint_rmse',.2,.01),('slide_rate',.5,.02)]:
            bm=float(b[start:n,baseline['columns'].index(key)].mean());cm=float(c[start:n,results['trace_columns'].index(key)].mean())
            means[key]={'original':bm,'mirror':cm}
            if cm>bm*(1+tolerance)+absolute:reasons.append(key+'_regression')
        if not reasons:accepted.add(row['clip'])
        checks.append({'id':row['clip'],'parent_id':original['id'],'accepted':not reasons,'reasons':reasons,'paired_means':means})
    report={'criteria':'both original and mirror survive; mirror joint RMSE <= original*1.2+0.01 rad; slide <= original*1.5+0.02; same seed0 original actor; no validation data',
            'accepted':len(accepted),'tested':len(checks),'checks':checks}
    manifest['clips']=[c for c in manifest['clips'] if not c.get('augmentation') or c['id'] in accepted]
    manifest['augmentation']['physics_guard']=report
    manifest['duration_seconds']={s:sum(c['seconds'] for c in manifest['clips'] if c['split']==s) for s in ['train','validation','replay']}
    # Originals and validation are never edited or relabeled.
    path.write_text(json.dumps(manifest,indent=2));(a.output/'mirror-guard.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'phase':'mirror_guard','accepted':len(accepted),'tested':len(checks)}),flush=True)

if __name__=='__main__':main()
