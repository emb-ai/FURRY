"""Audit mirrored references without selecting data by the baseline's ability.

Admission is determined by mirror_dataset's data/FK/joint-limit checks. A policy
fall is a training difficulty, not evidence that a reference is corrupt.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from env import Motions
from evaluate import evaluate


def pair_diagnostic(original_fell, mirror_fell, joint_original, joint_mirror, slide_original, slide_mirror):
    flags=[]
    if original_fell:flags.append('original_baseline_failed')
    if mirror_fell:flags.append('mirrored_baseline_failed')
    if joint_mirror>joint_original*1.2+.01:flags.append('joint_rmse_regression')
    if slide_mirror>slide_original*1.5+.02:flags.append('slide_rate_regression')
    return {'included':True,'diagnostic_flags':flags,
            'paired_means':{'joint_rmse':{'original':joint_original,'mirror':joint_mirror},
                            'slide_rate':{'original':slide_original,'mirror':slide_mirror}}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--initial',required=True);p.add_argument('--source',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cached-rollouts',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    path=a.dataset/'manifest.json';manifest=json.loads(path.read_text());motions=Motions(a.dataset,'train')
    indices=[i for i,m in enumerate(motions.items) if m['meta'].get('augmentation')=='sagittal_reflection']
    mirrors=[c for c in manifest['clips'] if c.get('augmentation')=='sagittal_reflection']
    geometry={c['parent_id']:c for c in manifest['augmentation']['checks']}
    for c in mirrors:
        g=geometry[c['parent_id']]
        assert g['max_fk_position_error_m']<=2e-5 and g['max_fk_rotation_matrix_error']<=2e-5
        assert hashlib.sha256((a.dataset/c['file']).read_bytes()).hexdigest()==c['sha256']
    baseline=json.loads(a.baseline.read_text());before=np.load(a.baseline.with_suffix('.npz'))
    policy_hash=hashlib.sha256(Path(a.initial).read_bytes()).hexdigest()
    assert baseline['policy_sha256']==policy_hash
    if a.cached_rollouts:
        results=json.loads((a.cached_rollouts/'mirrored-baseline.json').read_text())
        provenance=json.loads((a.cached_rollouts/'source-provenance.json').read_text())
        assert results['policy_sha256']==policy_hash
        assert results['dataset_manifest_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ['env.py','evaluate.py','policy.py','rewards.py']:
            assert provenance['training_sha256'][name]==hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),name
        with np.load(a.cached_rollouts/'mirrored-baseline.npz') as data:results['_traces']={k:data[k].copy() for k in data.files}
    else:
        results=evaluate(a.assets,a.dataset,a.initial,a.source,'train',(0,),a.output/'mirrored-baseline.json',clip_indices=indices)
    assert {r['clip'] for r in results['trials']}=={c['id'] for c in mirrors}
    trials={t['id']:t for t in baseline['trials']};parents={c['id']:c['parent_id'] for c in mirrors};checks=[]
    for row in results['trials']:
        original=trials[parents[row['clip']]]
        b=before[original['id']];c=results['_traces'][row['trace_key']];n=min(len(b),len(c));start=min(30,n-1);means={}
        for key in ['joint_rmse','slide_rate']:
            means[key]=(float(b[start:n,baseline['columns'].index(key)].mean()),float(c[start:n,results['trace_columns'].index(key)].mean()))
        checks.append({'id':row['clip'],'parent_id':original['id'],**pair_diagnostic(original['fell'],row['fell'],*means['joint_rmse'],*means['slide_rate'])})
    report={'selection':'Data checks only: FK reflection, proper rotations, joint limits, involution and source/file checksums. Baseline falls/tracking are diagnostic only; no clips removed.',
            'included':len(checks),'tested':len(checks),'cached_rollouts':str(a.cached_rollouts) if a.cached_rollouts else None,'checks':checks}
    manifest['augmentation']['baseline_diagnostics']=report
    path.write_text(json.dumps(manifest,indent=2));(a.output/'mirror-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'phase':'mirror_audit','included':len(checks),'tested':len(checks),'excluded_for_baseline_performance':0}),flush=True)

if __name__=='__main__':main()
