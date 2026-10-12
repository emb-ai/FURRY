"""Fingerprint every input affecting baseline diagnostics/dynamics."""
import hashlib
import json
from pathlib import Path


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contract(assets,dataset,source,control_hz,seeds,split,stage=None,limit=None,window_metrics=True):
    code=Path(__file__).resolve().parent;assets=Path(assets);dataset=Path(dataset)
    modules=['locomotion_eval.py','evaluate.py','env.py','rewards.py','policy.py',
        'gait_evaluation.py','sole_geometry.py','window_reward.py','clearance_evaluation.py',
        'evaluation_contract.py','g1_sim/controller.py']
    manifest=json.loads((dataset/'manifest.json').read_text())
    clips=[c for c in manifest['clips'] if c['split']==split and (stage is None or c['stage']==stage)]
    for c in clips:
        if digest(dataset/c['file'])!=c['sha256']:raise ValueError('Evaluation clip changed: '+c['id'])
    return dict(schema='clearance-v2-baseline-contract-1',policy_control_hz=control_hz,physics_hz=1000,
        seeds=list(seeds),split=split,stage=stage,limit=limit,window_metrics=window_metrics,
        dataset_manifest_sha256=digest(dataset/'manifest.json'),
        clips_sha256={c['id']:c['sha256'] for c in clips},
        asset_sha256={str(p.relative_to(assets)):digest(p) for p in sorted(assets.rglob('*')) if p.is_file()},
        evaluator_sha256={n:digest(code/n) for n in modules},actor_source_sha256=digest(source))


def baseline_matches(report,policy,expected):
    report=Path(report);sidecar=report.with_suffix('.fingerprint.json')
    if not report.exists() or not report.with_suffix('.npz').exists() or not sidecar.exists():return False
    observed=json.loads(sidecar.read_text())
    return observed==dict(contract=expected,policy_sha256=digest(policy))
