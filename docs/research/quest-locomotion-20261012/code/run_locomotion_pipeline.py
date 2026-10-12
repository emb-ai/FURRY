"""One original-simulator finetune with independent MuJoCo baseline/checkpoints.

Runs on one physical GPU through an isolated uv environment. No automatic promotion or headset deployment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--resume',action='store_true');p.add_argument('--comparison-baseline',type=Path);p.add_argument('--save-every',type=int,default=100)
p.add_argument('--lower-body-coefficient',type=float,default=1.);p.add_argument('--yaw-airtime',action='store_true')
p.add_argument('--fast-critic',action='store_true')
p.add_argument('--nonlooping',action='store_true')
p.add_argument('--leg-faults',action='store_true')
p.add_argument('--seed',type=int,default=1)
p.add_argument('--control-hz',type=int,choices=(50,100),default=50)
p.add_argument('--previous-training',type=Path,help='Re-evaluate saved checkpoints at primary frequency after training')
p.add_argument('--init-checkpoint',type=Path);p.add_argument('--init-manifest',type=Path)
p.add_argument('--window-reward',action='store_true');p.add_argument('--sole-urdf',type=Path);p.add_argument('--eval-assets',type=Path);p.add_argument('--eval-dataset',type=Path)
p.add_argument('--run-id',required=True);p.add_argument('--distributed-size',type=int,choices=(1,),default=1)
a=p.parse_args()
if bool(a.init_checkpoint) != bool(a.init_manifest):p.error('--init-checkpoint and --init-manifest must be provided together')
root=a.root.resolve();out=root/'runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
old=root
py=root/'eval-env/bin/python'
# Use one explicit, immutable runtime/dataset variant for both baseline and candidates.
env=dict(os.environ,PYTHONPATH=str(root/'adapter')+os.pathsep+str(root),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
base_cmd=[str(py),str(Path(__file__).with_name('locomotion_eval.py')),'--assets',str(a.eval_assets or old/'assets'),'--dataset',str(a.eval_dataset or old/'dataset'),'--source',str(root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py')]
if a.window_reward:base_cmd+=['--window-metrics']
def evaluate(policy,dest,baseline=None,control_hz=None,split='validation',stage=None):
    cmd=base_cmd+['--policy',str(policy),'--output',str(dest),'--control-hz',str(control_hz or a.control_hz),'--split',split]
    if stage is not None:cmd+=['--stage',str(stage)]
    if baseline:cmd+=['--baseline',str(baseline)]
    cache=root/'locomotion-v1-preflight-baseline.json'
    if baseline is None and control_hz is None and split=='validation' and stage is None and a.comparison_baseline and policy.resolve()==a.comparison_baseline.resolve() and cache.exists():
        import shutil
        shutil.copy2(cache,dest);shutil.copy2(cache.with_suffix('.npz'),dest.with_suffix('.npz'))
        dest.with_suffix('.log').write_text('Reused complete baseline evaluated on this host, same policy/assets/holdout/50Hz/seeds0,1. Source: '+str(cache)+'\n')
        return
    with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if a.eval_dataset:
    from checkpoint_lineage import audit_evaluation
    evaluation_audit=audit_evaluation(json.loads((root/'dataset/manifest.json').read_text()),a.eval_dataset)
    (out/'evaluation-split-audit.json').write_text(json.dumps(evaluation_audit,indent=2))
(out/'pipeline-config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
if a.init_checkpoint:
    import torch
    from checkpoint_lineage import audit
    ancestor=torch.load(a.init_checkpoint,map_location='cpu',weights_only=True)
    lineage=audit(ancestor,a.init_manifest,json.loads((root/'dataset/manifest.json').read_text()))
    (out/'initialization-audit.json').write_text(json.dumps(lineage,indent=2))
    del ancestor
baseline=out/'baseline.json'
evaluate(a.comparison_baseline or a.init_checkpoint or root/'TWIST2/assets/ckpts/twist2_1017_20k.onnx',baseline)
train_out=out/'training'
cmd=[sys.executable,str(Path(__file__).with_name('locomotion_finetune.py')),'--root',str(root),'--output',str(train_out),'--headless','--num_envs',str(a.num_envs),'--sim_device','cuda:0','--rl_device','cuda:0','--device','cuda:0','--num_threads','1','--iterations',str(a.iterations)]
cmd+=['--lower-body-coefficient',str(a.lower_body_coefficient)]
if a.yaw_airtime:cmd+=['--yaw-airtime']
cmd+=['--seed',str(a.seed),'--save-every',str(a.save_every)]
if a.resume:cmd+=['--resume']
if a.nonlooping:cmd+=['--nonlooping']
if a.leg_faults:cmd+=['--leg-faults']
if a.init_checkpoint:cmd+=['--init-checkpoint',str(a.init_checkpoint),'--init-manifest',str(a.init_manifest)]
if a.fast_critic:cmd+=['--fast-critic']
if a.window_reward:cmd+=['--window-reward']
if a.sole_urdf:cmd+=['--sole-urdf',str(a.sole_urdf)]
results=[];done=set()
def plot_curves():
    if not (train_out/'metrics.jsonl').exists():return
    cmd=[sys.executable,str(root/'adapter/plot_single_gpu_losses.py'),'--control' if not a.leg_faults else '--faults',str(train_out/'metrics.jsonl'),'--output',str(out/'loss-curves.png')]
    if a.resume:cmd+=['--history-control' if not a.leg_faults else '--history-faults',str(a.init_checkpoint.parent/'plot-history.jsonl')]
    with (out/'plot.log').open('a') as plot_log:subprocess.run(cmd,stdout=plot_log,stderr=subprocess.STDOUT,check=True)
with (out/'training.log').open('w') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
        while True:
            for checkpoint in sorted(train_out.glob('checkpoint_*.pt')):
                if checkpoint.name in done:continue
                dest=out/(checkpoint.stem+'.json')
                plot_curves()
                evaluate(checkpoint,dest,baseline)
                verdict=json.loads(dest.with_suffix('.comparison.json').read_text())
                from locomotion_verdict import summarize
                (dest.with_suffix('.gait-verdict.json')).write_text(json.dumps(summarize(verdict),indent=2))
                results.append({'checkpoint':checkpoint.name,**verdict});done.add(checkpoint.name)
                (out/'runtime-results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({'checkpoint':checkpoint.name,'accepted':verdict['accepted'],'reasons':verdict['reasons']}),flush=True)
            if proc.poll() is not None:
                if proc.returncode:raise RuntimeError(f'Trainer exited {proc.returncode}; inspect training.log')
                if any(c.name not in done for c in train_out.glob('checkpoint_*.pt')):continue
                break
            time.sleep(20)
        if proc.returncode:raise RuntimeError(f'Trainer exited {proc.returncode}; inspect training.log')
    finally:
        if proc.poll() is None:
            (train_out/'STOP').touch()
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.terminate()
plot_curves()
# Keep compatibility and train diagnostics separate from the primary holdout.
parent=a.comparison_baseline or a.init_checkpoint or root/'TWIST2/assets/ckpts/twist2_1017_20k.onnx'
final=train_out/'final.pt'
compat_hz=100 if a.control_hz==50 else 50
compat=out/f'compatibility-{compat_hz}hz';compat.mkdir()
evaluate(parent,compat/'baseline.json',control_hz=compat_hz)
evaluate(final,compat/'final.json',compat/'baseline.json',control_hz=compat_hz)
backward=out/'train-stage05';backward.mkdir()
evaluate(parent,backward/'baseline.json',split='train',stage=5)
evaluate(final,backward/'final.json',backward/'baseline.json',split='train',stage=5)
if a.previous_training:
    previous=out/'previous-run';previous.mkdir()
    for checkpoint in sorted(a.previous_training.glob('checkpoint_*.pt')):
        evaluate(checkpoint,previous/(checkpoint.stem+'.json'),baseline)
    (previous/'provenance.json').write_text(json.dumps({'training_directory':str(a.previous_training),'purpose':'Retrospective evaluation only; not initialization or validation-to-train transfer'},indent=2))
if a.nonlooping:
    # Identical clean/delayed/frozen input probes for control and treatment.
    # Author probe contains existing TRAIN only, never called a validation set.
    stress=out/'fault-sensitivity';stress.mkdir()
    for dataset,split,limit in [(root/'author-probe','train',4),(a.eval_dataset,'validation',3)]:
        for label,policy in [('baseline',parent),('final',final)]:
            dest=stress/(dataset.name+'-'+label+'.json')
            cmd=[str(py),str(root/'adapter/conflict_probe.py'),'--assets',str(a.eval_assets),'--dataset',str(dataset),'--policy',str(policy),'--source',str(root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py'),'--split',split,'--limit',str(limit),'--seconds','12','--output',str(dest)]
            with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
status={'phase':'pipeline_complete','results':results,'training_output':str(train_out),'distributed_size':a.distributed_size,'run_id':a.run_id,'deployment':False,'validation':'12 held-out Quest clips, seeds 0/1; explicit eval-dataset and eval-assets used identically for baseline/checkpoints. Previously inspected validation, not an independent test.','selection':'Existing baseline acceptance remains diagnostic; new gait/progress metrics are reported on matched prefixes without new arbitrary gates. No early stop or automatic deployment.','primary_control_hz':a.control_hz,'compatibility_control_hz':compat_hz,'train_diagnostic':'stage 5 stays train; separate reports' }
(out/'result.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
