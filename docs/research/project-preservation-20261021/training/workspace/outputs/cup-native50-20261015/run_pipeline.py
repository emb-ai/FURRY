from pathlib import Path
import os,subprocess,json,time,sys
O=Path(__file__).resolve().parent;B=O.parent/'cup-fast-check-20261015';OLD=O.parent/'cup-homie-finetune-20261015';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-68d1fffd-2ea2-3f95-f100-38f39855bcdf',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',LP_NUM_THREADS='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');state={'pid':os.getpid(),'started':time.time(),'gpu':6,'steps':10000}
def status(**kwargs):
 state.update(kwargs);p=O/'pipeline-status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(O/'pipeline-status.json')
def run(cmd,name):
 with (O/(name+'.log')).open('w') as f:
  p=subprocess.Popen([str(x) for x in cmd],cwd=O,env=env,stdout=f,stderr=subprocess.STDOUT);status(child_pid=p.pid);rc=p.wait()
 if rc:raise RuntimeError(f'{name} exited {rc}')
try:
 status(phase='rendering');run([B/'env-dp/bin/python',O/'parallel_render.py'],'render')
 status(phase='training');run([B/'env-act/bin/python',O/'finetune_native50.py','--kind','act','--base',B,'--data',O/'cup-native50.hdf5','--source',OLD/'act-run/latest.pt','--out',O/'act-run','--steps','10000','--lr','1e-5','--save-every','5000'],'train')
 status(phase='exporting');run([B/'env-act/bin/python',O/'export_act.py','--source',O/'act-run/latest.pt','--out',O/'exports/act'],'export')
 status(phase='evaluating_native50');run([B/'env-dp/bin/python',O/'evaluate.py','--weights',O/'exports/act','--out',O/'evaluation-native50','--hz','50'],'evaluation-native50')
 status(phase='evaluating_old30_same_platform');run([B/'env-dp/bin/python',O/'evaluate.py','--weights',O/'exports/old-act-cuda','--out',O/'evaluation-old30','--hz','30'],'evaluation-old30')
 new=json.loads((O/'evaluation-native50/summary.json').read_text());old=json.loads((O/'evaluation-old30/summary.json').read_text());assert new['n']==old['n']==10 and not new['errors'] and not old['errors']
 (O/'comparison.json').write_text(json.dumps({'new_native50':new,'old30_same_platform':old,'caveats':['All59 demonstrations are training data;10 fixed random full resets are validation.','Neutral gaze and zero controller state; no demonstration prefix.','Same64-step chunk spans1.28s at50Hz vs2.13s at30Hz.','10000 additional updates; this compares a corrected-rate finetune to the previous checkpoint, not an equal-compute ablation.','Missing reset-to-task demonstrations remain absent.','Linux Mesa rendering differs from Mac by image MAE6.6–7.6/255 on6 probes; both evaluation arms use Linux. New training also uses Linux rendering, so this is not a frequency-only causal ablation.']},indent=2));status(phase='completed',results={'native50':new['successes'],'old30':old['successes'],'n':10},elapsed_s=time.time()-state['started'])
except Exception as exc:status(phase='failed',error=repr(exc));raise
