from pathlib import Path
import os,sys,json,time,subprocess
O=Path(__file__).resolve().parent;B=O.parent/'cup-fast-check-20261015';OLD=O.parent/'cup-homie-finetune-20261015';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-68d1fffd-2ea2-3f95-f100-38f39855bcdf',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',LP_NUM_THREADS='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');state={'pid':os.getpid(),'started':time.time(),'gpu':6,'steps':10000,'state_dim':131}
def status(**kw):
 state.update(kw);p=O/'proprio-pipeline-status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(O/'proprio-pipeline-status.json')
def run(cmd,name):
 with (O/(name+'.log')).open('w') as f:
  child=subprocess.Popen([str(x) for x in cmd],cwd=O,env=env,stdout=f,stderr=subprocess.STDOUT);status(child_pid=child.pid);rc=child.wait()
 if rc:raise RuntimeError(f'{name} exited {rc}')
try:
 status(phase='waiting_for_same_dataset')
 while not (O/'cup-native50.hdf5').exists():
  if (O/'pipeline-status.json').exists():assert json.loads((O/'pipeline-status.json').read_text())['phase']!='failed','Base preparation failed'
  time.sleep(5)
 status(phase='training');run([B/'env-act/bin/python',O/'finetune_proprio.py','--kind','act','--base',B,'--data',O/'cup-native50.hdf5','--source',OLD/'act-run/latest.pt','--out',O/'act-proprio-run','--steps','10000','--lr','1e-5','--save-every','5000'],'proprio-train')
 status(phase='exporting');run([B/'env-act/bin/python',O/'export_proprio.py','--source',O/'act-proprio-run/latest.pt','--out',O/'exports/act-proprio'],'proprio-export')
 status(phase='evaluating');run([B/'env-dp/bin/python',O/'evaluate.py','--weights',O/'exports/act-proprio','--out',O/'evaluation-proprio','--hz','50'],'evaluation-proprio')
 s=json.loads((O/'evaluation-proprio/summary.json').read_text());assert s['n']==10 and not s['errors'];status(phase='completed',successes=s['successes'],n=10,elapsed_s=time.time()-state['started'])
except Exception as e:status(phase='failed',error=repr(e));raise
