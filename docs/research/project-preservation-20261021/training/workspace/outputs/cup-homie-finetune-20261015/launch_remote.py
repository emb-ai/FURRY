from pathlib import Path
import json,os,subprocess,time
O=Path('/home/jovyan/vlad/FURRY/cup-homie-finetune-20261015');B=O.parent/'cup-fast-check-20261015'
gpu='GPU-68d1fffd-2ea2-3f95-f100-38f39855bcdf';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
clip=json.loads((O/'normalization-range-check.json').read_text())['required_clip_sample_range'];runs=[]
for kind,source,python in [('act',B/'act-run-gpu6-colleague-20261015/step-100000.pt',B/'env-act/bin/python'),('dp-epsilon',B/'dp-epsilon-gpu6-resume-20261015/latest.pt',B/'env-dp/bin/python'),('dp-sample',B/'dp-sample/latest.pt',B/'env-dp/bin/python')]:
 out=O/(kind+'-run');assert not out.exists(),str(out)
 cmd=[str(python),str(O/'finetune.py'),'--kind',kind,'--base',str(B),'--data',str(O/'new-cup.hdf5'),'--source',str(source),'--out',str(out),'--steps','20000','--lr','1e-5','--clip-range',str(clip)]
 with (O/(kind+'-run.log')).open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=O,start_new_session=True)
 runs.append({'kind':kind,'pid':p.pid,'gpu':6,'gpu_uuid':gpu,'out':str(out),'source':str(source),'command':cmd,'started':time.time()});(O/'jobs.json').write_text(json.dumps(runs,indent=2))
print(json.dumps(runs),flush=True)
