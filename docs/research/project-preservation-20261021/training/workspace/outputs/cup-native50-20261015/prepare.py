from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np
O=Path(__file__).resolve().parent;SRC=O.parent/'cup-homie-finetune-20261015';sys.path.insert(0,str(O.parent/'cup-dp-training-20261015'));from capture_snapshot import inspect_episode,columns
P=O/'render-input';P.mkdir(exist_ok=True);names=json.loads((SRC/'quality-audit.json').read_text())['accepted'];rows=[]
for i,name in enumerate(names):
 e=inspect_episode(SRC/'recordings'/name);steps=np.rint(e['frames']['step']).astype('int64');ids=np.arange(e['start'],e['end']+1);ids=ids[steps[ids]%20==0];assert len(ids)>1 and np.all(np.diff(steps[ids])==20)
 if i==0:shutil.copyfile(e['path']/'learning-model.mjb',P/'learning-model.mjb')
 assert hashlib.sha256((e['path']/'learning-model.mjb').read_bytes()).hexdigest()==e['capture']['model_sha256']
 act=e['actions'][ids].astype('float32');prev=np.zeros_like(act);prev[1:]=act[:-1];gaze=prev[:,37:39].copy();gaze[0]=act[0,37:39]
 np.savez_compressed(P/f'demo_{i}.npz',actions=act,previous_command=prev,qpos=columns(e['frames'],'qpos',57)[ids],qvel=columns(e['frames'],'qvel',55)[ids],mocap_pos=columns(e['scene'],'mocap_pos',3)[ids],mocap_quat=columns(e['scene'],'mocap_quat',4)[ids],userdata=columns(e['scene'],'userdata',8)[ids],gaze=gaze,source_frame_index=ids,source_sim_time=e['frames']['sim_time'][ids],source_step=steps[ids])
 rows.append({'episode':name,'key':f'demo_{i}','samples':len(ids),'model_sha256':e['capture']['model_sha256']});print(i+1,len(ids),flush=True)
report={'fps':50,'phase':'native step modulo20=0','episodes':rows,'n':len(rows),'samples':sum(r['samples'] for r in rows),'cold_start_examples':False,'all_train':True};(P/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'n':len(rows),'samples':report['samples']}))
