from pathlib import Path
import json,sys,time,os
import numpy as np,torch
import random_eval as current
O=Path(__file__).resolve().parent;D=O/'train-diagnostic';D.mkdir(exist_ok=True)
ev=current.ev;ev.O=D
sys.path.insert(0,str(O.parent/'act-cup-eval-20261015'))
import evaluate_act as act
act.O=D;act.lib=current.lib
names=json.loads((O/'quality-audit.json').read_text())['accepted'];names=[names[i] for i in np.linspace(0,len(names)-1,8,dtype=int)]
(D/'selection.json').write_text(json.dumps({'selection':'8 evenly spaced accepted episodes fixed before tests','episodes':names},indent=2))
state={'pid':os.getpid(),'phase':'recorded_controls','started':time.time(),'results':[]}
def save():
 p=D/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(D/'status.json')
save()
try:
 torch.set_num_threads(4)
 for name in names:
  e=ev.inspect_episode(O/'recordings'/name)
  for mode in ['oracle100','oracle30']:
   state['active']=name;state['mode']=mode;save()
   state['results'].append(ev.run(e,mode,None,None,'cpu',0,90,8,'new-homie-recorded'));save()
 state['phase']='act_train';save()
 model=torch.jit.load(str(O/'finetuned-weights/act/act-inference.pt')).eval();stats=json.loads((O/'finetuned-weights/act/stats.json').read_text())
 for name in names:
  state['active']=name;save();e=ev.inspect_episode(O/'recordings'/name)
  r=act.run(e,model,stats,8,90,'new-train');r['checkpoint_step']=20000;r['mode']='act';state['results'].append(r);save()
 state['phase']='completed';state['summary']={m:{'successes':sum(x['success'] for x in state['results'] if x['mode']==m),'n':sum(x['mode']==m for x in state['results']),'errors':[x['error'] for x in state['results'] if x['mode']==m and x['error']]} for m in ['oracle100','oracle30','act']};save();print(json.dumps(state['summary']),flush=True)
except Exception as e:state['phase']='failed';state['error']=repr(e);save();raise
