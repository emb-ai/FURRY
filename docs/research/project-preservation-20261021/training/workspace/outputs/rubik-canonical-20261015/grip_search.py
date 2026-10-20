from pathlib import Path
import os
import numpy as np
os.environ['BRIDGE']='bridge-snapshot.dylib'
p=Path(__file__).resolve().parent
code=(p/'nudge_search.py').read_text()
code=code.replace("exec((p/'search_return.py').read_text().split('try:\\n obj,Q=pickup')[0])", "source=(p/'search_return.py').read_text().split('try:\\n obj,Q=pickup')[0];source=source.replace('goals=g.copy();grips=np.array(gr);applied=', 'gr=np.minimum(gr,limits);goals=g.copy();grips=np.array(gr);applied=');limits=np.ones(2);exec(source)")
a=code.index('cases=[');b=code.index('finally:lib.free_snapshot',a)
code=code[:a]+'''cases=[(.5,.8),(.6,.8),(.7,.8),(.8,.8),(.5,.6),(.6,.6),(.7,.6),(.8,.6)]
part=int(os.environ.get('PART','0')); cases=cases[4*part:4*part+4]
try:
 for n,(left,right) in enumerate(cases):
  restore();limits=np.array([left,right]);run=f'grip-{part}-{n}';os.environ['DZ']='.008';air_regrip();note('grip-choice',values=limits.tolist())
  turn(obj,Q,110)
  np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
  (O/f'{run}.json').write_text(json.dumps(events,indent=2))
''' + code[b:]
exec(compile(code,str(p/'grip_search.py'),'exec'))
