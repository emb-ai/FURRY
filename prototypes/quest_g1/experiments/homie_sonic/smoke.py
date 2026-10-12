from pathlib import Path
import sys,time,json
import numpy as np,mujoco
APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP))
from g1_sim.autonomous_legs import Homie,Sonic
from g1_sim.controller import standing_command
ROOT=APP.parents[3];assets=APP/'android/assets';models=ROOT/'outputs/homie-sonic-comparison-20261012/models'
print('assets',assets,flush=True)
for backend in sys.argv[1:] or ['homie','sonic']:
 m=mujoco.MjModel.from_xml_path(str(assets/'scene-stand.xml'));m.opt.timestep=.001
 p=Homie(m,assets) if backend=='homie' else Sonic(m,assets,models,Path(__file__).with_name('sonic_constants.json'))
 d=mujoco.MjData(m);p.initialize(d);ref=np.r_[d.qpos[:7],d.qpos[p.qa]]
 cmd=np.r_[0,0,d.qpos[2],0,0,0,d.qpos[p.qa]];start=time.monotonic()
 for i in range(250):
  if i==100:cmd[0]=.3
  p.update(d,cmd,ref)
  for j in range(20):p.apply(d)
  if i%50==0:print(backend,i,d.qpos[:3],np.max(np.abs(p.target)),time.monotonic()-start,flush=True)
  if d.qpos[2]<.35:print('FALL',flush=True);break
 print(backend,'done',d.time,d.qpos[:7],flush=True)
