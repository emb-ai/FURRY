from pathlib import Path
import os,sys,ctypes,json
import numpy as np
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p.parent/'rubik-canonical-20261015'));os.environ['BRIDGE']='bridge-contacts.dylib'
from common import *
O=p;lib.cube_contacts.argtypes=[ptr,dp,ctypes.c_int];lib.cube_contacts.restype=ctypes.c_int;lib.actual_ctrl.argtypes=[ptr,dp];lib.actuator_force.argtypes=[ptr,dp]
z=dict(np.load(p.parent/'rubik-regrasp-20261015/adaptive_support.npz'));s=Sim('rubik');s.step(n=2500);cq=int(s.m.joint('rubik_free').qposadr[0]);bids=[s.m.body('rubik_cubie_'+str(i)).id for i in range(8)];rows=[];allContacts=[];saved=[];ctrls=[];forces=[];vel=[]
try:
 for k,c in enumerate(z['commands']):
  s.step(c[:14].reshape(2,7),c[-2:])
  if k%1000==0:print('REPLAY',k/100,flush=True)
  if k%10 or k<11500:continue
  buf=np.zeros((256,17));n=lib.cube_contacts(s.p,buf.ravel(),256);contacts=buf[:n].copy();center=s.d.xpos[bids].mean(axis=0);w=np.zeros((2,6))
  for ct in contacts:
   h=int(ct[0]);w[h,:3]+=ct[11:14];w[h,3:]+=np.cross(ct[5:8]-center,ct[11:14])
  control=np.zeros(s.m.nu);force=control.copy();lib.actual_ctrl(s.p,control);lib.actuator_force(s.p,force);v=np.zeros(s.m.nv);lib.velocity(s.p,v)
  rows.append([k/100,*center,*w.ravel(),*c[-2:],*s.state()[[0,2,5,7,66]]]);allContacts.append(contacts);saved.append(s.q.copy());ctrls.append(control);forces.append(force);vel.append(v)
  assert np.max(abs(s.q-z['qpos'][k]))<1e-10
 np.savez_compressed(p/'contact-audit.npz',metrics=rows,qpos=saved,qvel=vel,ctrl=ctrls,forces=forces,contacts=np.array(allContacts,dtype=object));print('EXACT CONTACT REPLAY',flush=True)
finally:s.close()
