from pathlib import Path
import json,numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path(str(P.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'))
qa=[int(m.joint('rubik_hinge_'+f).qposadr[0]) for f in ['R','L','B','F','U','D']]
for name in __import__('sys').argv[1:]:
 p=P/(name+'.npz')
 if not p.exists():continue
 z=np.load(p);t=z['trace'];events=json.loads(p.with_suffix('.json').read_text());print('\n'+name)
 last=0
 for event in events:
  k=int(np.argmin(abs(t[:,0]-event['time'])));last=min(last,k);slice_=t[last:k+1];u=z['userdata'][k];f=int(u[3]);q=z['qpos'][k];quat=R.from_quat(t[k,[6,7,8,5]]);ang=quat.as_rotvec(degrees=True)
  err=np.rad2deg(np.remainder(q[qa[f]]-u[1]+np.pi/4,np.pi/2)-np.pi/4) if f>=0 else 0
  print(event['kind'],round(event['time'],2),'turns',int(u[2]),'face',f,'err',round(err,2),'rot',ang.round(1).tolist(),'xyz',t[k,2:5].round(3).tolist(),'hands',t[k,12:14].astype(int).tolist(),'minZ',round(float(slice_[:,4].min()),3),'supportFrames',int((z['support'][last:k+1]>0).sum()),'Rfree',int((slice_[:,13]==0).sum()))
  last=k+1
