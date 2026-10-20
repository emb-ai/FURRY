from pathlib import Path
import subprocess,concurrent.futures,json,hashlib
import os,mujoco,onnxruntime
os.environ["DYLD_LIBRARY_PATH"]=str(Path(mujoco.__file__).parent)+":"+str(Path(onnxruntime.__file__).parent/"capi")
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path.cwd();O=R/'outputs/policy-comparison-20261016';A=R/'outputs/training-capture-20261015/project/android/assets';SRC=R/'outputs/stance-finetune-20261013/A-prepared/sources/epoch1.txt'
V=[('twist2-20k','TWIST2 Original','base'),('quest-111586-500','TWIST2 Finetuned','base'),('quest-111586-500-lift10','TWIST2 Lift','lift'),('quest-stance12-B-s11','TWIST2 Narrow','narrow'),('homie-v2','HOMIE v2','base'),('dexweave-wide-2977','DexWeave Wide','base'),('dexweave-B-4174','DexWeave Narrow','narrow'),('sonic-vr3','SONIC VR3','base')]
def refs(mode):
 if (O/f"refs-{mode}.csv").exists():return
 with (O/f'refs-{mode}.log').open('w') as log:
  subprocess.run([str(O/'native'),'refs',str(A),str(SRC),str(O/f'refs-{mode}.csv'),mode],stdout=log,stderr=log,check=True)
 print('refs done',mode,flush=True)
with concurrent.futures.ThreadPoolExecutor(3) as ex:list(ex.map(refs,['base','lift','narrow']))
raw={mode:np.loadtxt(O/f'refs-{mode}.csv',delimiter=',') for mode in ['base','lift','narrow']}
b=raw['base'];q0=b[0,1:37];rot=Rotation.from_quat(q0[[4,5,6,3]]);yaw=rot.as_euler('xyz')[2];align=Rotation.from_euler('z',-yaw)
times=np.arange(2001)*.01;arrays={};cmds={}
for mode,d in raw.items():
 d=d[np.r_[True,np.diff(d[:,0])>0]]
 t=(d[:,0]-b[0,0])*1e-9;assert abs(t[0])<1e-8 and t[-1]>=20
 q=d[:,1:37].copy();q[:,:3]=align.apply(q[:,:3]-[q0[0],q0[1],0]);qr=(align*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat();q[:,3:7]=qr[:,[3,0,1,2]]
 dst=np.column_stack([np.interp(times,t,q[:,k]) for k in range(36)]);dst[:,3:7]=Slerp(t,Rotation.from_quat(q[:,[4,5,6,3]]))(times).as_quat()[:,[3,0,1,2]]
 cmd=np.column_stack([np.interp(times,t,d[:,37+k]) for k in range(35)])
 arrays[mode]=dst;cmds[mode]=cmd
 np.savetxt(O/f'input-{mode}.txt',np.c_[times[:-1],dst[:-1],cmd[:-1]],fmt='%.17g')
np.savez_compressed(O/'references.npz',time=times,**arrays)
cat=json.loads((A/'policy_catalog.json').read_text());results=[]
def rollout(v):
 id,title,mode=v
 with (O/f'{id}.log').open('w') as log:r=subprocess.run([str(O/'native'),'run',str(A),id,str(O/f'input-{mode}.txt'),str(O/f'{id}.csv')],stdout=log,stderr=log)
 print('rollout done',id,r.returncode,flush=True)
 return dict(id=id,title=title,reference=mode,exit_code=r.returncode,policy=next(x for x in cat if x['id']==id))
with concurrent.futures.ThreadPoolExecutor(2) as ex:results=list(ex.map(rollout,V))
(O/'protocol.json').write_text(json.dumps(dict(variants=results,source=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),clip='stage09-run00 first20s',native_assets=str(A),duration_s=20,physics_hz=1000,policy_hz=50,reference='Continuous current native Meta/GMR from original calibration, three native menu input modes. No offline filtering or floor guard. Observer camera target from FK. Common yaw and XY transform. Each policy native reset and common baseline neutral physical pose; zero velocities and empty histories; native .5s command blend.',catch_up='Native v2 for TWIST2, HOMIE, SONIC; native absolute reference for DexWeave.',metric='Mean root XY Euclidean error in cm, mean absolute wrapped root yaw error in degrees, same timestamps and common base GMR reference; no posthoc alignment. Falls recorded separately, no reset.'),indent=2))
