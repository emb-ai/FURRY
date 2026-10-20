from pathlib import Path
import subprocess,os,json,sys,time,hashlib
import h5py,numpy as np
O=Path(__file__).resolve().parent;env=os.environ.copy();env.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',LP_NUM_THREADS='2',OMP_NUM_THREADS='2');jobs=[];started=time.time()
for i in range(4):
 with (O/f'render-{i}.log').open('w') as f:p=subprocess.Popen([sys.executable,str(O/'render_dataset.py'),'--shard',str(i)],stdout=f,stderr=subprocess.STDOUT,env=env)
 jobs.append(p)
assert all(p.wait()==0 for p in jobs),'Renderer worker failed'
out=O/'cup-native50.hdf5';assert not out.exists()
with h5py.File(out.with_suffix('.partial.hdf5'),'w') as h:
 h.attrs['fps']=50;h.attrs['sampling']='native HOMIE phase, step modulo20=0';h.attrs['action_schema']='body35_grip2_gaze2';g=h.create_group('data')
 for i in range(4):
  with h5py.File(O/f'shard-{i}.hdf5') as src:
   for name in src['data']:src.copy(src['data'][name],g,name=name)
 names=[f'demo_{i}'.encode() for i in range(59)];assert len(g)==59;h.create_dataset('mask/train',data=np.array(names,dtype='S32'));h.create_dataset('mask/valid',data=np.array([],dtype='S32'));count=sum(len(g[n.decode()]['actions']) for n in names)
out.with_suffix('.partial.hdf5').replace(out);r={'status':'completed','episodes':59,'frames':count,'fps':50,'elapsed_s':time.time()-started,'bytes':out.stat().st_size,'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(out.read_bytes()).hexdigest()};(O/'render-status.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
