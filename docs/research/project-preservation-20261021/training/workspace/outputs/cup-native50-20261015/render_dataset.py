from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,h5py,mujoco as mj
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--shard',type=int,default=-1);a=p.parse_args();O=Path(__file__).resolve().parent;P=O/'render-input';manifest=json.loads((P/'manifest.json').read_text());m=mj.MjModel.from_binary_path(str(P/'learning-model.mjb'));assert mj.__version__=='3.3.7',mj.__version__;d=mj.MjData(m);r=mj.Renderer(m,height=360,width=640);cam=m.camera('ego').id;m.light_castshadow[:]=0

def image(z,i):
 d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.mocap_pos.flat[:]=z['mocap_pos'][i];d.mocap_quat.flat[:]=z['mocap_quat'][i];d.userdata[:]=z['userdata'][i];mj.mj_forward(m,d);yaw,pitch=z['gaze'][i];cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch);R=d.cam_xmat[cam].reshape(3,3)@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]);r.update_scene(d,camera=cam)
 for c in r.scene.camera:c.pos[:]=d.cam_xpos[cam];c.forward[:]=-R[:,2];c.up[:]=R[:,1]
 r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0;return np.asarray(Image.fromarray(r.render()).resize((224,224),Image.Resampling.BILINEAR))
if a.preview:
 ims=[]
 for name in ['demo_0','demo_29','demo_58']:
  z=np.load(P/(name+'.npz'))
  for i in [0,len(z['actions'])//2]:ims.append(image(z,i))
 np.save(O/'renderer-check.npy',np.array(ims));r.close();print('preview done');raise SystemExit()
out=O/('cup-native50.hdf5' if a.shard<0 else f'shard-{a.shard}.hdf5');assert not out.exists();start=time.time()
with h5py.File(out.with_suffix('.partial.hdf5'),'w') as h:
 h.attrs['fps']=50;h.attrs['sampling']='native HOMIE policy phase; source step modulo20=0';h.attrs['action_schema']='body35_grip2_gaze2';data=h.create_group('data');names=[]
 for entry in manifest['episodes']:
  if a.shard>=0 and int(entry['key'].split('_')[1])%4!=a.shard:continue
  key=entry['key'];z=np.load(P/(key+'.npz'));n=len(z['actions']);names.append(key.encode());demo=data.create_group(key);demo.attrs['episode']=entry['episode'];demo.attrs['num_samples']=n;obs=demo.create_group('obs');images=obs.create_dataset('agentview_image',(n,224,224,3),dtype='uint8',chunks=(1,224,224,3),compression='lzf')
  for k in ['actions','source_frame_index','source_sim_time','source_step']:demo.create_dataset(k,data=z[k])
  obs.create_dataset('previous_command',data=z['previous_command']);obs.create_dataset('physical_qpos',data=z['qpos']);obs.create_dataset('physical_qvel',data=z['qvel'])
  for i in range(n):images[i]=image(z,i)
  h.flush();status={'status':'rendering','episodes':len(names),'target_episodes':59,'elapsed_s':time.time()-start};(O/('render-status.json' if a.shard<0 else f'render-status-{a.shard}.json')).write_text(json.dumps(status));print(json.dumps(status),flush=True)
 h.create_dataset('mask/train',data=np.array(names,dtype='S32'));h.create_dataset('mask/valid',data=np.array([],dtype='S32'))
out.with_suffix('.partial.hdf5').replace(out);r.close();status.update(status='completed',sha256=hashlib.sha256(out.read_bytes()).hexdigest(),samples=manifest['samples']);(O/('render-status.json' if a.shard<0 else f'render-status-{a.shard}.json')).write_text(json.dumps(status,indent=2));print(json.dumps(status))
