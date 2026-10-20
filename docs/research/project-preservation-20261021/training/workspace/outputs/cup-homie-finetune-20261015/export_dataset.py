from pathlib import Path
import json,sys,time,hashlib
import numpy as np,h5py
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'cup-dp-training-20261015'))
from capture_snapshot import inspect_episode,digest,SCHEMA
from training_renderer import TrainingRenderer
audit=json.loads((O/'quality-audit.json').read_text());out=O/'new-cup.hdf5';assert not out.exists()
report={'episodes':[],'all_accepted_in_train':True,'validation':'10 independently sampled native scenes, fixed seeds; no held-out demonstrations','fps':30};started=time.time()
with h5py.File(out.with_suffix('.partial.hdf5'),'w') as h:
 h.attrs['schema']=SCHEMA;h.attrs['fps']=30;h.attrs['action_schema']='body35_grip2_gaze2';data=h.create_group('data');names=[]
 for j,name in enumerate(audit['accepted']):
  e=inspect_episode(O/'recordings'/name);ids=e['indices'];n=len(ids);key=f'demo_{j}';names.append(key.encode());d=data.create_group(key);d.attrs['episode']=name;d.attrs['family']=json.dumps(e['family']);d.attrs['num_samples']=n
  actions=e['actions'][ids].astype(np.float32);prev=np.zeros_like(actions);prev[1:]=actions[:-1];d.create_dataset('actions',data=actions);obs=d.create_group('obs');obs.create_dataset('previous_command',data=prev)
  images=obs.create_dataset('agentview_image',(n,224,224,3),dtype='uint8',chunks=(1,224,224,3),compression='lzf');r=TrainingRenderer(e);r.model.light_castshadow[:]=0
  try:
   for k,idx in enumerate(ids):images[k]=r.render_observation(idx,ids[max(0,k-1)])
  finally:r.close()
  d.create_dataset('source_frame_index',data=ids);d.create_dataset('source_sim_time',data=e['frames']['sim_time'][ids]);d.create_dataset('timestamps',data=e['grid'])
  report['episodes'].append({'episode':name,'demo':key,'samples':n,'family':e['family'],'identity':e['identity']});h.flush();print(j+1,len(audit['accepted']),name,n,round(time.time()-started,1),flush=True)
 h.create_dataset('mask/train',data=np.array(names,dtype='S32'));h.create_dataset('mask/valid',data=np.array([],dtype='S32'));h.attrs['report']=json.dumps(report)
out.with_suffix('.partial.hdf5').replace(out);report['sha256']=digest(out);report['seconds']=time.time()-started;report['frames']=sum(r['samples'] for r in report['episodes']);(O/'dataset-report.json').write_text(json.dumps(report,indent=2));print('EXPORTED',report['frames'],report['sha256'],flush=True)
