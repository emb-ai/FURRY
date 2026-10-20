"""Explicit manifest adapter for the two audited cup model families."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
import h5py
from capture_snapshot import inspect_episode,digest,SCHEMA
from training_renderer import TrainingRenderer
O=Path(__file__).resolve().parent
PLAN=O.parent/'cup-training-plan-20261015'
def main():
 manifest=json.loads((PLAN/'training-manifest.json').read_text())
 out=O/'cup44-no-shadow-causal.hdf5'
 if out.exists():raise FileExistsError(out)
 report={'schema':'furry-cup44-export-v1','renderer':{'shadows':False,'resolution':[224,224],'source_resolution':[640,360],'mujoco':'3.3.7','camera':'current physical head + previous sampled gaze; archived first gaze for initial state'},'episodes':[],'manifest_sha256':digest(PLAN/'training-manifest.json')}
 seen=set();t0=time.time()
 with h5py.File(out.with_suffix('.partial.hdf5'),'w') as h:
  h.attrs['schema']=SCHEMA;h.attrs['fps']=30;h.attrs['renderer']=json.dumps(report['renderer']);h.attrs['action_schema']='body35_grip2_gaze2'
  data=h.create_group('data');masks={'train':[],'valid':[]}
  for split,rows in manifest['splits'].items():
   for row in rows:
    e=inspect_episode(Path(manifest['raw_root'])/row['episode'])
    assert e['identity']==row['identity'] and e['identity'] not in seen
    assert e['family']==manifest['model_families'][row['family_id']]['family']
    seen.add(e['identity']);ids=e['indices'];n=len(ids);name='demo_'+str(len(seen)-1)
    d=data.create_group(name);d.attrs['num_samples']=n;d.attrs['episode']=row['episode'];d.attrs['family']=json.dumps(e['family']);d.attrs['capture']=json.dumps(e['capture']);d.attrs['manifest']=json.dumps(e['meta'])
    d.create_dataset('actions',data=e['actions'][ids].astype('float32'))
    prev=np.zeros((n,39),np.float32);prev[1:]=e['actions'][ids[:-1]]
    obs=d.create_group('obs');obs.create_dataset('previous_command',data=prev)
    imgs=obs.create_dataset('agentview_image',(n,224,224,3),dtype='uint8',chunks=(1,224,224,3),compression='lzf')
    r=TrainingRenderer(e);r.model.light_castshadow[:]=0
    try:
     for k,idx in enumerate(ids):imgs[k]=r.render_observation(idx,ids[max(0,k-1)])
    finally:r.close()
    d.create_dataset('source_frame_index',data=ids);d.create_dataset('timestamps',data=e['grid']);d.create_dataset('source_sim_time',data=e['frames']['sim_time'][ids])
    hashes={p.name:digest(p) for p in e['path'].iterdir() if p.is_file() and p.suffix in ('.json','.csv','.mjb')}
    d.attrs['source_hashes']=json.dumps(hashes)
    masks[split].append(name.encode());report['episodes'].append(dict(row,split=split,demo=name,samples=n,source_hashes=hashes))
    h.flush();print(f'{len(seen)}/44 {split} {row["episode"]} {n} frames {time.time()-t0:.1f}s',flush=True)
  for split,names in masks.items():h.create_dataset('mask/'+split,data=np.array(names,dtype='S32'))
  report['total_frames']=sum(x['samples'] for x in report['episodes']);h.attrs['report']=json.dumps(report)
 out.with_suffix('.partial.hdf5').rename(out);report['sha256']=digest(out);report['seconds']=time.time()-t0
 (O/'dataset-report.json').write_text(json.dumps(report,indent=2));print(report['sha256'],flush=True)
if __name__=='__main__':main()
