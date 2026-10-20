"""Prepare a separate, read-only-source action contract for a future native50 export.
Does not render RGB, replace current HDF5, or launch training.
"""
from pathlib import Path
import sys,json
import numpy as np
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
sys.path.insert(0,str(O.parent/'cup-dp-training-20261015'));from capture_snapshot import inspect_episode
out=D/'native50-actions';out.mkdir(exist_ok=True);rows=[]
for name in json.loads((O/'quality-audit.json').read_text())['accepted']:
 e=inspect_episode(O/'recordings'/name);steps=np.rint(e['frames']['step']).astype(np.int64);indices=np.arange(e['start'],e['end']+1);ids=indices[steps[indices]%20==0];assert len(ids)>1 and np.all(np.diff(steps[ids])==20)
 a=e['actions'][ids].astype(np.float32);prev=np.zeros_like(a);prev[1:]=a[:-1];np.savez_compressed(out/(name+'.npz'),actions=a,previous_command=prev,source_frame_index=ids,source_sim_time=e['frames']['sim_time'][ids],source_step=steps[ids])
 rows.append({'episode':name,'samples':len(ids),'first_source_frame':int(ids[0]),'first_sim_time':float(e['frames']['sim_time'][ids[0]]),'body_commands_at_native_policy_ticks_exact':bool(np.array_equal(a[:,:35],e['actions'][ids,:35].astype(np.float32)))})
report={'schema':'native-homie50-action-contract-v1','fps':50,'sampling':'source step modulo20 equals0; pre-action native HOMIE ticks','source_raw_episodes_unchanged':True,'ready_to_train':False,'remaining':['Render matching causal RGB at selected frames and fit/export consistent observation statistics.','Use50Hz actions in inference; preserve physical durations when choosing chunk length.','Address missing full-reset demonstration prefix; these files do not create missing observations.','Retest complete model rollout from full reset.'],'episodes':rows,'n':len(rows),'samples':sum(r['samples'] for r in rows)}
(out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'n':len(rows),'samples':report['samples'],'all_exact':all(r['body_commands_at_native_policy_ticks_exact'] for r in rows)}))
