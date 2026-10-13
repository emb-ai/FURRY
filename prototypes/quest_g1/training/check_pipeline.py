"""Verify deployed observation/PD parity and the prepared data contract."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from env import Motions,Env
from g1_sim.controller import Controller

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',required=True);p.add_argument('--policy',required=True);a=p.parse_args()
 train=Motions(a.dataset,'train');val=Motions(a.dataset,'validation');ts={c['meta']['stage'] for c in train.items};vs={c['meta']['stage'] for c in val.items};assert not ts&vs
 assert not {c['meta']['sha256'] for c in train.items}&{c['meta']['sha256'] for c in val.items}
 replay=np.array([c['meta']['split']=='replay' for c in train.items]);assert np.isclose(train.weights[replay].sum(),.2)
 e=Env(a.assets,val);o=e.reset(clip=0,start=0);d=mujoco.MjData(e.model);d.qpos[:]=e.data.qpos;d.qvel[:]=e.data.qvel;mujoco.mj_forward(e.model,d);ctrl=Controller(e.model,policy=a.policy)
 error=0
 for k in range(100):
  command=e.motion['cmd'][e.k].copy();action=ctrl.session.run(None,{ctrl.input_name:o[None]})[0][0]
  for j in range(10):ctrl.step(d,command)
  o,_,done,_=e.step(action);error=max(error,float(abs(d.qpos-e.data.qpos).max()))
  assert np.allclose(d.qpos,e.data.qpos,atol=1e-6),error
  if done:break
 assert o.shape==(1432,) and np.isfinite(o).all()
 e.max_steps=1;e.reset(clip=0,start=0);_,_,done,info=e.step(np.zeros(29));assert done and info['truncated']
 # Contact instrumentation must not mutate integration or solver warm-start state.
 spec=mujoco.mjtState.mjSTATE_INTEGRATION;before=np.empty(mujoco.mj_stateSize(e.model,spec));after=before.copy()
 mujoco.mj_getState(e.model,e.data,before,spec);e.contact_sample();mujoco.mj_getState(e.model,e.data,after,spec);assert np.array_equal(before,after)
 obs_before=e.last_obs.copy();critic_before=e.critic_observation();e.data.qpos[0]+=1;mujoco.mj_forward(e.model,e.data);critic_shifted=e.critic_observation()
 assert np.array_equal(obs_before,e.last_obs) and np.max(abs(critic_before-critic_shifted))>=1-1e-6
 assert np.isfinite(critic_shifted).all() and len(critic_shifted)>1432
 report={'critic_dim':len(critic_shifted),'critic_observes_world_error':True,'contact_monitor_state_unchanged':True,'train_clips':len(train.items),'validation_clips':len(val.items),'disjoint_stages':True,'replay_sampling_fraction':float(train.weights[replay].sum()),'deployed_controller_max_qpos_error':error,'time_limit_truncation_checked':True}
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
