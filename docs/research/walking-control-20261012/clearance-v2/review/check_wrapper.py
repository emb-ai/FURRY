"""Independent CPU integration oracle; reads but never modifies v2-source."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
sys.path.insert(0,str(SOURCE))
from clearance_reward import install,sole_geometry_cost
from sole_kinematics import SoleKinematics

import argparse
p=argparse.ArgumentParser();p.add_argument('--urdf',required=True,type=Path);args=p.parse_args()
URDF=args.urdf
names=[j.get('name') for j in ET.parse(URDF).getroot().findall('joint') if j.get('type') in ('revolute','continuous')]
torch.set_num_threads(1)

class Gym:
    def get_actor_dof_names(self,*a):return names
    def get_actor_count(self,*a):return 1
    def get_actor_rigid_body_names(self,*a):return ['left_ankle_roll_link','right_ankle_roll_link']
    def get_actor_rigid_body_properties(self,*a):return [SimpleNamespace(mass=80.)]

class Environment:
    def __init__(self):
        self.dt=.02;self.num_obs=1432;self.num_envs=2;self.device='cpu'
        self.gym=Gym();self.envs=[0,1];self.actor_handles=[0,0]
        self.cfg=SimpleNamespace(terrain=SimpleNamespace(mesh_type='plane'))
        self.feet_indices=torch.tensor([0,1]);self.rigid_body_states=torch.zeros((2,2,13))
        self.rigid_body_states[...,6]=1;self.rigid_body_states[...,2]=.035
        self.contact_forces=torch.zeros((2,2,3));self.contact_forces[...,2]=392.4
        self.reward_names=['feet_slip'];self.reward_functions=[lambda:torch.full((2,),7.)]
        self.reward_scales={'feet_slip':-.1*self.dt};self.episode_sums={'feet_slip':torch.zeros(2)}
        self.original_reset_count=0
    def compute_reward(self):
        self.reward=sum(fn()*self.reward_scales[name] for name,fn in zip(self.reward_names,self.reward_functions))
    def reset_idx(self,ids):self.original_reset_count+=1

control,treatment=Environment(),Environment()
cs,ts=install(control,URDF,False),install(treatment,URDF,True)
assert control.reward_names==['feet_slip']
assert treatment.reward_names==['feet_slip','low_moving_foot']
assert treatment.reward_scales['low_moving_foot']==-.02
for env in (control,treatment):env.compute_reward()
assert ts.terms[1].eq(0).all() # No pre-reset position can generate a valid velocity.
for env in (control,treatment):
    env.rigid_body_states[:,0,0]+=.02
    env.compute_reward()
expected=float(-np.expm1(-(.065**2*np.tanh(2))/.05))
np.testing.assert_allclose(ts.terms[1].numpy(),expected,rtol=1e-6,atol=1e-7)
np.testing.assert_allclose(control.reward.numpy(),-.014,rtol=1e-7)
np.testing.assert_allclose((treatment.reward-ts.terms[0]*treatment.reward_scales['feet_slip']).numpy(),-.02*expected,rtol=1e-6,atol=1e-8)
# An actual contact does not exempt the moving, low foot from the clearance cost.
assert treatment.contact_forces[:,0,2].gt(5).all() and ts.terms[1].gt(0).all()
# Reset one env, simulate an arbitrary relocation; only that env loses its first FD sample.
treatment.reset_idx(torch.tensor([0]));treatment.rigid_body_states[0,:,0]+=10
treatment.rigid_body_states[1,0,0]+=.02;treatment.compute_reward()
assert ts.terms[1][0]==0 and ts.terms[1][1]>0 and treatment.original_reset_count==1
# No extra phase features or actor dimensions appear in the state.
assert not hasattr(ts,'features') and not hasattr(ts,'feature_dim')

# Independent SciPy quaternion and sphere-bottom oracle, both feet, arbitrary tilt.
kin=SoleKinematics(URDF,names,dtype=torch.float64)
rng=np.random.default_rng(9031);q=rng.normal(size=(30,2,4));q/=np.linalg.norm(q,axis=-1,keepdims=True)
pos=rng.normal(size=(30,2,3))*.1;pos[...,2]+=.05
points=kin.world_soles(torch.tensor(pos),torch.tensor(q))
expected_points=np.empty(points.shape)
for n in range(30):
    for side in range(2):expected_points[n,side]=pos[n,side]+Rotation.from_quat(q[n,side]).apply(kin.points[side].numpy())
np.testing.assert_allclose(points.numpy(),expected_points,rtol=1e-12,atol=1e-12)
previous=points.clone();previous[...,0]-=.006;previous[...,1]+=.008
cost,terms,h,v=sole_geometry_cost(points,previous,kin.radii,.02)
expected_height=(expected_points[...,2]-kin.radii.numpy()).min(-1)
expected_deficit=np.maximum(.065-expected_height,0)
expected_terms=expected_deficit**2*np.tanh(2*.5)
np.testing.assert_allclose(h.numpy(),expected_height,rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(v[...,:2].numpy(),np.broadcast_to([.3,-.4],(30,2,2)),rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(terms.numpy(),expected_terms,rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(cost.numpy(),-np.expm1(-expected_terms.sum(-1)/.05),rtol=1e-12,atol=1e-12)
# Radius stays a world-normal support distance under tilt. Rotating a local
# bottom point would be a different, incorrect sphere-clearance calculation.
result=dict(status='PASS',scope='Independent CPU wrapper/math tests; Isaac GPU smoke still required',
    integration_cases=['control exact old reward','dt once at registration','actual contact cannot gate clearance',
                       'first sample suppressed','subset reset prevents FD teleport spike','no added phase observations'],
    independent_geometry_cases=60,quaternion='SciPy XYZW vs Torch',sphere_support='min(world_centres_z-radius)',
    moving_floor_cost=expected,expected_weighted_reward=-.02*expected,
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'clearance_reward.py',SOURCE/'sole_kinematics.py',SOURCE/'clearance_finetune.py']})
(HERE/'wrapper-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
