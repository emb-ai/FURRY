"""Test the preflight synchronization adapter, without Isaac or a simulation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('clearance_reward_sync_test_target',HERE/'clearance_reward.py')
candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)


class Kinematics:
    def __init__(self):
        self.radii=torch.full((2,4),.005)
        self.points=torch.zeros((2,4,3))
    def world_soles(self,pos,quat):return pos[:,:,None,:]+self.points[None]
    def reference(self,root_pos,root_quat,dof):
        return self.world_soles(root_pos[:,None]+dof.reshape(-1,2,3),None),None


def fixture(wrong_geometry=False,advancing_refresh=False):
    env=SimpleNamespace(num_envs=2,num_obs=1432,device='cpu',sim=object(),
        feet_indices=torch.tensor([0,1]),root_states=torch.zeros((2,13)),
        dof_pos=torch.zeros((2,6)),rigid_body_states=torch.zeros((2,2,13)),
        reward_scales={'low_moving_foot':-.02})
    env.rigid_body_states[...,6]=1
    env.rigid_body_states[:,0,0]=.05;env.rigid_body_states[:,1,0]=-.01
    if wrong_geometry:env.rigid_body_states[:,0,0]+=.005
    calls=[]
    class Gym:
        def get_sim_time(self,sim):return 2.0+(0.002 if advancing_refresh and 'rigid' in calls else 0)
        def refresh_actor_root_state_tensor(self,sim):calls.append('root');env.root_states[:,0]=.02
        def refresh_dof_state_tensor(self,sim):
            calls.append('dof');env.dof_pos[:,0]=.03;env.dof_pos[:,3]=-.03
        def refresh_rigid_body_state_tensor(self,sim):calls.append('rigid')
        def simulate(self,sim):calls.append('simulate');raise AssertionError('No simulation allowed')
    env.gym=Gym()
    return env,SimpleNamespace(kinematics=Kinematics()),calls


class SyncTests(unittest.TestCase):
    def test_stale_buffers_synced_without_simulation(self):
        env,state,calls=fixture()
        report=candidate.gpu_contract_checks(env,state)
        self.assertEqual(calls,['root','dof','rigid'])
        self.assertGreater(report['initial_unsynchronized_FK_max_abs_m'],.04)
        self.assertLess(report['actual_sphere_vs_root_joint_FK_max_abs_m'],1e-7)
        self.assertEqual(report['sim_time_before_refresh'],report['sim_time_after_refresh'])
        self.assertFalse(report['refresh_advanced_simulation'])
        self.assertAlmostEqual(report['reward_scale'],-.02)
        self.assertGreater(report['moving_floor_cost'],0)
    def test_real_geometry_mismatch_still_rejected(self):
        env,state,calls=fixture(wrong_geometry=True)
        with self.assertRaisesRegex(ValueError,'Rigid-body link pose/FK mismatch'):
            candidate.gpu_contract_checks(env,state)
        self.assertEqual(calls,['root','dof','rigid'])
    def test_refresh_cannot_advance_time(self):
        env,state,calls=fixture(advancing_refresh=True)
        with self.assertRaisesRegex(ValueError,'refresh changed simulation time'):
            candidate.gpu_contract_checks(env,state)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyncTests))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
                candidate_sha256=hashlib.sha256((HERE/'clearance_reward.py').read_bytes()).hexdigest(),
                scope='CPU fake buffer timing only; actual Isaac geometry probe recorded separately')
    (HERE/'sync-tests.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
