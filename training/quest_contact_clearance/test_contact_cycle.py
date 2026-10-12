import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
import mujoco
from contact_clearance import kinematics, contact_penalty, clearance_penalty
from nonlooping_v2 import completion, install
from build_phase import infer


class Rewards(unittest.TestCase):
    def states(self):
        s = torch.zeros(1,2,13);s[...,6]=1;s[...,2]=.03
        centers = torch.tensor([[[-.1,-.03,-.02],[-.1,.03,-.02],[.1,-.03,-.02],[.1,.03,-.02]]]*2)
        radii = torch.full((2,4),.01)
        return s,centers,radii,torch.zeros(2,3)

    def test_static_load_not_duplicated(self):
        args=self.states();h,g,v,p=kinematics(*args)
        loss,slip,missing,load=contact_penalty(g,v,torch.tensor([[200.,200.]]),torch.tensor([True]),torch.tensor([400.]))
        self.assertAlmostEqual(float(loss),0.)
        self.assertAlmostEqual(float(load.sum()),400.)
        self.assertTrue(torch.allclose(h,torch.zeros_like(h),atol=1e-7))

    def test_rotation_produces_sliding(self):
        args=self.states();args[0][...,12]=2.
        h,g,v,p=kinematics(*args)
        loss,*_=contact_penalty(g,v,torch.tensor([[200.,200.]]),torch.tensor([True]),torch.tensor([400.]))
        self.assertGreater(float(loss),.1)

    def test_airborne_missing_support_and_clearance_overshoot(self):
        args=self.states();args[0][...,2]=1.
        h,g,v,p=kinematics(*args)
        loss,*_=contact_penalty(g,v,torch.zeros(1,2),torch.tensor([True]),torch.tensor([400.]))
        self.assertEqual(float(loss),1.)
        target=torch.full((1,2),.035);valid=torch.ones_like(target)
        self.assertEqual(float(clearance_penalty(target,target,valid)),0.)
        self.assertGreater(float(clearance_penalty(target+.03,target,valid)),0.)
        self.assertEqual(float(clearance_penalty(h,target,valid*0)),0.)

    def test_mujoco_geometry_and_point_velocity(self):
        args=self.states();s,c,r,origin=args
        quat=np.array([.91,.11,.31,.21]);quat/=np.linalg.norm(quat)
        geoms=''.join('<geom type="sphere" pos="%s" size=".01"/>'%' '.join(map(str,v)) for v in c[0].numpy())
        m=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><body name="foot"><freejoint/>'+geoms+'</body></worldbody></mujoco>')
        d=mujoco.MjData(m);d.qpos[:3]=[.2,.1,.5];d.qpos[3:7]=quat;d.qvel[:]=[.1,.2,.3,.4,.5,.6];mujoco.mj_forward(m,d)
        s[0,:, :3]=torch.tensor(d.xpos[1],dtype=torch.float32)
        s[0,:,3:7]=torch.tensor(quat[[1,2,3,0]],dtype=torch.float32)
        jac=np.zeros((3,m.nv));jr=np.zeros_like(jac)
        mujoco.mj_jac(m,d,jac,jr,d.xpos[1],1)
        s[0,:,7:10]=torch.tensor(jac@d.qvel,dtype=torch.float32)
        s[0,:,10:13]=torch.tensor(jr@d.qvel,dtype=torch.float32)
        h,g,v,points=kinematics(s,c,r,origin)
        self.assertAlmostEqual(float(h[0,0]),float((d.geom_xpos[:,2]-.01).min()),places=6)
        for i,point in enumerate(points[0,0].numpy()):
            mujoco.mj_jac(m,d,jac,None,point,1)
            np.testing.assert_allclose(v[0,0,i].numpy(),jac@d.qvel,atol=1e-6)

    def test_teacher_requires_complete_supported_swing(self):
        h=np.zeros((100,2));p=np.zeros((100,2,3))
        h[30:60,0]=.04*np.sin(np.linspace(0,np.pi,30))
        p[:,:,2]=h
        table,intervals=infer(h,p,100.)
        self.assertTrue(intervals)
        self.assertGreater(table[:,0,0].max(),.03)
        h[:,1]=.1
        table,intervals=infer(h,p,100.)
        self.assertFalse(intervals)


class Nonlooping(unittest.TestCase):
    def test_completion_edges(self):
        f=completion(torch.tensor([500,100,25,1,100,100]),torch.full((6,),10.),
            torch.tensor([0.,8.,8.,9.999,8.,8.]),torch.tensor([1,1,0,1,1,0]).bool(),
            torch.tensor([0,0,1,0,1,0]).bool(),.02)
        np.testing.assert_allclose(f.numpy(),[1,1,.25,1,.5,.949],atol=1e-6)

    @unittest.skipUnless(os.environ.get('TWIST_SOURCE'), 'set TWIST_SOURCE for actual pinned methods')
    def test_pinned_termination_and_curriculum(self):
        path=Path(os.environ['TWIST_SOURCE']);tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='check_termination' for x in n.body))
        selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('check_termination','_update_motion_difficulty','reset_idx')]
        namespace={'torch':torch}
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
        Fake=type('Fake',(),{n.name:namespace[n.name] for n in selected})
        e=Fake();e.device='cpu';e.num_envs=6;e.dt=.02
        e.contact_forces=torch.zeros(6,1,3);e.termination_contact_indices=torch.tensor([0])
        e.root_states=torch.zeros(6,13);e._ref_root_pos=torch.zeros(6,3)
        e.roll=torch.zeros(6);e.pitch=torch.zeros(6);e.viewer=None;e._pose_termination=False;e._pose_termination_dist=.7
        e._motion_ids=torch.arange(6);e._motion_time_offsets=torch.tensor([0.,8.,8.,9.999,8.,8.])
        e.episode_length_buf=torch.tensor([500,100,25,1,100,100]);e.max_episode_length=99
        e.contact_forces[[2,4],0,2]=10
        e.cfg=NS(rewards=NS(root_height_diff_threshold=.5,termination_roll=1.,termination_pitch=1.),motion=NS(motion_curriculum_gamma=.01))
        e.motion_difficulty=torch.full((6,),5.)
        e._get_motion_times=lambda:e.episode_length_buf*e.dt+e._motion_time_offsets
        recorded=[]
        def frame(ids,t):
            recorded.append(t.clone());return tuple(torch.ones(len(ids),3) for _ in range(9))
        e._motion_lib=NS(get_motion_length=lambda ids:torch.full_like(ids,10.,dtype=torch.float32),num_motions=lambda:6,calc_motion_frame=frame)
        counts=install(e);e.check_termination()
        self.assertEqual(e.time_out_buf.tolist(),[True,True,False,True,False,True])
        self.assertTrue(e.terminal_reasons['failure'][4])
        e._update_motion_difficulty(torch.arange(6))
        self.assertLess(float(e.motion_difficulty[1]),5.)
        self.assertGreater(float(e.motion_difficulty[4]),5.)
        times=torch.tensor([-1.,11.]);copy=times.clone();frames=e._motion_lib.calc_motion_frame(torch.tensor([0,1]),times)
        self.assertTrue(torch.equal(times,copy));self.assertGreaterEqual(float(recorded[-1].min()),0.)
        self.assertLess(float(recorded[-1].max()),10.)
        self.assertEqual(float(frames[2][1].sum()),0.)


if __name__=='__main__':unittest.main()
