"""Physical contracts, independently of optimizer implementation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from adapter_core import *
from adapter_env import AdapterEnv
from continuation import set_clearance_multiplier,restore_optimizer
from clearance_reward import ClearanceConfig,velocity_clearance_cost
from swing_intent import SwingIntent,intent_cost,scheduled_swing,initialize_with_extra_inputs


class Contracts(unittest.TestCase):
    def test_expected_transfer_still_penalizes_a_frozen_low_foot(self):
        intent=SwingIntent(1,'cpu');points=torch.zeros(1,2,3)
        held=torch.tensor([False]);xy=torch.zeros(1,2);yaw=torch.zeros(1)
        intent.update(points,held,xy,yaw)
        self.assertEqual(float(intent.cost(torch.zeros(1,2))),0.)
        points[:,0,0]=.01;intent.update(points,held,xy,yaw)
        frozen_height=torch.zeros(1,2,requires_grad=True)
        cost=intent.cost(frozen_height)
        self.assertGreater(float(cost),.05);cost.sum().backward()
        self.assertLess(float(frozen_height.grad[0,0]),0.)
        self.assertEqual(float(frozen_height.grad[0,1]),0.)
        self.assertEqual(float(intent.cost(torch.full((1,2),.07))),0.)

    def test_standing_and_reset_do_not_request_a_swing(self):
        intent=SwingIntent(2,'cpu');points=torch.zeros(2,2,3)
        for _ in range(100):intent.update(points,torch.tensor([False,True]),torch.zeros(2,2),torch.zeros(2))
        self.assertEqual(float(intent.weight.sum()),0.)
        intent.reset(torch.arange(2));points+=100
        intent.update(points,torch.tensor([False,True]),torch.zeros(2,2),torch.zeros(2))
        self.assertEqual(float(intent.weight.sum()),0.)

    def test_commanded_turn_cycles_without_actual_motion_or_contact(self):
        intent=SwingIntent(1,'cpu');points=torch.zeros(1,2,3);seen=torch.zeros(2)
        for _ in range(100):
            intent.update(points,torch.tensor([True]),torch.zeros(1,2),torch.ones(1)*.5)
            seen+=intent.weight[0]
            self.assertLessEqual(float(intent.weight.sum()),1.)
            self.assertTrue(torch.equal(intent.features()[:,:2],intent.weight))
            if float(intent.target.max())>.03:self.assertGreater(float(intent.cost(torch.zeros(1,2))),0.)
        self.assertTrue((seen>10).all())
        for _ in range(55):intent.update(points,torch.tensor([True]),torch.zeros(1,2),torch.zeros(1))
        self.assertEqual(float(intent.weight.sum()),0.)

    def test_added_phase_inputs_preserve_initial_actor_mapping(self):
        class Model(torch.nn.Module):
            def __init__(self,n):
                super().__init__();self.actor=torch.nn.Module()
                self.actor.actor_backbone=torch.nn.Sequential(torch.nn.Linear(n,16),torch.nn.ELU(),torch.nn.Linear(16,12))
                self.critic=torch.nn.Sequential(torch.nn.Linear(n,16),torch.nn.ELU(),torch.nn.Linear(16,1))
        torch.manual_seed(5);old=Model(1467);new=Model(1473)
        initialize_with_extra_inputs(new,old.state_dict());x=torch.randn(8,1467);extra=torch.randn(8,6)
        self.assertTrue(torch.allclose(old.actor.actor_backbone(x),new.actor.actor_backbone(torch.cat((x,extra),1)),atol=1e-6))
        self.assertEqual(float(new.actor.actor_backbone[0].weight[:,-6:].abs().sum()),0.)

    def test_clearance_multiplier_preserves_geometry_and_one_dt(self):
        state=SimpleNamespace(config=ClearanceConfig())
        env=SimpleNamespace(dt=.02,reward_scales={'low_moving_foot':-.02})
        height=torch.tensor([[0.,.1]]);speed=torch.tensor([[[1.,0.],[1.,0.]]])
        before=velocity_clearance_cost(height,speed,state.config)[0]
        set_clearance_multiplier(env,state,5.)
        after=velocity_clearance_cost(height,speed,state.config)[0]
        self.assertTrue(torch.equal(before,after))
        self.assertAlmostEqual(env.reward_scales['low_moving_foot'],-.1)
        self.assertTrue(torch.allclose(after*env.reward_scales['low_moving_foot'],5*before*(-.02)))
        with self.assertRaises(ValueError):set_clearance_multiplier(env,state,5.)

    def test_resume_restores_adam_moments_and_adaptive_lr(self):
        import copy
        p=torch.nn.Parameter(torch.tensor([.7]));opt=torch.optim.Adam([p],lr=.00007)
        for _ in range(3):
            opt.zero_grad();p.square().sum().backward();opt.step()
        q=torch.nn.Parameter(p.detach().clone());other=torch.optim.Adam([q],lr=.0002)
        alg=SimpleNamespace(optimizer=other,learning_rate=.0002)
        checkpoint={'optimizer':copy.deepcopy(opt.state_dict()),'iteration':300}
        self.assertEqual(restore_optimizer(alg,checkpoint),300)
        self.assertEqual(alg.learning_rate,.00007)
        for param,optimizer in [(p,opt),(q,other)]:
            optimizer.zero_grad();param.square().sum().backward();optimizer.step()
        self.assertTrue(torch.equal(p,q))

    def test_zero_identity_upper_unchanged_and_limits(self):
        raw=torch.randn(32,29)*.1;zero=torch.zeros(32,12)
        q,d=corrected_joints(raw,zero,zero,torch.full((29,),-1.),torch.ones(29))
        self.assertTrue(torch.equal(q,raw));self.assertTrue(torch.equal(d,zero))
        q,d=corrected_joints(raw,zero,torch.full_like(zero,100),torch.full((29,),-.15),torch.full((29,),.15))
        self.assertTrue(torch.equal(q[:,12:],raw[:,12:]));self.assertLessEqual(float(q[:,:12].max()),.150001)

    def test_yaw_wrap_and_stopped_operator_correction(self):
        q=yaw_quat(torch.tensor([3.12]));goal=yaw_quat(torch.tensor([-3.12]));p=torch.zeros(1,3)
        xy,wz,err,dy=camera_servo(p,torch.zeros(1),q,torch.tensor([[1.,0,0]]),goal,p,q)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(xy)),.4,places=6)
        self.assertGreater(float(dy),0);self.assertLess(float(dy),.05)
        self.assertAlmostEqual(float(wz),3*float(dy),places=6)

    def test_observation_history_layout(self):
        c=torch.arange(35.)[None];p=torch.arange(92.)[None];h=torch.arange(1270.).reshape(1,10,127)
        x,current=tracker_observation(c,p,h)
        self.assertEqual(x.shape,(1,1432));self.assertTrue(torch.equal(x[:,:127],current))
        self.assertTrue(torch.equal(x[:,127:1397],h.flatten(1)));self.assertTrue(torch.equal(x[:,1397:],c))

    def test_projected_reference_is_not_reconstructed(self):
        torch.manual_seed(44);raw=torch.randn(4096,29);projected=raw[:,:12].clamp(-.2,.2)
        delta=projected-raw[:,:12]
        self.assertFalse(torch.equal(raw[:,:12]+delta,projected))
        authoritative=raw.clone();authoritative[:,:12]=projected
        wrapper=AdapterEnv.__new__(AdapterEnv);wrapper.e=SimpleNamespace(_ref_dof_pos=authoritative)
        wrapper.raw={'dof_pos':raw,'root_pos':torch.zeros(4096,3),'root_rot':yaw_quat(torch.zeros(4096))}
        wrapper.delta=delta;wrapper.navigation=lambda:(torch.zeros(4096,2),torch.zeros(4096),None,None)
        self.assertTrue(torch.equal(wrapper.command(True)[:,6:],authoritative))


if __name__=='__main__':unittest.main()
