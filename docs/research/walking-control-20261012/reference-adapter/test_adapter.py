"""Physical contracts, independently of optimizer implementation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from adapter_core import *
from adapter_env import AdapterEnv


class Contracts(unittest.TestCase):
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
