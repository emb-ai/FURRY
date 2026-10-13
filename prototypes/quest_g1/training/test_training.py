"""Run with python -m unittest discover -s training -p test_training.py."""
import unittest
import numpy as np
from rewards import RewardConfig,sliding_rate,global_tracking_reward,discrete_rate
from ppo_utils import advantages,actor_lr,GAMMA,GAE_LAMBDA
from evaluate import compare,METRICS

class RewardTests(unittest.TestCase):
 def test_sustained_drag_costs_more_than_short_impact(self):
  cfg=RewardConfig();drag=.2*sliding_rate([100],[.1],cfg);impact=.005*sliding_rate([100],[1.],cfg)
  self.assertGreater(drag,impact*3)
 def test_no_contact_or_no_slip_no_penalty(self):
  self.assertEqual(sliding_rate([0],[10],RewardConfig()),0)
  self.assertEqual(sliding_rate([100],[0],RewardConfig()),0)
 def test_physical_time_invariance(self):
  rate=sliding_rate([50,100],[.1,.2],RewardConfig())
  for hz in [50,100,1000]:self.assertAlmostEqual(sum(rate/hz for _ in range(hz)),rate)
  for t in [.5,1,2,10]:
   self.assertAlmostEqual(GAMMA**(100*t),.99**(50*t));self.assertAlmostEqual((GAMMA*GAE_LAMBDA)**(100*t),(.99*.95)**(50*t))
 def test_global_signal_not_saturated_at_one_meter(self):
  error=np.tile([1.,0,0],(9,1));self.assertAlmostEqual(global_tracking_reward(error,RewardConfig()),.4)
  self.assertLess(global_tracking_reward(error,RewardConfig(legacy=True)),1e-30)
 def test_action_change_weight_same_per_second(self):
  cfg=RewardConfig();totals=[]
  for hz in [50,100,200]:
   dt=1/hz;delta_action=.7*dt
   totals.append(sum(discrete_rate(delta_action,dt,cfg)*dt for _ in range(hz)))
  np.testing.assert_allclose(totals,[.014]*3)
 def test_touchdown_cost_same_per_event(self):
  cfg=RewardConfig()
  for dt in [.02,.01,.005]:self.assertAlmostEqual(discrete_rate(-.2,dt,cfg)*dt,-.004)
 def test_invalid_config(self):
  with self.assertRaises(ValueError):RewardConfig(slide_weight=-1)
  with self.assertRaises(ValueError):RewardConfig(slide_epsilon=0)

class PPOTests(unittest.TestCase):
 def test_terminal_does_not_leak_next_episode(self):
  r=np.array([[1.],[100.]],np.float32);v=np.array([[.4],[5.]],np.float32);d=np.array([[True],[False]])
  adv,ret=advantages(r,v,d,np.array([7.]),gamma=.9,lam=.8)
  self.assertAlmostEqual(float(ret[0,0]),1);self.assertAlmostEqual(float(adv[0,0]),.6,places=6)
 def test_timeout_bootstrap_once(self):
  r=np.array([[1.+.9*2.],[100.]],np.float32);v=np.array([[.4],[5.]],np.float32)
  adv,ret=advantages(r,v,np.array([[True],[False]]),np.array([7.]),gamma=.9,lam=.8)
  self.assertAlmostEqual(float(ret[0,0]),2.8,places=6)
 def test_actor_lr_ramp(self):
  self.assertEqual(actor_lr(0,1e-5,50),0);self.assertAlmostEqual(actor_lr(1,1e-5,50),1e-6)
  self.assertAlmostEqual(actor_lr(50,1e-5,50),1e-5);self.assertAlmostEqual(actor_lr(100,1e-5,50),1e-5)

class AcceptanceTests(unittest.TestCase):
 def report(self,slip=1,track=1,fall=False,n=100):
  arr=np.ones((n,len(METRICS)));arr[:,METRICS.index('slide_rate')]=slip;arr[:,METRICS.index('joint_rmse')]=track
  return {'dataset_manifest_sha256':'x','trace_columns':METRICS,'trials':[{'trace_key':'a','fell':fall}], '_traces':{'a':arr},'falls':int(fall),'survived_seconds':n*.01}
 def test_identity_is_not_improvement(self):self.assertFalse(compare(self.report(),self.report())['accepted'])
 def test_slip_improves_without_tracking_regression(self):self.assertTrue(compare(self.report(),self.report(slip=.8))['accepted'])
 def test_standing_hack_rejected(self):self.assertFalse(compare(self.report(),self.report(slip=0,track=2))['accepted'])
 def test_early_fall_cannot_hide_loss(self):self.assertFalse(compare(self.report(),self.report(slip=.1,fall=True,n=50))['accepted'])
 def test_prefix_comparison(self):
  b=self.report(n=200,fall=True);c=self.report(n=300)
  c['_traces']['a'][200:]=1000
  self.assertTrue(compare(b,c)['accepted']) # extra survival is not compared against missing frames

try:
 import torch
 from critic import Critic,value_metrics,critic_ready,discounted_returns
except ImportError:
 torch=None

@unittest.skipIf(torch is None,'PyTorch environment required')
class CriticTests(unittest.TestCase):
 def test_biased_value_cannot_pass_on_explained_variance_alone(self):
  y=np.arange(100,dtype=float);m=value_metrics(y+100,y);self.assertAlmostEqual(m['explained_variance'],1)
  self.assertFalse(critic_ready(m));self.assertTrue(critic_ready(value_metrics(y,y)))
 def test_constant_target_is_not_readiness(self):self.assertFalse(critic_ready(value_metrics(np.ones(100),np.ones(100))))
 def test_returns_have_no_bootstrap(self):np.testing.assert_allclose(discounted_returns([1,2,3],.5),[2.75,3.5,3])
 def test_critic_normalization_finite_and_frozen(self):
  c=Critic(3);x=torch.tensor([[1.,0,2],[1.,0,4]]);c.fit_normalization(x);mean=c.mean.clone();scale=c.scale.clone();loss=c(x).sum();loss.backward()
  self.assertTrue(torch.isfinite(c(x)).all());self.assertTrue(torch.equal(mean,c.mean));self.assertTrue(torch.equal(scale,c.scale));self.assertIsNone(c.mean.grad)

if __name__=='__main__':unittest.main()
