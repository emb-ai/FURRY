from pathlib import Path
import tempfile,unittest,sys
import h5py,numpy as np,torch
from train import fit_normalizer,Sequences,Policy,scheduler,update_ema,EMAModel
import copy
O=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)/'test.h5'
  with h5py.File(self.p,'w') as h:
   h.attrs['schema']='furry-visual-il-v1'
   for name,offset in [('train',0),('valid',10000)]:
    d=h.create_group('data/'+name);a=np.repeat(np.arange(6,dtype='float32')[:,None],39,axis=1)+offset;c=np.r_[np.zeros((1,39),np.float32),a[:-1]]
    d.create_dataset('actions',data=a);d.create_dataset('obs/previous_command',data=c);d.create_dataset('obs/agentview_image',data=np.broadcast_to(np.arange(6,dtype='uint8')[:,None,None,None],(6,224,224,3)));h.create_dataset('mask/'+name,data=np.array([name.encode()]))
 def tearDown(self):self.tmp.cleanup()
 def test_train_only_stats(self):
  n=fit_normalizer(self.p);np.testing.assert_array_equal(n['actions']['max'],5);np.testing.assert_array_equal(n['obs/previous_command']['max'],4)
 def test_action_observation_alignment_and_padding(self):
  n=fit_normalizer(self.p);d=Sequences(self.p,'train',n,horizon=4,action_steps=3);b=d[2]
  np.testing.assert_array_equal(b['image'][:,0,0,0],[1,2]);a=(b['action'].numpy()-n['actions']['offset'])/n['actions']['scale']
  np.testing.assert_allclose(a[:4,0],[1,2,3,4],atol=1e-6);self.assertEqual(a[-1,0],4)
  self.assertEqual(d[0]['image'][0,0,0,0],0);self.assertEqual(b['action'].shape,(4,39))
  self.assertAlmostEqual(float(a[1,0]),2)
  d.handle.close()
 def test_sampler_padding_matches_upstream(self):
  n=fit_normalizer(self.p);d=Sequences(self.p,'train',n,horizon=6,action_steps=3)
  self.assertEqual(len(d),4) # pad_before=1, pad_after=2, sequence_length=6
  self.assertEqual(d.index[-1][1],3)
 def test_ema_copies_batchnorm_running_statistics(self):
  model=torch.nn.Sequential(torch.nn.BatchNorm1d(3));averaged=copy.deepcopy(model);ema=EMAModel(averaged,power=.75)
  model.train();model(torch.randn(8,3)+7);update_ema(ema,model)
  self.assertTrue(torch.equal(model[0].running_mean,averaged[0].running_mean));self.assertTrue(torch.equal(model[0].num_batches_tracked,averaged[0].num_batches_tracked))
 def test_r3m_loaded_and_eval_deterministic(self):
  torch.set_num_threads(2);n=fit_normalizer(self.p);m=Policy(O/'weights/r3m-resnet18-encoder.pt',n).eval();d=Sequences(self.p,'train',n,horizon=4,action_steps=3);b={k:v[None] for k,v in d[2].items()}
  with torch.no_grad():
   a=m.encode(b);z=m.encode(b);self.assertTrue(torch.equal(a,z));self.assertEqual(a.shape,(1,1102))
   out=m(torch.zeros(1,64,39),torch.tensor([1]),b);self.assertEqual(out.shape,(1,64,39));self.assertTrue(torch.isfinite(out).all())
  d.handle.close()
if __name__=='__main__':unittest.main()
