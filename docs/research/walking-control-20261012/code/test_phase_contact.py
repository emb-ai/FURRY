"""CPU oracle checks; optional real clips via --contact-dir and --oracle."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import torch
from phase_contact import ReferencePhase, SoleKinematics, contact_terms, qrot

ARGS = None

class PhaseTests(unittest.TestCase):
    def test_phase_real_oracle_parity(self):
        if not ARGS.oracle or not ARGS.contact_dir:self.skipTest('Supply real oracle files')
        spec=importlib.util.spec_from_file_location('reference_oracle',ARGS.oracle)
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        for path in sorted(ARGS.contact_dir.glob('*-contact.npz')):
            z=np.load(path);reference=mod.label_reference_contacts(z['times'],z['sole_points'],z['radii'],segment_ids=z['segment_ids'],foot_yaw=z['foot_yaw'])
            for dtype in (torch.float64,torch.float32):
                phase=ReferencePhase(1,z['sole_points'].shape[2],'cpu',dtype=dtype)
                states=[];conf=[]
                tensor=lambda x:torch.tensor(x,dtype=dtype)
                for i,t in enumerate(z['times']):
                    if i and z['segment_ids'][i]!=z['segment_ids'][i-1]:phase.reset([0])
                    s,c=phase.update(tensor([t]),tensor(z['sole_points'][i:i+1]),tensor(z['radii']),tensor(z['foot_yaw'][i:i+1]))
                    states.append(s[0].numpy().copy());conf.append(c[0].numpy().copy())
                np.testing.assert_array_equal(states,reference['state'],err_msg=str(dtype)+str(path))
                np.testing.assert_allclose(conf,reference['confidence'],atol=2e-12 if dtype==torch.float64 else 3e-5,rtol=1e-10 if dtype==torch.float64 else 3e-5)

    def test_invalid_ground_sliding_and_causal_dwell(self):
        phase=ReferencePhase(1,1,'cpu',dtype=torch.float64)
        p=torch.zeros((1,2,1,3),dtype=torch.float64);r=torch.zeros((2,1),dtype=torch.float64);y=torch.zeros((1,2),dtype=torch.float64)
        seen=[]
        for i in range(10):
            p[0,1,0,0]=i*.02 # 1m/s along ground is UNKNOWN, not SWING.
            s,_=phase.update(torch.tensor([i*.02]),p,r,y);seen.append(s.clone())
        self.assertTrue(all(x[0,1]==-1 for x in seen));self.assertEqual(seen[3][0,0],-1);self.assertEqual(seen[4][0,0],1)
        s,_=phase.update(torch.tensor([.2]),p,r,y,torch.tensor([[False,False]]));self.assertTrue((s==-1).all())
        p[...,2]=.05
        s,_=phase.update(torch.tensor([.22]),p,r,y);self.assertTrue((s==-1).all())
        for t in [.24,.26]:s,_=phase.update(torch.tensor([t]),p,r,y)
        self.assertTrue((s==0).all())
        s,_=phase.update(torch.tensor([.5]),p,r,y);self.assertTrue((s==-1).all())

    def test_fk_matches_mujoco(self):
        import mujoco as mj
        tree=ET.parse(ARGS.urdf);root=tree.getroot()
        names=[j.get('name') for j in root.findall('joint') if j.get('type') in ('revolute','continuous')]
        kin=SoleKinematics(ARGS.urdf,names,dtype=torch.float64)
        for link in root.findall('link'):
            for element in list(link):
                if element.tag=='visual' or (element.tag=='collision' and element.find('geometry/mesh') is not None):link.remove(element)
        model=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mj.MjData(model)
        rng=np.random.default_rng(19)
        for _ in range(20):
            angles=rng.uniform(-.2,.2,len(names));rootpos=rng.normal(size=(1,3));quat=rng.normal(size=(1,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
            for j,name in enumerate(names):
                ji=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name);data.qpos[model.jnt_qposadr[ji]]=angles[j]
            mj.mj_kinematics(model,data)
            actual,_=kin.reference(torch.tensor(rootpos),torch.tensor(quat),torch.tensor(angles[None]))
            expected=[]
            for side,name in enumerate(kin.foot_names):
                b=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
                local=data.xpos[b]+kin.points[side].numpy()@data.xmat[b].reshape(3,3).T
                expected.append(rootpos+qrot(torch.tensor(quat),torch.tensor(local)).numpy())
            np.testing.assert_allclose(actual[0].numpy(),expected,atol=1e-10)

    def _terms(self,state,z=0,force=400,translation=.02,omega=0):
        p=torch.zeros((1,2,1,3));p[...,2]=z+.01;prev=p.clone();prev[...,0]-=translation
        w=torch.zeros((1,2,3));w[...,1]=omega
        return contact_terms(torch.tensor([state]),torch.ones((1,2)),torch.full((1,2),.05),p,prev,torch.full((2,1),.01),w,torch.full((1,2),float(force)),.02,torch.tensor([True]),torch.tensor([[800.]]))

    def test_reward_phase_no_achieved_contact_leakage(self):
        self.assertAlmostEqual(float(self._terms([-1,-1])[0]),2.)
        self.assertTrue(all(float(x)==0 for x in self._terms([-1,-1])[1:]))
        slip,early,clear=self._terms([1,1]);self.assertAlmostEqual(float(slip),2.);self.assertEqual(float(early),0)
        slip,early,clear=self._terms([0,0]);self.assertAlmostEqual(float(slip),2.);self.assertGreater(float(early),0);self.assertGreater(float(clear),0)

    def test_rotation_material_point_and_task_contact(self):
        # Centres translate +1m/s; omega_y=100rad/s makes bottom-point -1m/s.
        self.assertAlmostEqual(float(self._terms([1,1],omega=100)[0]),0.,places=6)
        self.assertEqual(float(self._terms([0,0],z=.10,force=400)[1]),0.)
        self.assertEqual(float(self._terms([0,0],z=.10,force=400)[2]),0.)
        self.assertEqual(float(self._terms([1,1],force=0)[0]),0.)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--urdf',type=Path,required=True);p.add_argument('--oracle',type=Path);p.add_argument('--contact-dir',type=Path);p.add_argument('--report',type=Path)
    ARGS,rest=p.parse_known_args();torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhaseTests))
    if ARGS.report:ARGS.report.write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),success=result.wasSuccessful(),torch=torch.__version__,numpy=np.__version__),indent=2))
    raise SystemExit(not result.wasSuccessful())
