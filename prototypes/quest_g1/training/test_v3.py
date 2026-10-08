import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from env import future_reference_features,FUTURE_SECONDS
from mirror_dataset import reflect
from ppo_utils import adaptive_actor_lr


class FutureTests(unittest.TestCase):
    def motion(self):
        q=np.zeros((250,36));q[:,2]=.8;q[:,3]=1;q[:,0]=np.arange(250)*.01
        cmd=np.zeros((250,35),np.float32);cmd[:,0]=1;cmd[:,2]=.8
        return {'q':q,'cmd':cmd}

    def test_future_change_is_visible_without_changing_present(self):
        m=self.motion();present=m['q'][0].copy();a=future_reference_features(m,0,present[:3],present[3:7])
        m['q'][190,0]+=1;b=future_reference_features(m,0,present[:3],present[3:7])
        np.testing.assert_array_equal(m['q'][0],present)
        self.assertGreater(np.linalg.norm(b-a),.99);self.assertEqual(len(a),20*42)
        self.assertAlmostEqual(FUTURE_SECONDS[-1],1.9)

    def test_end_padding_has_mask_and_zero_velocity(self):
        m=self.motion();m['cmd'][:,5]=2
        x=future_reference_features(m,248,m['q'][248,:3],m['q'][248,3:7]).reshape(20,42)
        self.assertTrue((x[:,-1]==0).all());self.assertTrue((x[:,[0,1,5]]==0).all())
        self.assertTrue(np.isfinite(x).all())

    def test_global_translation_does_not_change_relative_features(self):
        m=self.motion();root=m['q'][20,:3].copy();quat=m['q'][20,3:7]
        x=future_reference_features(m,20,root,quat);shift=np.array([3.,-2.,0.])
        m['q'][:,:3]+=shift
        np.testing.assert_allclose(future_reference_features(m,20,root+shift,quat),x,atol=1e-6)


class ScheduleAndReflectionTests(unittest.TestCase):
    def test_adaptive_lr_reduction_survives_next_iteration(self):
        reduced=adaptive_actor_lr(3e-6,.04,.008,3e-6)
        self.assertLess(reduced,3e-6)
        self.assertEqual(adaptive_actor_lr(reduced,.008,.008,4e-6),reduced)
        self.assertLessEqual(adaptive_actor_lr(3e-6,.001,.008,3e-6),3e-6)

    def test_mirror_is_involution_and_proper_rotation(self):
        rng=np.random.default_rng(12);q=rng.normal(size=(10,36))
        q[:,3:7]=Rotation.random(10,random_state=rng).as_quat()[:,[3,0,1,2]]
        np.testing.assert_array_equal(reflect(reflect(q)),q)
        S=np.diag([1.,-1.,1.]);r=reflect(q)
        np.testing.assert_allclose(Rotation.from_quat(r[:,[4,5,6,3]]).as_matrix(),S@Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix()@S,atol=1e-12)


try:
    from critic import mc_plan
except ImportError:
    mc_plan=None

@unittest.skipIf(mc_plan is None,'PyTorch required')
class CollectionTests(unittest.TestCase):
    def test_coverage_and_fixed_episode_seeds(self):
        motions=SimpleNamespace(items=list(range(37)))
        plan=mc_plan(motions,74,123)
        self.assertEqual(set(c for c,s in plan[:37]),set(range(37)))
        self.assertEqual(set(c for c,s in plan[37:]),set(range(37)))
        self.assertEqual(plan,mc_plan(motions,74,123));self.assertEqual(len(set(s for c,s in plan)),74)

class MirrorAdmissionTests(unittest.TestCase):
    def test_baseline_failures_do_not_remove_training_clips(self):
        import hashlib,json,sys,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import mirror_guard
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=root/'dataset';dataset.mkdir();initial=root/'initial.pt';initial.write_bytes(b'policy')
            np.savez(dataset/'mirror.npz',qpos=np.zeros((2,36)))
            original={'id':'original','split':'train'}
            mirrored={'id':'mirror','parent_id':'original','split':'train','augmentation':'sagittal_reflection','file':'mirror.npz','sha256':hashlib.sha256((dataset/'mirror.npz').read_bytes()).hexdigest()}
            clips=[original,mirrored];manifest={'clips':clips,'augmentation':{'checks':[{'parent_id':'original','max_fk_position_error_m':0.,'max_fk_rotation_matrix_error':0.}]}}
            (dataset/'manifest.json').write_text(json.dumps(manifest))
            baseline={'policy_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'columns':['joint_rmse','slide_rate'],'trials':[{'id':'original','fell':True}]}
            (root/'baseline.json').write_text(json.dumps(baseline));np.savez(root/'baseline.npz',original=np.full((100,2),.1))
            result={'trials':[{'clip':'mirror','seed':0,'trace_key':'mirror__seed0','fell':True}],'trace_columns':['joint_rmse','slide_rate'],'_traces':{'mirror__seed0':np.ones((100,2))}}
            args=['mirror_guard.py','--assets','fixture','--dataset',str(dataset),'--initial',str(initial),'--source','fixture','--baseline',str(root/'baseline.json'),'--output',str(root/'audit')]
            with patch.object(sys,'argv',args),patch.object(mirror_guard,'Motions',return_value=SimpleNamespace(items=[{'meta':mirrored}])),patch.object(mirror_guard,'evaluate',return_value=result):mirror_guard.main()
            self.assertEqual(json.loads((dataset/'manifest.json').read_text())['clips'],clips)
            audit=json.loads((root/'audit/mirror-audit.json').read_text())
            self.assertEqual(audit['included'],1)
            self.assertIn('mirrored_baseline_failed',audit['checks'][0]['diagnostic_flags'])


if __name__=='__main__':unittest.main()
