"""Guard against flattering scores from censored or unavailable references."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.summarize_gait_ablation import align_pair, orientation_rms_deg, summarize


class AblationTests(unittest.TestCase):
    @staticmethod
    def times(segments, values):
        a=np.zeros(len(values),dtype=[('segment',float),('wall_s',float)])
        a['segment']=segments;a['wall_s']=values
        return a

    def test_either_rollout_can_end_first(self):
        short=self.times([1,1],[0,.01])
        long=self.times([1,1,1],[0,.01,.02])
        for left,right in [(short,long),(long,short)]:
            ai,bi=align_pair(left,right)
            np.testing.assert_array_equal(left['wall_s'][ai],[0,.01])
            np.testing.assert_array_equal(right['wall_s'][bi],[0,.01])

    def test_time_matching_never_crosses_reset_segments(self):
        a=self.times([1,1,2,2],[0,.01,0,.01])
        b=self.times([1,1,2,2],[0,.01,0,.02])
        ai,bi=align_pair(a,b)
        np.testing.assert_array_equal(ai,[0,1,2])
        np.testing.assert_array_equal(a['segment'][ai],b['segment'][bi])
        malformed=self.times([1,1],[.01,0])
        with self.assertRaises(ValueError):align_pair(malformed,b)

    def test_wrist_orientation_metric_detects_wrong_angle(self):
        a=Rotation.identity(2).as_matrix().reshape(1,2,3,3)
        b=Rotation.from_euler('z',[90,90],degrees=True).as_matrix().reshape(1,2,3,3)
        self.assertAlmostEqual(orientation_rms_deg(a,b),90,places=8)
        self.assertAlmostEqual(orientation_rms_deg(a,a),0,places=8)

    def test_saved_commands_do_not_fabricate_full_pose_reference(self):
        fields=['segment','wall_s','sim_s','tilt','height','contacts','ref_available']
        fields += ['q'+str(i) for i in range(36)]
        a=np.zeros(3,dtype=[(name,float) for name in fields])
        a['segment']=1;a['wall_s']=a['sim_s']=[0,.01,.02]
        a['height']=a['q2']=.8;a['q3']=1
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'saved.csv'
            np.savetxt(p,np.column_stack([a[n] for n in fields]),delimiter=',',header=','.join(fields),comments='')
            p.with_suffix('.log').write_text('segment=1 fell=0 minz=0.8\n')
            result=summarize(p,p,None)
            self.assertEqual(result['samples'],0)
            self.assertEqual(result['falls'],0)
            self.assertIn('full GMR',result['failure'])
