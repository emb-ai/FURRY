import copy
import json
from pathlib import Path
import unittest
from clearance_evaluation import summarize_v2


def baseline():
    metrics=dict(low_moving_foot_cost=.1,slide_rate=.1,window_displacement_error_m=.1,
        upper_joint_rmse=.1,upper_key_rmse_m=.1,lower_joint_rmse=.1,joint_rmse=.1,key_rmse_m=.1,
        foot_directed_progress_m_s=1.,reference_foot_speed_m_s=1.,moving_reference_foot_fraction=.5,
        frozen_requested_foot_fraction=0.)
    gait=dict(seconds=10.,moving_reference_path_m=10.,directed_progress_m=10.,
        heading_delta_error_mean_deg=2.,lateral_error_abs_mean_m=.1)
    result=dict(reasons=[],new_falls=[],accepted=True,tolerances=dict(tracking=.05,sliding=.02,minimum_improvement=.02),
        matched_prefix_means=dict(baseline=metrics,candidate=copy.deepcopy(metrics)),
        paired_gait=[dict(baseline=gait,candidate=copy.deepcopy(gait))])
    result['matched_prefix_means']['candidate']['low_moving_foot_cost']=.07
    return result


class VerdictTests(unittest.TestCase):
    def test_raised_legs_may_exceed_bad_reference(self):
        x=baseline();x['reasons']=['joint_rmse_regression','key_rmse_m_regression']
        for k in ['joint_rmse','lower_joint_rmse','key_rmse_m']:x['matched_prefix_means']['candidate'][k]=.4
        self.assertTrue(summarize_v2(x)['accepted'])
    def test_frozen_reward_escape_rejected(self):
        x=baseline();m=x['matched_prefix_means']['candidate'];m['low_moving_foot_cost']=0.;m['foot_directed_progress_m_s']=0.
        x['paired_gait'][0]['candidate']['directed_progress_m']=0.
        self.assertFalse(summarize_v2(x)['accepted'])
    def test_new_fall_rejected(self):
        x=baseline();x['reasons']=['new_falls_on_baseline_successes'];x['new_falls']=['clip']
        self.assertFalse(summarize_v2(x)['accepted'])
    def test_slowdown_rejected(self):
        x=baseline();x['matched_prefix_means']['candidate']['foot_directed_progress_m_s']=.8
        self.assertFalse(summarize_v2(x)['accepted'])
    def test_no_motion_is_explicitly_undefined(self):
        x=baseline()
        for side in ['baseline','candidate']:
            x['matched_prefix_means'][side]['reference_foot_speed_m_s']=0
            x['paired_gait'][0][side]['moving_reference_path_m']=0
        v=summarize_v2(x);self.assertFalse(v['accepted']);self.assertEqual(len(v['undefined_progress_metrics']),2)


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VerdictTests))
    Path(__file__).with_name('verdict-tests.json').write_text(json.dumps(dict(tests=r.testsRun,success=r.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(not r.wasSuccessful())
