"""Independent verdict counterexamples; no policy training or evaluation."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent;sys.path.insert(0,str(SOURCE))
from clearance_evaluation import summarize_v2

metrics=dict(low_moving_foot_cost=.06,slide_rate=.3,window_displacement_error_m=.15,
    root_xy_error_m=.15,upper_joint_rmse=.1,upper_key_rmse_m=.05,lower_joint_rmse=.1,
    joint_rmse=.1,key_rmse_m=.1,foot_directed_progress_m_s=.5,reference_foot_speed_m_s=.5,
    moving_reference_foot_fraction=.5,frozen_requested_foot_fraction=.01)
gait=dict(seconds=10.,moving_reference_path_m=1.,directed_progress_m=1.,
          heading_delta_error_mean_deg=3.,lateral_error_abs_mean_m=.04)
base=dict(matched_prefix_means=dict(baseline=deepcopy(metrics),candidate=deepcopy(metrics)),
    paired_gait=[dict(baseline=deepcopy(gait),candidate=deepcopy(gait))],
    reasons=[],new_falls=[],accepted=True,tolerances=dict(tracking=.05,sliding=.02,minimum_improvement=.02))
base['matched_prefix_means']['candidate']['low_moving_foot_cost']=.03
base['matched_prefix_means']['candidate']['slide_rate']=.29
cases={}
def check(name,c,accepted):
    verdict=summarize_v2(c);cases[name]=verdict
    assert verdict['accepted']==accepted,(name,verdict)

check('clearance_improvement_with_progress_preserved',deepcopy(base),True)
c=deepcopy(base);c['matched_prefix_means']['candidate'].update(low_moving_foot_cost=0.,slide_rate=.01,foot_directed_progress_m_s=0.,frozen_requested_foot_fraction=.5)
c['paired_gait'][0]['candidate']['directed_progress_m']=0.
check('frozen_robot_cannot_win_by_low_cost_and_slip',c,False)
assert any('progress' in reason for reason in cases['frozen_robot_cannot_win_by_low_cost_and_slip']['reasons'])
c=deepcopy(base);c['matched_prefix_means']['candidate']['foot_directed_progress_m_s']=0.
check('root_moves_but_feet_do_not_follow_requested_xy',c,False)
c=deepcopy(base);c['new_falls']=['clip_seed0'];c['reasons']=['new_falls_on_baseline_successes']
check('new_fall_cannot_win',c,False)
c=deepcopy(base);c['matched_prefix_means']['candidate']=deepcopy(metrics)
check('equal_metrics_are_not_improvement',c,False)
c=deepcopy(base);c['matched_prefix_means']['candidate'].update(lower_joint_rmse=.5,joint_rmse=.4,key_rmse_m=.3)
c['reasons']=['joint_rmse_regression','key_rmse_m_regression'];c['accepted']=False
check('leg_reference_departure_alone_does_not_veto_lift',c,True)
c=deepcopy(base);c['matched_prefix_means']['candidate']['upper_joint_rmse']=.2
check('upper_body_regression_is_still_rejected',c,False)
c=deepcopy(base);c['reasons']=['root_xy_error_m_regression'];c['matched_prefix_means']['candidate']['root_xy_error_m']=.3
check('root_tracking_regression_is_still_rejected',c,False)
c=deepcopy(base)
for label in ('baseline','candidate'):
    c['paired_gait'][0][label].update(moving_reference_path_m=0.,directed_progress_m=0.)
    c['matched_prefix_means'][label].update(reference_foot_speed_m_s=0.,foot_directed_progress_m_s=0.)
check('no_reference_motion_cannot_prove_progress',c,False)
c=deepcopy(base);c['matched_prefix_means']['candidate']['foot_directed_progress_m_s']=-.5
check('wrong_direction_cannot_win',c,False)
result=dict(status='PASS',cases={name:dict(accepted=v['accepted'],reasons=v['reasons']) for name,v in cases.items()},
            source_sha256=hashlib.sha256((SOURCE/'clearance_evaluation.py').read_bytes()).hexdigest(),
            scope='Synthetic verdict contracts only; no trained-policy improvement claim')
(HERE/'verdict-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
