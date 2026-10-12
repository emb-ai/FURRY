"""Causal achieved-foot diagnostics; no phase labels or simulation mutations."""
import numpy as np
import mujoco

METRICS=['low_moving_foot_cost','weighted_clearance_deficit_m2','actual_sole_clearance_m',
         'actual_foot_speed_m_s','reference_foot_speed_m_s','foot_directed_progress_m_s',
         'moving_reference_foot_fraction','frozen_requested_foot_fraction','lower_joint_rmse']


class ClearanceRecorder:
    def __init__(self,env):
        self.env=env;self.actual=mujoco.MjData(env.model);self.reference=mujoco.MjData(env.model)
        self.geoms=[]
        for foot in env.feet:
            ids=[]
            for g in np.flatnonzero(env.model.geom_bodyid==foot):
                m=env.model;floor=env.floor
                if not ((m.geom_contype[g]&m.geom_conaffinity[floor]) or (m.geom_contype[floor]&m.geom_conaffinity[g])):continue
                if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE:raise ValueError('Expected sphere-only runtime sole')
                ids.append(g)
            if len(ids)!=4:raise ValueError('Expected four physical sole spheres')
            self.geoms.append(ids)
        self.geoms=np.asarray(self.geoms);self.radii=env.model.geom_size[self.geoms,0]

    def points(self):
        e=self.env;self.actual.qpos[:]=e.data.qpos
        q=e.motion['q'][e.k];self.reference.qpos[:7]=q[:7];self.reference.qpos[e.qa]=q[7:]
        mujoco.mj_kinematics(e.model,self.actual);mujoco.mj_kinematics(e.model,self.reference)
        return self.actual.geom_xpos[self.geoms].copy(),self.reference.geom_xpos[self.geoms].copy()

    def reset(self):
        a,r=self.points();self.history=[(a,r)];self.elapsed=0

    def record(self):
        a,r=self.points();pa,pr=self.history[0];span=len(self.history)*.01
        v=(a.mean(1)-pa.mean(1))/span;rv=(r.mean(1)-pr.mean(1))/span
        speed=np.linalg.norm(v[:,:2],axis=-1);rspeed=np.linalg.norm(rv[:,:2],axis=-1)
        h=(a[...,2]-self.radii).min(-1);terms=np.maximum(.065-h,0)**2*np.tanh(2*speed)
        cost=-np.expm1(-terms.sum()/.05)
        # Only the diagnostic freeze ratio uses a movement threshold. Reward
        # above has no threshold. Aggregate numerator/denominator before ratios.
        moving=rspeed>.1;frozen=moving&(speed<.03)
        directed=np.sum(v[:,:2]*rv[:,:2],axis=-1)/np.maximum(rspeed,1e-12)
        q=self.env.motion['q'][self.env.k];dq=self.env.data.qpos[self.env.qa]-q[7:]
        values=[cost,terms.mean(),h.mean(),speed.mean(),rspeed.mean(),directed.mean(),moving.mean(),frozen.mean(),np.sqrt(np.mean(dq[:12]**2))]
        self.history.append((a,r));self.history=self.history[-2:]
        return dict(zip(METRICS,map(float,values)))


def summarize_v2(comparison):
    """Use existing5% tracking/2% slipping tolerances, never old trip gates."""
    totals={}
    # Lower-leg vertical tracking can worsen when the desired correction lifts
    # a foot above a deficient reference. Keep all-body/joint/key error as
    # diagnostics; guard upper tracking, footXY/root progress and safety below.
    protected={'new_falls_on_baseline_successes','more_falls','less_survival','root_xy_error_m_regression','sliding_regression'}
    reasons=[r for r in comparison['reasons'] if r in protected];pairs=comparison['paired_gait']
    for label in ('baseline','candidate'):
        rows=[p[label] for p in pairs if not p[label].get('insufficient_duration')]
        seconds=sum(r['seconds'] for r in rows);path=sum(r['moving_reference_path_m'] for r in rows)
        m=comparison['matched_prefix_means'][label].copy()
        m['directed_root_progress_ratio']=sum(r['directed_progress_m'] for r in rows)/path if path>.1 else None
        for key in ('heading_delta_error_mean_deg','lateral_error_abs_mean_m'):
            m[key]=sum(r[key]*r['seconds'] for r in rows)/seconds if seconds else None
        m['directed_foot_progress_ratio']=m['foot_directed_progress_m_s']/m['reference_foot_speed_m_s'] if m['reference_foot_speed_m_s']>.01 else None
        m['frozen_foot_given_reference_motion_fraction']=m['frozen_requested_foot_fraction']/m['moving_reference_foot_fraction'] if m['moving_reference_foot_fraction']>0 else None
        totals[label]=m
    b,c=totals['baseline'],totals['candidate']
    tolerances=comparison['tolerances'];track=tolerances['tracking'];slip=tolerances['sliding']
    for key,tol in [('low_moving_foot_cost',slip),('window_displacement_error_m',track),
                    ('heading_delta_error_mean_deg',track),('lateral_error_abs_mean_m',track),
                    ('upper_joint_rmse',track),('upper_key_rmse_m',track)]:
        if b.get(key) is None or c.get(key) is None:reasons.append('undefined_'+key)
        elif c[key]>b[key]*(1+tol)+1e-4:reasons.append(key+'_regression')
    undefined=[]
    for key in ('directed_root_progress_ratio','directed_foot_progress_ratio'):
        if b[key] is None or c[key] is None:undefined.append(key)
        elif c[key]<b[key]-abs(b[key])*track-1e-4:reasons.append(key+'_regression')
    # A fall must never turn missing motion evidence into automatic acceptance.
    if undefined:reasons.append('insufficient_reference_motion_for_progress_check')
    improvement=any(c[k]<b[k]*(1-tolerances['minimum_improvement'])-1e-4 for k in ('low_moving_foot_cost','slide_rate'))
    reasons=sorted(set(reasons))
    return dict(accepted=not reasons and improvement,non_regressing=not reasons,improved=improvement,
        reasons=reasons,totals=totals,undefined_progress_metrics=undefined,legacy_accepted=comparison['accepted'],
        tolerances=tolerances,old_trip_counts_used_for_acceptance=False,
        leg_z_tracking_freedom='Whole-body joint/key and lower-joint errors are diagnostics; intended clearance correction may exceed low reference legZ. Training imitation rewards remain unchanged.',
        freeze_loophole='Actual-speed clearance may improve by stopping. Require directed root and foot progress plus tracking/survival guards.',
        purpose='Prospective diagnostic comparison, no PPO early stop or automatic deployment',
        holdout='Same previously used12Quest clips; not an independent test')
