"""Causal labels for desired reference contact, independent of achieved forces.

This is an offline/training experiment, not enabled by the Quest runtime.
UNKNOWN is intentional: ground-level translation is not evidence of airborne
motion. Labels describe reference geometry, never measured human load shares.
"""
from dataclasses import asdict, dataclass
from enum import IntEnum
import numpy as np


class Phase(IntEnum):
    UNKNOWN = -1
    SWING = 0
    STANCE = 1


@dataclass(frozen=True)
class ContactPhaseConfig:
    stance_enter_height: float = .012
    stance_exit_height: float = .025
    swing_enter_height: float = .035
    swing_exit_height: float = .025
    stance_enter_speed: float = .10
    stance_exit_speed: float = .20
    stance_enter_vertical_speed: float = .12
    stance_exit_vertical_speed: float = .25
    stance_enter_yaw_rate: float = .52
    stance_exit_yaw_rate: float = .90
    stance_dwell_s: float = .08
    swing_dwell_s: float = .04
    max_gap_s: float = .06
    velocity_tau_s: float = .025
    support_band_m: float = .008
    max_reference_penetration_m: float = .01

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Contact thresholds must be finite and positive')
        if not self.stance_enter_height < self.stance_exit_height <= self.swing_enter_height:
            raise ValueError('Invalid contact-height hysteresis')
        if self.swing_exit_height >= self.swing_enter_height:
            raise ValueError('Invalid swing-height hysteresis')
        if self.stance_enter_speed >= self.stance_exit_speed:
            raise ValueError('Invalid speed hysteresis')


def label_reference_contacts(times, sole_points, radii=0., valid=None,
                             segment_ids=None, foot_yaw=None,
                             config=ContactPhaseConfig()):
    """Label two feet from world-space reference sole geometry, z-up/metres.

    `sole_points`: [N,2,P,3] sphere centres (capsule endpoints also work).
    `radii`: scalar, [P], or [2,P]; zero for explicit sole corner points.
    `valid`: [N] or [N,2] source confidence/validity gate supplied by caller.
    `segment_ids`: [N], MUST change across reset/calibration/clip boundaries.
    `foot_yaw`: optional [N,2] radians in a common world frame.
    Times are seconds. Inference is causal; invalid samples/gaps clear history.
    UNKNOWN has NaN desired_contact and confidence=0: callers must mask it.
    No actual contact/force is accepted, preventing achieved-contact leakage.
    """
    t=np.asarray(times,dtype=float); p=np.asarray(sole_points,dtype=float)
    if t.ndim!=1 or p.ndim!=4 or p.shape[:2]!=(len(t),2) or p.shape[-1]!=3 or p.shape[2]<1:
        raise ValueError('Expected times[N], sole_points[N,2,P,3]')
    if not np.isfinite(t).all():raise ValueError('Nonfinite time')
    radii=np.broadcast_to(np.asarray(radii,dtype=float),p.shape[1:3])
    if not np.isfinite(radii).all() or np.any(radii<0):raise ValueError('Invalid sole radii')
    good=np.ones((len(t),2),bool) if valid is None else np.asarray(valid,dtype=bool)
    if good.shape==(len(t),):good=np.repeat(good[:,None],2,axis=1)
    if good.shape!=(len(t),2):raise ValueError('Expected valid[N] or valid[N,2]')
    good=good.copy() & np.isfinite(p).all(axis=(2,3))
    segments=np.zeros(len(t),int) if segment_ids is None else np.asarray(segment_ids)
    if segments.shape!=(len(t),):raise ValueError('Expected segment_ids[N]')
    yaw=None if foot_yaw is None else np.asarray(foot_yaw,dtype=float)
    if yaw is not None:
        if yaw.shape!=(len(t),2):raise ValueError('Expected foot_yaw[N,2]')
        good &= np.isfinite(yaw)
    same=segments[1:]==segments[:-1]
    if np.any((np.diff(t)<=0)&same):raise ValueError('Non-increasing time within a segment')
    shape=(len(t),2)
    output={k:np.full(shape,np.nan) for k in ('desired_contact','clearance','horizontal_speed','vertical_speed','yaw_rate')}
    output['state']=np.full(shape,int(Phase.UNKNOWN),np.int8)
    output['confidence']=np.zeros(shape)
    states=np.full(2,int(Phase.UNKNOWN));candidate=states.copy();duration=np.zeros(2)
    smooth=np.zeros((2,3)); previous_good=np.zeros(2,bool)
    for n in range(len(t)):
        new=n==0 or segments[n]!=segments[n-1] or t[n]-t[n-1]>config.max_gap_s
        if new:states[:]=candidate[:]=int(Phase.UNKNOWN);duration[:]=0;previous_good[:]=False
        dt=0 if new else t[n]-t[n-1]
        for side in range(2):
            if not good[n,side]:
                states[side]=candidate[side]=int(Phase.UNKNOWN);duration[side]=0;previous_good[side]=False
                continue
            bottom=p[n,side,:,2]-radii[side];h=float(bottom.min());output['clearance'][n,side]=h
            if h < -config.max_reference_penetration_m:
                states[side]=candidate[side]=int(Phase.UNKNOWN);duration[side]=0;previous_good[side]=False
                continue
            if not previous_good[side]:previous_good[side]=True;continue
            velocity=(p[n,side]-p[n-1,side])/dt
            # Rigid rocking may move the ankle while the lowest contact stays
            # fixed. Evaluate the lowest sole candidates, not the ankle origin.
            low=bottom <= h+config.support_band_m
            horizontal=float(np.sqrt(np.mean(np.sum(velocity[low,:2]**2,axis=1))))
            vertical=float((h-output['clearance'][n-1,side])/dt)
            yaw_rate=0. if yaw is None else float(np.arctan2(np.sin(yaw[n,side]-yaw[n-1,side]),np.cos(yaw[n,side]-yaw[n-1,side]))/dt)
            values=np.array([horizontal,vertical,yaw_rate]);alpha=1-np.exp(-dt/config.velocity_tau_s)
            smooth[side]=values if n==1 or not np.isfinite(output['horizontal_speed'][n-1,side]) else smooth[side]+alpha*(values-smooth[side])
            speed,vz,wy=smooth[side]
            for key,value in zip(('horizontal_speed','vertical_speed','yaw_rate'),(speed,vz,wy)):output[key][n,side]=value
            stance=(h<=config.stance_enter_height and speed<=config.stance_enter_speed and abs(vz)<=config.stance_enter_vertical_speed and abs(wy)<=config.stance_enter_yaw_rate)
            hold_stance=(states[side]==Phase.STANCE and h<=config.stance_exit_height and speed<=config.stance_exit_speed and abs(vz)<=config.stance_exit_vertical_speed and abs(wy)<=config.stance_exit_yaw_rate)
            swing=h>=config.swing_enter_height
            hold_swing=states[side]==Phase.SWING and h>=config.swing_exit_height
            proposed=int(Phase.STANCE if stance or hold_stance else Phase.SWING if swing or hold_swing else Phase.UNKNOWN)
            if proposed==Phase.UNKNOWN:
                states[side]=candidate[side]=proposed;duration[side]=0;continue
            if proposed!=candidate[side]:candidate[side]=proposed;duration[side]=0
            duration[side]+=dt
            dwell=config.stance_dwell_s if proposed==Phase.STANCE else config.swing_dwell_s
            if proposed!=states[side] and duration[side]+1e-12<dwell:states[side]=int(Phase.UNKNOWN);continue
            states[side]=proposed
            # Confidence belongs to this geometric estimate, not human GRF.
            if proposed==Phase.STANCE:
                confidence=float(np.exp(-max(h,0)/config.stance_exit_height-speed/config.stance_exit_speed))
                probability=.5+.5*confidence
            else:
                confidence=float(np.clip((h-config.stance_enter_height)/(config.swing_enter_height-config.stance_enter_height),0,1))
                probability=.5*(1-confidence)
            output['state'][n,side]=proposed;output['confidence'][n,side]=confidence;output['desired_contact'][n,side]=probability
    output['known']=output['state']!=Phase.UNKNOWN
    return output


def mujoco_reference_sole_points(model, qpos, foot_names=('left_ankle_roll_link','right_ankle_roll_link')):
    """FK physical floor-contact geometry for reference qpos, no dynamics.

    Capsule endpoints/spheres/box corners yield exact flat-floor clearance.
    Only geoms colliding with named z=0 floor are included; self-contact boxes
    do not accidentally become sole geometry. Does not inspect actual forces.
    """
    import mujoco as mj
    q=np.asarray(qpos,dtype=float)
    if q.ndim!=2 or q.shape[1]!=model.nq:raise ValueError('Expected qpos[N,model.nq]')
    floor=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'floor')
    if floor<0 or model.geom_type[floor]!=mj.mjtGeom.mjGEOM_PLANE:raise ValueError('Named flat floor required')
    if np.linalg.norm(model.geom_pos[floor])>1e-9 or not np.allclose(model.geom_quat[floor],[1,0,0,0]):raise ValueError('Expected floor at z=0')
    geometry=[]
    for name in foot_names:
        body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
        if body<0:raise ValueError('Missing foot body '+name)
        pads=[]
        for g in range(model.ngeom):
            if model.geom_bodyid[g]!=body:continue
            if not ((model.geom_contype[g]&model.geom_conaffinity[floor]) or (model.geom_contype[floor]&model.geom_conaffinity[g])):continue
            kind=model.geom_type[g];size=model.geom_size[g]
            if kind==mj.mjtGeom.mjGEOM_SPHERE:local=[(np.zeros(3),size[0])]
            elif kind==mj.mjtGeom.mjGEOM_CAPSULE:local=[(np.array([0,0,z*size[1]]),size[0]) for z in (-1,1)]
            elif kind==mj.mjtGeom.mjGEOM_BOX:local=[(np.array([x,y,z])*size,0.) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
            else:raise ValueError('Unsupported sole geom type')
            pads.extend((g,point,radius) for point,radius in local)
        if not pads:raise ValueError('No foot-floor geometry')
        geometry.append((body,pads))
    count=max(len(pads) for _,pads in geometry)
    # Duplicate last point for asymmetric geometry; this preserves minima.
    for _,pads in geometry:
        while len(pads)<count:pads.append(pads[-1])
    points=np.zeros((len(q),2,count,3));yaw=np.zeros((len(q),2));radii=np.zeros((2,count));data=mj.MjData(model)
    for n,pose in enumerate(q):
        if not np.isfinite(pose).all():points[n]=np.nan;yaw[n]=np.nan;continue
        data.qpos[:]=pose;mj.mj_kinematics(model,data)
        for side,(body,pads) in enumerate(geometry):
            R=data.xmat[body].reshape(3,3);yaw[n,side]=np.arctan2(R[1,0],R[0,0])
            for k,(g,local,radius) in enumerate(pads):
                points[n,side,k]=data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3).dot(local);radii[side,k]=radius
    return dict(sole_points=points,radii=radii,foot_yaw=yaw,
                geometry=[dict(foot=foot_names[s],geom_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g) for g,_,_ in pads]) for s,(_,pads) in enumerate(geometry)])
