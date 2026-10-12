"""Offline reference feasibility experiment; not enabled by the runtime.

This solves a constrained retargeting subproblem. It does not overwrite named
motor angles with rules, infer actual loads, or promise dynamically stable gait.
Root XY/orientation and upper body remain the supplied reference; root height
and leg joints are optimized under their limits and physical sole constraints.
At most one reference support point may be anchored at any instant.
"""
from dataclasses import dataclass
import numpy as np
import mujoco as mj
from scipy.optimize import minimize


@dataclass(frozen=True)
class ContactProjectionConfig:
    root_height_scale_m: float = .02
    joint_scale_rad: float = .15
    max_root_height_delta_m: float = .04
    max_joint_delta_rad: float = .18
    max_anchor_drift_m: float = .04
    support_band_m: float = .008
    feasibility_tolerance_m: float = 1e-6
    max_iterations: int = 60

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('Projection settings must be positive and finite')


class SingleSupportProjection:
    """Causal projection with explicit UNKNOWN/release and one support anchor.

    `phase` uses -1 UNKNOWN, 0 SWING, 1 STANCE, from independent source labels.
    Reset on every source/reset/calibration boundary. Reference support that
    moves farther than the configured drift budget releases for this frame;
    the *next* eligible frame may establish a new anchor. This is an explicit
    experimental feasibility bound, not a learned corrective-step controller.
    No actual robot contact or catch-up error enters reference phase inference.
    """
    def __init__(self, model, config=ContactProjectionConfig(), joint_limits=None):
        self.model=model;self.config=config;self.data=mj.MjData(model)
        root=np.flatnonzero(model.jnt_type==mj.mjtJoint.mjJNT_FREE)
        if len(root)!=1 or model.jnt_qposadr[root[0]]!=0:
            raise ValueError('Expected one root free joint at qpos0')
        names=[f'{side}_{part}_joint' for side in ('left','right')
               for part in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        ids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in names]
        if min(ids)<0:raise ValueError('Missing leg joint')
        self.qadr=np.r_[2,model.jnt_qposadr[ids]].astype(int)
        self.vadr=np.r_[2,model.jnt_dofadr[ids]].astype(int)
        self.limits=model.jnt_range[ids].copy()
        if joint_limits is not None:
            limits=np.asarray(joint_limits,dtype=float)
            if limits.shape!=(12,2):raise ValueError('Expected joint_limits[12,2]')
            self.limits[:,0]=np.maximum(self.limits[:,0],limits[:,0])
            self.limits[:,1]=np.minimum(self.limits[:,1],limits[:,1])
        if np.any(self.limits[:,0]>=self.limits[:,1]):raise ValueError('Empty joint limits')
        floor=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'floor')
        if floor<0 or model.geom_type[floor]!=mj.mjtGeom.mjGEOM_PLANE or np.linalg.norm(model.geom_pos[floor])>1e-9 or not np.allclose(model.geom_quat[floor],[1,0,0,0]):
            raise ValueError('Requires flat z=0 floor')
        self.bodies=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
        self.pads=[]
        for side,body in enumerate(self.bodies):
            if body<0:raise ValueError('Missing foot body')
            for g in range(model.ngeom):
                if model.geom_bodyid[g]!=body:continue
                if not ((model.geom_contype[g]&model.geom_conaffinity[floor]) or (model.geom_contype[floor]&model.geom_conaffinity[g])):continue
                if model.geom_type[g]!=mj.mjtGeom.mjGEOM_SPHERE:raise ValueError('This narrow experiment requires spherical sole pads')
                self.pads.append((side,g,body,float(model.geom_size[g,0])))
        if set(s for s,_,_,_ in self.pads)!={0,1}:raise ValueError('Missing sole spheres')
        self.scales=np.r_[config.root_height_scale_m,np.full(12,config.joint_scale_rad)]
        self.reset()

    def reset(self):
        self.anchor_side=-1;self.anchor_local=None;self.anchor_xy=None

    def _fk(self,qpos):
        self.data.qpos[:]=qpos
        mj.mj_kinematics(self.model,self.data);mj.mj_comPos(self.model,self.data)

    def _anchor_world(self):
        body=self.bodies[self.anchor_side]
        return self.data.xpos[body]+self.data.xmat[body].reshape(3,3).dot(self.anchor_local)

    def _floor_clearance(self):
        return np.array([self.data.geom_xpos[g,2]-radius for _,g,_,radius in self.pads])

    def _point_jacobian(self,point,body):
        jac=np.zeros((3,self.model.nv));mj.mj_jac(self.model,self.data,jac,None,point,body)
        return jac[:,self.vadr]

    def project(self, reference_qpos, phase, confidence=None, reset=False):
        q=np.asarray(reference_qpos,dtype=float)
        phase=np.asarray(phase,dtype=int)
        if q.shape!=(self.model.nq,) or not np.isfinite(q).all() or phase.shape!=(2,) or not np.isin(phase,[-1,0,1]).all():
            raise ValueError('Invalid reference pose/phase')
        confidence=np.ones(2) if confidence is None else np.asarray(confidence,dtype=float)
        if confidence.shape!=(2,) or not np.isfinite(confidence).all():raise ValueError('Invalid reference confidence')
        if reset:self.reset()
        self._fk(q);before=self._floor_clearance();release='';blocked=False
        if self.anchor_side>=0 and phase[self.anchor_side]!=1:
            release='reference_phase';self.reset()
        if self.anchor_side>=0:
            drift=np.linalg.norm(self._anchor_world()[:2]-self.anchor_xy)
            if drift>self.config.max_anchor_drift_m:
                release='reference_drift';self.reset();blocked=True
        if self.anchor_side<0 and not blocked:
            choices=np.flatnonzero(phase==1)
            if len(choices):
                side=int(choices[np.argmax(confidence[choices])]);body=self.bodies[side]
                pads=[(g,r) for s,g,_,r in self.pads if s==side]
                bottom=np.array([self.data.geom_xpos[g,2]-r for g,r in pads])
                low=bottom<=bottom.min()+self.config.support_band_m
                point=np.mean([self.data.geom_xpos[g] for (g,_),keep in zip(pads,low) if keep],axis=0)
                self.anchor_side=side
                self.anchor_local=self.data.xmat[body].reshape(3,3).T.dot(point-self.data.xpos[body])
                self.anchor_xy=point[:2].copy()
        anchor_side=self.anchor_side
        anchor_xy=np.full(2,np.nan) if anchor_side<0 else self.anchor_xy.copy()
        lo=np.r_[-self.config.max_root_height_delta_m,np.maximum(-self.config.max_joint_delta_rad,self.limits[:,0]-q[self.qadr[1:]])]
        hi=np.r_[self.config.max_root_height_delta_m,np.minimum(self.config.max_joint_delta_rad,self.limits[:,1]-q[self.qadr[1:]])]
        if np.any(lo>hi):raise ValueError('Input reference farther outside limits than allowed correction')
        cache=None
        def evaluate(delta):
            nonlocal cache
            if cache is None or not np.array_equal(cache,delta):
                pose=q.copy();pose[self.qadr]+=delta;self._fk(pose);cache=delta.copy()
        def floor(delta):evaluate(delta);return self._floor_clearance()
        def floor_jac(delta):
            evaluate(delta)
            return np.array([self._point_jacobian(self.data.geom_xpos[g],body)[2] for _,g,body,_ in self.pads])
        def anchor(delta):evaluate(delta);return self._anchor_world()[:2]-self.anchor_xy
        def anchor_jac(delta):evaluate(delta);return self._point_jacobian(self._anchor_world(),self.bodies[self.anchor_side])[:2]
        def solve(use_anchor):
            constraints=[dict(type='ineq',fun=floor,jac=floor_jac)]
            if use_anchor:constraints.append(dict(type='eq',fun=anchor,jac=anchor_jac))
            x0=np.clip(np.zeros(13),lo,hi);x0[0]=np.clip(max(0.,-before.min()),lo[0],hi[0])
            return minimize(lambda d:.5*np.sum((d/self.scales)**2),x0,
                            jac=lambda d:d/self.scales**2,bounds=list(zip(lo,hi)),constraints=constraints,
                            method='SLSQP',options=dict(maxiter=self.config.max_iterations,ftol=1e-11))
        result=solve(self.anchor_side>=0);evaluate(result.x)
        tolerance=self.config.feasibility_tolerance_m
        error=0. if self.anchor_side<0 else float(np.linalg.norm(anchor(result.x)))
        feasible=floor(result.x).min()>=-tolerance and error<=tolerance
        if not feasible and self.anchor_side>=0:
            self.reset();release='anchor_infeasible';anchor_side=-1;anchor_xy[:]=np.nan
            result=solve(False);evaluate(result.x);error=0.;feasible=floor(result.x).min()>=-tolerance
        pose=q.copy()
        if feasible:pose[self.qadr]+=result.x
        else:self.reset();anchor_side=-1;anchor_xy[:]=np.nan;release='floor_infeasible';self._fk(pose)
        return dict(qpos=pose,feasible=bool(feasible),optimizer_success=bool(result.success),
                    optimizer_message=str(result.message),iterations=int(result.nit),
                    anchor_side=anchor_side,anchor_xy=anchor_xy,release=release,
                    anchor_error_m=error,min_clearance_before_m=float(before.min()),
                    min_clearance_after_m=float(self._floor_clearance().min()),
                    root_height_delta_m=float(pose[2]-q[2]),joint_delta_rad=pose[self.qadr[1:]]-q[self.qadr[1:]])
