"""Read-only gait diagnostics at 100 Hz, independent of actor control frequency.

Swing labels come from fixed dataset qpos, not from the simulated actor. This
is not numerically interchangeable with the Meta/GMR replay study. No reward
or training sample weights are changed here.
"""
import numpy as np
import mujoco

SOLE = np.array([[-.05,.025,-.035],[-.05,-.025,-.035],[.12,.03,-.035],[.12,-.03,-.035]])
RAW_COLUMNS = ['gait_robot_x','gait_robot_y','gait_ref_x','gait_ref_y','gait_robot_yaw','gait_ref_yaw'] + [
    'gait_'+name+'_'+side for side in ('l','r') for name in ('ref_height','ref_speed','height','force','brake')]


def yaw_wxyz(q):
    w,x,y,z=np.asarray(q)
    return np.arctan2(2*(w*z+x*y),w*w+x*x-y*y-z*z)


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


class GaitRecorder:
    def __init__(self, env):
        self.env=env
        self.cache={}

    def reset(self):
        e=self.env
        if e.clip not in self.cache:
            data=mujoco.MjData(e.model); q=e.motion['q']
            low=np.zeros((len(q),2));xy=np.zeros((len(q),2,2))
            for i,row in enumerate(q):
                data.qpos[:7]=row[:7];data.qpos[e.qa]=row[7:]
                mujoco.mj_kinematics(e.model,data)
                for side,body in enumerate(e.feet):
                    low[i,side]=(data.xpos[body]+SOLE@data.xmat[body].reshape(3,3).T)[:,2].min()
                    xy[i,side]=data.xpos[body,:2]
            speed=np.zeros_like(low);span=5
            speed[span:-span]=np.linalg.norm(xy[2*span:]-xy[:-2*span],axis=2)/(2*span*.01)
            self.cache[e.clip]=(low-low.min(axis=1,keepdims=True),speed)
        self.ref_height,self.ref_speed=self.cache[e.clip]

    def record(self):
        e=self.env; d=e.data; m=e.model;q=e.motion['q'][e.k]
        values=list(d.qpos[:2])+list(q[:2])+[yaw_wxyz(d.qpos[3:7]),yaw_wxyz(q[3:7])]
        forces=np.zeros((2,3));f=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            if e.floor not in (g1,g2):continue
            body=m.geom_bodyid[g2 if g1==e.floor else g1]
            sides=np.flatnonzero(e.feet==body)
            if len(sides):
                mujoco.mj_contactForce(m,d,i,f)
                forces[sides[0]]+=(c.frame.reshape(3,3).T@f[:3])*(1 if g1==e.floor else -1)
        for side,body in enumerate(e.feet):
            height=(d.xpos[body]+SOLE@d.xmat[body].reshape(3,3).T)[:,2].min()
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,int(body),velocity,0)
            xy=velocity[3:5];speed=np.linalg.norm(xy)
            brake=max(0.,-float(forces[side,:2]@xy)/speed) if speed>1e-8 else 0.
            values.extend([self.ref_height[e.k,side],self.ref_speed[e.k,side],height,abs(forces[side,2]),brake])
        return dict(zip(RAW_COLUMNS,values))


def runtime_env_class(base, control_hz=100, gait_metrics=False):
    if control_hz not in (50,100):raise ValueError('Only 50 and 100 Hz are supported')
    class RuntimeEnv(base):
        policy_stride=100//control_hz
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.gait=GaitRecorder(self) if gait_metrics else None
        def observation(self):
            # The intermediate 10 ms sample must not enter actor history at 50 Hz.
            if self.steps%self.policy_stride:return self.last_obs
            return super().observation()
        def reset(self,*args,**kwargs):
            obs=super().reset(*args,**kwargs)
            if self.gait:self.gait.reset()
            return obs
        def step(self,action):
            obs,reward,done,info=super().step(action)
            if self.gait:info.update(self.gait.record())
            return obs,reward,done,info
    return RuntimeEnv


def summarize_gait(trace,columns,dt=.01,warmup=1.):
    """Recompute on a paired prefix; do not compare different survival lengths.

    Event thresholds match main's study: >=30 ms unloaded (<5 N), landing
    >=50 N, reference height >1 cm, reference speed >0.3 m/s, braking >50 N.
    Count only complete >=80 ms reference swings after warmup. Path/progress
    are sampled at 0.5 s; progress projects robot displacement on each reference
    segment. Stationary reference segments do not contribute to its ratio.
    """
    start=int(round(warmup/dt))
    if len(trace)<=start:return {'seconds':0.,'insufficient_duration':True}
    def col(name):return trace[:,columns.index('gait_'+name)]
    seconds=(len(trace)-start)*dt
    trips=braking=0; peaks=[];floor_time=0; swing_time=0
    for side in ('l','r'):
        height,speed,actual,force,brake=[col(n+'_'+side) for n in ('ref_height','ref_speed','height','force','brake')]
        swing=(height>.01)&(speed>.3)
        unloaded=0
        for i,f in enumerate(force):
            if f<5:unloaded+=1;continue
            if i>=start and unloaded*dt>=.03-1e-9 and f>=50 and swing[i]:
                trips+=1;braking+=int(brake[i]>50)
            unloaded=0
        edges=np.flatnonzero(np.diff(np.r_[False,swing,False].astype(int)))
        for a,b in zip(edges[::2],edges[1::2]):
            if a<start or b==len(trace) or (b-a)*dt<.08-1e-9:continue
            peaks.append(float(actual[a:b].max()))
            floor_time+=int((force[a:b]>5).sum());swing_time+=b-a
    xy=np.c_[col('robot_x'),col('robot_y')];ref=np.c_[col('ref_x'),col('ref_y')]
    yaw=col('robot_yaw');refyaw=col('ref_yaw')
    delta_yaw=wrap((yaw-yaw[start])-(refyaw-refyaw[start]))[start:]
    # Decompose accumulated displacement error in the local reference tangent.
    direction=np.gradient(ref,dt,axis=0);speed=np.linalg.norm(direction,axis=1)
    tangent=np.c_[np.cos(refyaw),np.sin(refyaw)]
    moving=speed>.05;tangent[moving]=direction[moving]/speed[moving,None]
    error=(xy-xy[start])-(ref-ref[start])
    along=(error*tangent).sum(1)[start:]
    lateral=(error[:,0]*(-tangent[:,1])+error[:,1]*tangent[:,0])[start:]
    stride=round(.5/dt);indices=np.arange(start,len(trace),stride)
    if len(indices) and indices[-1]!=len(trace)-1:indices=np.r_[indices,len(trace)-1]
    rd=np.diff(ref[indices],axis=0);ad=np.diff(xy[indices],axis=0);length=np.linalg.norm(rd,axis=1)
    elapsed=np.diff(indices)*dt;valid=length>.05*elapsed
    progress=float(np.sum(np.sum(ad[valid]*rd[valid],axis=1)/length[valid]))
    moving_path=float(length[valid].sum());ref_path=float(length.sum());actual_path=float(np.linalg.norm(ad,axis=1).sum())
    return {'seconds':seconds,'trips':trips,'braking_trips':braking,'trips_per_s':trips/seconds,
            'braking_trips_per_s':braking/seconds,'complete_swings':len(peaks),
            'swing_peak_median_m':float(np.median(peaks)) if peaks else None,
            'swings_below_1cm_fraction':float(np.mean(np.array(peaks)<.01)) if peaks else None,
            'swing_floor_contact_fraction':floor_time/swing_time if swing_time else None,
            'heading_error_mean_deg':float(np.degrees(np.abs(wrap(yaw-refyaw)[start:])).mean()),
            'heading_delta_error_mean_deg':float(np.degrees(np.abs(delta_yaw)).mean()),
            'heading_delta_error_final_deg':float(np.degrees(abs(delta_yaw[-1]))),
            'along_error_mean_m':float(along.mean()),'along_error_abs_mean_m':float(abs(along).mean()),
            'lateral_error_abs_mean_m':float(abs(lateral).mean()),
            'directed_progress_m':progress,'moving_reference_path_m':moving_path,
            'directed_progress_ratio':progress/moving_path if moving_path>.1 else None,
            'robot_path_m':actual_path,'reference_path_m':ref_path,
            'path_length_ratio':actual_path/ref_path if ref_path>.1 else None}
