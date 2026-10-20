"""Shared causal state encoder for recorded and live native MuJoCo states."""
import numpy as np
import mujoco

SCHEMA = 'cup-state-homie-v1'
ACTION_DIM = 37  # body35 plus two original scalar grips, no gaze
HISTORY = 8
HORIZON = 16
EXECUTE = 4

class Encoder:
    def __init__(self, model):
        self.m = model
        self.joints = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        assert len(self.joints) == 43
        self.qa = model.jnt_qposadr[self.joints]
        self.va = model.jnt_dofadr[self.joints]
        self.root = model.body('pelvis').id
        self.cup = model.body('cup').id
        self.wrists = [model.body(side+'_wrist_yaw_link').id for side in ('left','right')]
        self.goal = model.body('cup_goal_marker').id
        assert model.body_mocapid[self.goal] >= 0
        jnames = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in self.joints]
        self.names = ['q/'+n for n in jnames]+['dq/'+n for n in jnames]
        self.names += ['root_height']+['gravity/'+a for a in 'xyz']
        self.names += ['root_velocity/'+str(i) for i in range(6)]
        for key,n in [('cup_pose',9),('cup_velocity',6),('goal',3),('left_wrist',9),('right_wrist',9),('previous_action',37)]:
            self.names += [key+'/'+str(i) for i in range(n)]

    def __call__(self, data, previous):
        mujoco.mj_forward(self.m,data)
        r = data.xmat[self.root].reshape(3,3); origin = data.xpos[self.root]
        def pose(body):
            rot = r.T@data.xmat[body].reshape(3,3)
            return np.r_[r.T@(data.xpos[body]-origin),rot[:,:2].reshape(-1)]
        def velocity(body):
            v=np.zeros(6);mujoco.mj_objectVelocity(self.m,data,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
            return np.r_[r.T@v[:3],r.T@v[3:]]
        out=np.r_[data.qpos[self.qa],data.qvel[self.va],origin[2],r.T@np.array([0,0,-1.]),
                  velocity(self.root),pose(self.cup),velocity(self.cup),r.T@(data.xpos[self.goal]-origin),
                  pose(self.wrists[0]),pose(self.wrists[1]),previous].astype(np.float32)
        assert out.shape==(len(self.names),) and np.isfinite(out).all()
        return out
