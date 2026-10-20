"""Goal-conditioned manipulation over the unchanged native HOMIE runtime.

Only 12 wrist coordinates and two scalar Grip values are actions. Native
snapshots are restored ONLY at episode resets, never between requested moves.
The policy has privileged simulator observations; this is not a vision policy.
"""
from pathlib import Path
import ctypes
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT.parent / 'rubik-canonical-20261015'))
os.environ['BRIDGE'] = 'bridge-original-snapshot.dylib'
os.environ['FORCE'] = '0'
os.environ.pop('NOSLIP', None)
import gymnasium as gym
from gymnasium import spaces
from common import Sim, lib, mujoco, ASSETS

lib.snapshot.argtypes = [ctypes.c_void_p]
lib.snapshot.restype = ctypes.c_void_p
lib.restore.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.free_snapshot.argtypes = [ctypes.c_void_p]
assert not hasattr(lib, 'fingers')
AXES = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1.]])
GROUP = R.create_group('O')
CONTROL_DT = .05
# Covers the observed wrist-goal speeds without clipping demonstration labels.
# Physical joint speed/force limits remain those of the original native runtime.
TRANSLATION_STEP = .60 * CONTROL_DT
ROTATION_STEP = 3.0 * CONTROL_DT

def turn_target(corners, face, sign):
    """Signed rotation about native face normal; sign is NOT Singmaster notation."""
    out = corners.copy()
    selected = corners[:, :3] @ AXES[face] > 0
    assert selected.sum() == 4
    rotation = R.from_rotvec(AXES[face] * sign * np.pi / 2)
    out[selected, :3] = rotation.apply(corners[selected, :3])
    rr = rotation * R.from_quat(corners[selected][:, [4,5,6,3]])
    out[selected, 3:] = rr.as_quat()[:, [3,0,1,2]]
    return out

def rot6(rotation):
    return rotation.as_matrix()[..., :, :2].reshape(-1)

class Geometry:
    def __init__(self, model, reference_qpos):
        self.m = model
        self.d = mujoco.MjData(model)
        self.cube = [model.body('rubik_cubie_' + str(i)).id for i in range(8)]
        self.core = model.body('rubik_core').id
        self.wr = [model.body(h + '_wrist_yaw_link').id for h in ['left','right']]
        self.hands = [[b for b in range(model.nbody) if (model.body(b).name or '').startswith(h + '_hand_')] for h in ['left','right']]
        joints = model.actuator_trnid[:, 0]
        self.qa = model.jnt_qposadr[joints]
        self.va = model.jnt_dofadr[joints]
        self.update(reference_qpos)
        self.home_center = self.d.xpos[self.cube].mean(0).copy()
        self.home_rotation = R.from_quat(self.d.xquat[self.core][[1,2,3,0]])
        self.home_fingers = [self.home_rotation.inv().apply(self.d.xpos[ids]-self.home_center) for ids in self.hands]
        self.home_finger_rot = [self.home_rotation.inv()*R.from_quat(self.d.xquat[ids][:,[1,2,3,0]]) for ids in self.hands]

    def update(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)

    def frame(self):
        c = self.d.xpos[self.cube].mean(0)
        core = R.from_quat(self.d.xquat[self.core][[1,2,3,0]])
        choices = core * GROUP
        grid = choices[np.argmin((self.home_rotation.inv()*choices).magnitude())]
        return c, core, grid

    def physical_corners(self):
        _, core, _ = self.frame()
        p = core.inv().apply(self.d.xpos[self.cube]-self.d.xpos[self.core])
        rr = core.inv()*R.from_quat(self.d.xquat[self.cube][:,[1,2,3,0]])
        return np.c_[p, rr.as_quat()[:,[3,0,1,2]]]

    def metrics(self, target):
        current = self.physical_corners()
        c, _, grid = self.frame()
        angular = (R.from_quat(current[:,[4,5,6,3]])*R.from_quat(target[:,[4,5,6,3]]).inv()).magnitude()
        hand_pos, hand_rot = [], []
        for h, ids in enumerate(self.hands):
            e = grid.inv().apply(self.d.xpos[ids]-c)-self.home_fingers[h]
            hand_pos.append(float(np.sqrt(np.mean(np.sum(e*e,axis=1)))))
            rr = grid.inv()*R.from_quat(self.d.xquat[ids][:,[1,2,3,0]])*self.home_finger_rot[h].inv()
            hand_rot.append(float(np.sqrt(np.mean(rr.magnitude()**2))))
        return dict(corner_position=float(np.linalg.norm(current[:,:3]-target[:,:3],axis=1).max()),
                    corner_angle=float(angular.max()), center=float(np.linalg.norm(c-self.home_center)),
                    grid_angle=float((grid*self.home_rotation.inv()).magnitude()),
                    hand_position=hand_pos, hand_angle=hand_rot)

    def observation(self, qvel, goals, grips, masks, target, command, released):
        c, core, grid = self.frame()
        current = self.physical_corners()
        wr = R.from_quat(self.d.xquat[self.wr][:,[1,2,3,0]])
        goal_r = R.from_quat(goals[:,[4,5,6,3]])
        onehot = np.zeros(12)
        onehot[2*command[0]+int(command[1]>0)] = 1
        contact = np.array([[(int(mask)>>i)&1 for i in range(8)] for mask in masks])
        met = self.metrics(target)
        return np.r_[self.d.qpos[self.qa], qvel[self.va]/5,
            (c-self.home_center)/.1, rot6(core),
            current[:,:3].ravel()/.04, rot6(R.from_quat(current[:,[4,5,6,3]])),
            target[:,:3].ravel()/.04, rot6(R.from_quat(target[:,[4,5,6,3]])),
            (self.d.xpos[self.wr]-c).ravel()/.2, rot6(wr),
            (goals[:,:3]-c).ravel()/.2, rot6(goal_r), grips,
            contact.ravel(), onehot, float(released),
            np.array(met['hand_position'])/.01, np.array(met['hand_angle']),
            (grid*self.home_rotation.inv()).as_rotvec()].astype(np.float32)

class NativeRubikEnv(gym.Env):
    metadata = {}
    def __init__(self, sequence_length=1, faces=(3,), signs=(-1,), seed=0, move_timeout=45.):
        self.sequence_length, self.faces, self.signs = sequence_length, tuple(faces), tuple(signs)
        self.move_timeout = move_timeout
        # Scene is deliberately fixed for the first curriculum stage; seed
        # randomizes requested commands. A recorded warmup is not transferable
        # to an independently randomized physical start without validation.
        self.s = Sim('rubik', policy='homie-standing', seed=0)
        self.s.step(n=2500)
        source = dict(np.load(ROOT.parent/'rubik-feedback-20261015/verified-nine.npz'))
        for cc in source['commands'][:4050]:
            if self.s.step(cc[:14].reshape(2,7), cc[-2:]):
                raise RuntimeError('Native failure during initial grasp replay')
        assert np.max(abs(self.s.q-source['qpos'][4049])) < 1e-9
        self.initial_goals = source['commands'][4049,:14].reshape(2,7).copy()
        self.initial_grips = source['commands'][4049,-2:].copy()
        self.geometry = Geometry(self.s.m, self.s.q)
        self.initial_snapshot = lib.snapshot(self.s.p)
        self.action_space = spaces.Box(-1., 1., (14,), dtype=np.float32)
        self.velocity = np.zeros(self.s.m.nv)
        self.reset(seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, self.observe().shape, dtype=np.float32)

    def observe(self):
        self.s.read()
        self.geometry.update(self.s.q)
        lib.velocity(self.s.p, self.velocity)
        masks = [lib.rubik_contacts(self.s.p,h) for h in range(2)]
        return self.geometry.observation(self.velocity,self.goals,self.grips,masks,
                                         self.target,self.command,self.released)

    def _next_goal(self):
        self.command = self.sequence[self.completed]
        self.target = turn_target(self.logical, *self.command)
        self.goal_steps = 0
        self.stable = 0
        self.release_ticks = np.zeros(2, dtype=int)
        self.released = False
        c, _, _ = self.geometry.frame()
        self.start_distance = np.linalg.norm(self.geometry.d.xpos[self.geometry.wr]-c,axis=1)
        self.potential = self._potential(self.geometry.metrics(self.target))

    @staticmethod
    def _potential(m):
        return -(2*m['corner_position']/.04 + 2*m['corner_angle']/(np.pi/2)
                 + .1*m['center']/.01 + .1*max(m['hand_position'])/.01
                 + .1*m['grid_angle']/.1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        lib.restore(self.s.p,self.initial_snapshot)
        self.s.read()
        self.geometry.update(self.s.q)
        self.goals, self.grips = self.initial_goals.copy(), self.initial_grips.copy()
        self.logical = self.s.state()[8:64].reshape(8,7).copy()
        requested = (options or {}).get('sequence')
        self.sequence = [tuple(x) for x in requested] if requested is not None else [
            (int(self.np_random.choice(self.faces)),int(self.np_random.choice(self.signs)))
            for _ in range(self.sequence_length)]
        if not self.sequence or any(f not in range(6) or s not in (-1,1) for f,s in self.sequence):
            raise ValueError('Invalid sequence')
        self.completed, self.no_contact_ticks = 0, 0
        self._next_goal()
        return self.observe(), {'sequence':self.sequence,'start':'replayed airborne grasp at 40.50s'}

    def step(self, action):
        action = np.clip(np.asarray(action,dtype=float),-1,1)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected finite 14-dimensional wrist/Grip action')
        old, old_grip = self.goals.copy(), self.grips.copy()
        delta = action[:12].reshape(2,6)
        target = old.copy()
        target[:,:3] += delta[:,:3]*TRANSLATION_STEP
        # Bounds apply to commanded wrists, never to physical qpos or object state.
        target[:,:3] = np.clip(target[:,:3], self.geometry.home_center+[-.4,-.4,-.25],
                              self.geometry.home_center+[.2,.4,.3])
        initial_r = R.from_quat(old[:,[4,5,6,3]])
        final_grip = (action[12:]+1)/2
        failure = None
        for k in range(5):
            u=(k+1)/5
            self.goals=old.copy()
            self.goals[:,:3]=old[:,:3]*(1-u)+target[:,:3]*u
            self.goals[:,3:]=(R.from_rotvec(delta[:,3:]*ROTATION_STEP*u)*initial_r).as_quat()[:,[3,0,1,2]]
            self.grips=old_grip*(1-u)+final_grip*u
            if self.s.step(self.goals,self.grips): failure='native_failure';break
            masks=np.array([lib.rubik_contacts(self.s.p,h) for h in range(2)])
            self.no_contact_ticks = self.no_contact_ticks+1 if not np.any(masks) else 0
            self.release_ticks=np.where((masks==0)&(self.grips<.1),self.release_ticks+1,0)
            c=self.s.d.xpos[self.geometry.cube].mean(0)
            distance=np.linalg.norm(self.s.d.xpos[self.geometry.wr]-c,axis=1)
            self.released |= bool(np.any((self.release_ticks>=20)&(distance>self.start_distance+.02)))
            if lib.supports(self.s.p):failure='external_support';break
            if c[2]<.90 or self.no_contact_ticks>=1:failure='cube_lost';break
        obs=self.observe()
        self.goal_steps+=1
        met=self.geometry.metrics(self.target)
        potential=self._potential(met)
        reward=potential-self.potential-.0005
        self.potential=potential
        contact=all(lib.rubik_contacts(self.s.p,h) for h in range(2))
        ready=(met['corner_position']<.001 and met['corner_angle']<np.deg2rad(3)
               and met['center']<.008 and met['grid_angle']<np.deg2rad(6)
               and max(met['hand_position'])<.008 and max(met['hand_angle'])<np.deg2rad(6)
               and contact and self.released)
        self.stable=self.stable+1 if ready else 0
        done=False
        released_for_this_goal=self.released
        completed_this_step=False
        if self.stable>=6 and failure is None:
            completed_this_step=True
            self.completed+=1;reward+=10
            self.logical=self.target.copy()
            done=self.completed==len(self.sequence)
            if not done:self._next_goal();obs=self.observe()
        if failure:reward-=10;done=True
        timeout=self.goal_steps*CONTROL_DT>=self.move_timeout
        info={'completed_moves':self.completed,'requested_moves':len(self.sequence),
              'success':bool(self.completed==len(self.sequence)), 'failure':failure,
              'metrics':met, 'released_and_withdrawn':released_for_this_goal,
              'completed_this_step':completed_this_step, 'next_command':self.command}
        return obs,float(reward),bool(done),bool(timeout and not done),info

    def close(self):
        if self.initial_snapshot:
            lib.free_snapshot(self.initial_snapshot);self.initial_snapshot=None
            self.s.close()
