"""Desktop port of the Quest native Simulation for HOMIE v2 (groot_decoupled_v1) collection scenes.

Mirrors android/native/simulation.cpp (Reset, Step, StepDecoupled, MakeDecoupledCommand),
groot_decoupled.h and collection_task.h for the non-standing decoupled backend on an
archived episode model (learning-model.mjb, motor profile already applied). The Quest
runtime sets one whole-body reference per 10 ms cycle and then calls Step 10 times;
``cycle`` does the same. Tennis/Rubik hooks are inert in collection scenes.
"""
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .controller import JOINTS

HOME = np.array([-.1,0,0,.3,-.2,0]*2+[0]*17)
KP = np.array([150,150,150,200,40,40]*2+[250]*3+[100,100,40,40,20,20,20]*2, float)
KD = np.array([2,2,2,4,2,2]*2+[5]*3+[5,5,2,2,2,2,2]*2, float)
SEED, ELAPSED, SETTLED, OUTCOME, LIFT0, LIFT1, STARTZ0, STARTZ1 = range(8)
MASK = 0xffffffff


def _session(path):
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 1; opt.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])


class Rng:
    """collection_task.h Next/Uniform on uint32 state."""
    def __init__(self, seed): self.s = int(seed) & MASK
    def next(self):
        self.s = (self.s + 0x9e3779b9) & MASK; z = self.s
        z = ((z ^ (z >> 16)) * 0x85ebca6b) & MASK; z = ((z ^ (z >> 13)) * 0xc2b2ae35) & MASK
        return z ^ (z >> 16)
    def uniform(self, lo, hi): return lo + (hi-lo)*(self.next()/4294967296.)


class QuestSim:
    def __init__(self, episode):
        episode = Path(episode)
        m = self.model = mujoco.MjModel.from_binary_path(str(episode/'learning-model.mjb'))
        self.data = mujoco.MjData(m); self.aux = mujoco.MjData(m)
        self.qa = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self.va = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.ai = np.array([m.actuator(n).id for n in JOINTS])
        self.torso = m.body('torso_link').id; self.pelvis = m.body('pelvis').id
        # Native: Balance is the primary policy, Walk the secondary.
        self.balance = _session(episode/'policy.onnx'); self.walk = _session(episode/'policy-secondary.onnx')
        self.hands, self.hand_side, self.hand_target, self.hand_range = [], [], [], []
        for aid in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if 'hand_' not in name: continue
            sign = 1. if name.startswith('left') else -1.; target = 0.
            if 'thumb_0' not in name:
                target = sign*(1. if '_1_' in name else 1.74) if 'thumb' in name else -sign*(1.57 if '_0_' in name else 1.74)
            j = m.actuator_trnid[aid, 0]
            self.hands.append(aid); self.hand_side.append(0 if sign > 0 else 1)
            self.hand_target.append(target); self.hand_range.append(m.jnt_range[j])
        self.hands = np.array(self.hands); self.hand_side = np.array(self.hand_side)
        self.hand_target = np.array(self.hand_target); self.hand_range = np.array(self.hand_range)
        self._task_init()
        self.reset(0)

    def _numeric(self, name):
        m = self.model; i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        return None if i < 0 else m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i]+m.numeric_size[i]].copy()

    def _task_init(self):
        m = self.model; c = self._numeric('collection_task')
        self.kind, self.timeout = int(c[0]), float(c[1])
        if self.kind != 3 or m.nuserdata != 8: raise ValueError('Only the bimanual collection task is ported')
        fall = self._numeric('collection_fall')
        self.fall_height, self.fall_upright = (fall[0], fall[1]) if fall is not None else (.50, .707)
        self.require_hands = self._numeric('collection_bimanual_hands') is not None
        if self._numeric('collection_basket_sides') is not None: raise ValueError('basket sides not ported')
        self.body = [m.body(f'task_object_{i}').id for i in range(2)]
        self.joint = [m.joint(f'task_object_{i}_free').id for i in range(2)]
        self.start = [self._numeric(f'collection_start_{i}') for i in range(2)]
        self.goal = [self._numeric(f'collection_goal_{i}') for i in range(2)]
        self.hand_body = np.array([2 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith('left_')
                                   else 1 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith('right_')
                                   else 0 for b in range(m.nbody)])

    def reset(self, seed):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[2] = .793; d.qpos[3] = 1; d.qpos[self.qa] = HOME
        d.qpos[self.qa[[18, 25]]] = 1.2; d.qpos[self.qa[16]] = .2; d.qpos[self.qa[23]] = -.2
        self.target = d.qpos[self.qa].copy()
        self.steps = 0; self.hand_grip = np.zeros(2)
        self.history = np.zeros(516, np.float32); self.last_action = np.zeros(15, np.float32)
        # Default decoupled command: zero velocity, height .74, zero torso, current arms.
        self.command = dict(velocity=np.zeros(3), height=.74, torso=np.zeros(3), arms=d.qpos[self.qa[15:]].copy())
        self.has_whole = False; self.whole = np.zeros(35, np.float32)
        mujoco.mj_forward(m, d)
        self._task_reset(seed)

    def _task_reset(self, seed):
        m, d = self.model, self.data
        d.userdata[:] = 0; d.userdata[SEED] = seed; rng = Rng(seed)
        for i in range(2):
            q = m.jnt_qposadr[self.joint[i]]; v = m.jnt_dofadr[self.joint[i]]; b = self.start[i]
            x = rng.uniform(b[0], b[1]); y = rng.uniform(b[2], b[3]); yaw = rng.uniform(-b[5], b[5])
            d.qpos[q:q+7] = [x, y, b[4], np.cos(yaw/2), 0, 0, np.sin(yaw/2)]; d.qvel[v:v+6] = 0
            d.userdata[STARTZ0+i] = b[4]
        mujoco.mj_forward(m, d)

    def load_state(self, qpos, qvel, userdata, ctrl, steps, whole=None):
        """Start from a recorded frame. HOMIE history is not recorded: it is filled
        with the current observation (zero previous action); grips come from ctrl."""
        m, d = self.model, self.data
        d.qpos[:] = qpos; d.qvel[:] = qvel; d.userdata[:] = userdata; d.ctrl[:] = ctrl
        self.steps = int(steps); mujoco.mj_forward(m, d)
        for h in range(2):
            k = (self.hand_side == h) & (np.abs(self.hand_target) > 0)
            self.hand_grip[h] = float(np.median(d.ctrl[self.hands[k]]/self.hand_target[k]))
        self.target = d.qpos[self.qa].copy(); self.last_action[:] = 0
        if whole is not None: self.set_reference(whole)
        self.command = self._make_command()
        self.history[:] = np.tile(self._observation(self.command), 6)

    def set_reference(self, whole):
        whole = np.asarray(whole, np.float32)
        if whole.shape != (35,) or not np.isfinite(whole).all(): raise ValueError('Invalid whole-body reference')
        self.whole = whole.copy(); self.has_whole = True

    def pause(self):
        if self.has_whole: self.whole[[0, 1, 5]] = 0

    def _make_command(self):
        c = dict(self.command)
        if not self.has_whole: return c
        w = self.whole; m, a = self.model, self.aux
        mujoco.mj_resetData(m, a)
        roll = np.zeros(4); pitch = np.zeros(4)
        mujoco.mju_axisAngle2Quat(roll, np.array([1., 0, 0]), float(w[3]))
        mujoco.mju_axisAngle2Quat(pitch, np.array([0, 1., 0]), float(w[4]))
        mujoco.mju_mulQuat(a.qpos[3:7], pitch, roll)
        a.qpos[self.qa[12:15]] = w[18:21]
        mujoco.mj_kinematics(m, a)
        r = a.xmat[self.torso]
        c.update(velocity=np.array([w[0], w[1], w[5]], np.float32), height=float(w[2]),
                 torso=np.array([np.arctan2(r[7], r[8]), np.arcsin(np.clip(-r[6], -1, 1)), np.arctan2(r[3], r[0])], np.float32),
                 arms=np.asarray(w[21:35], float))
        return c

    def _observation(self, c):
        """grootdecoupled::Observe."""
        d = self.data; w, x, y, z = d.qpos[3:7]
        obs = np.empty(86, np.float32)
        obs[0:3] = np.float32(c['velocity'])*np.float32([2, 2, .5]); obs[3] = c['height']
        obs[4:7] = c['torso']; obs[7:10] = d.qvel[3:6]*.5
        obs[10:13] = [2*(w*y-x*z), -2*(w*x+y*z), 1-2*(w*w+z*z)]
        obs[13:42] = d.qpos[self.qa]-HOME; obs[42:71] = d.qvel[self.va]*.05; obs[71:86] = self.last_action
        return obs

    def _step_decoupled(self):
        c = self.command = self._make_command()
        self.history[:-86] = self.history[86:]; self.history[-86:] = self._observation(c)
        walking = np.sqrt(float(np.sum(np.float64(np.float32(c['velocity']))**2))) >= .05
        s = self.walk if walking else self.balance
        action = s.run(None, {s.get_inputs()[0].name: self.history.reshape(1, -1)})[0].ravel()
        if not np.isfinite(action).all(): raise RuntimeError('Decoupled policy output is not finite')
        self.last_action = action.astype(np.float32)
        self.target[:15] = HOME[:15] + np.float32(.25)*self.last_action
        self.target[15:] = np.float32(c['arms'])

    def step(self, grip, right_grip):
        m, d = self.model, self.data
        if self.steps % 20 == 0: self._step_decoupled()
        d.ctrl[self.ai] = (np.float32(self.target)-d.qpos[self.qa])*KP - d.qvel[self.va]*KD
        self.hand_grip += np.clip(np.array([grip, right_grip])-self.hand_grip, -.002, .002)
        d.ctrl[self.hands] = np.clip(self.hand_target*self.hand_grip[self.hand_side], self.hand_range[:, 0], self.hand_range[:, 1])
        mujoco.mj_step(m, d)
        self._task_finish()
        self.steps += 1
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < min(.35, self.fall_height): raise RuntimeError('G1 fell: reset required')

    def cycle(self, whole, grip, right_grip):
        """One Quest 10 ms cycle: optional reference, then 10 physics steps."""
        if whole is None: self.pause()
        else: self.set_reference(whole)
        for _ in range(10): self.step(grip, right_grip)

    def _task_finish(self):
        m, d = self.model, self.data; s = d.userdata
        if s[OUTCOME]: return
        s[ELAPSED] += m.opt.timestep
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < self.fall_height or d.xmat[self.pelvis, 8] < self.fall_upright:
            s[OUTCOME] = 2; return
        n = d.ncon; geom = d.contact.geom[:n]; dist = d.contact.dist[:n]
        all_ok = True; vel = np.zeros(6)
        for i in range(2):
            p = d.xpos[self.body[i]]; g = self.goal[i]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.body[i], vel, 0)
            touching = correct = False
            for side in range(2):
                a, b = geom[:, side], geom[:, 1-side]
                hit = (a >= 0) & (b >= 0) & (dist <= 0)
                hit &= (m.geom_bodyid[np.maximum(a, 0)] == self.body[i]) & ((m.geom_contype[np.maximum(b, 0)] & 4) != 0)
                if hit.any():
                    touching = True
                    correct |= bool((self.hand_body[m.geom_bodyid[b[hit]]] == (2 if i == 0 else 1)).any())
            if p[2] > s[STARTZ0+i]+.045 and (not self.require_hands or correct): s[LIFT0+i] = 1
            inside = all(abs(p[k]-g[k]) < g[3+k] for k in range(3))
            all_ok = all_ok and bool(s[LIFT0+i]) and inside and not touching and np.linalg.norm(vel[3:]) < .06 and np.linalg.norm(vel[:3]) < .25
        s[SETTLED] = s[SETTLED]+m.opt.timestep if all_ok else 0
        if s[SETTLED] >= .5: s[OUTCOME] = 1
        elif s[ELAPSED] >= self.timeout: s[OUTCOME] = 3

    @property
    def outcome(self): return int(self.data.userdata[OUTCOME])
