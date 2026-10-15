"""Desktop port of the Quest native Simulation for HOMIE (groot_decoupled_v1) collection scenes.

Mirrors android/native/simulation.cpp (Reset, Step, StepDecoupled, MakeDecoupledCommand),
groot_decoupled.h, collection_task.h and cup_placement_cue.h on an archived episode model
(learning-model.mjb, motor profile already applied). Both decoupled modes are ported:
HOMIE v2 (walking) and HOMIE Standing (input patch homie_stand_v1: zero navigation, arm
gravity compensation, relaxed start pose). Tasks: cup (kind 1, fixed or sampled goal) and
bimanual (kind 3). The Quest
runtime sets one whole-body reference per 10 ms cycle and then calls Step 10 times;
``cycle`` does the same. Tennis/Rubik hooks are inert in collection scenes.
"""
from pathlib import Path

import json

import mujoco
import numpy as np
import onnxruntime as ort

from .controller import JOINTS

HOME = np.array([-.1,0,0,.3,-.2,0]*2+[0]*17)
KP = np.array([150,150,150,200,40,40]*2+[250]*3+[100,100,40,40,20,20,20]*2, float)
KD = np.array([2,2,2,4,2,2]*2+[5]*3+[5,5,2,2,2,2,2]*2, float)
SEED, ELAPSED, SETTLED, OUTCOME, LIFT0, LIFT1, STARTZ0, STARTZ1 = range(8)
MASK = 0xffffffff
# Standing rest arms (shoulder pitch, elbow) of the Quest builds that recorded the 2026-10-15 cup
# set (simulation.cpp sha256 prefixes); the build in git starts relaxed at (0, 1.4).
LEGACY_STANDING_REST = {'d2c4a56d': (-1.2, 1.2), '86322849': (-1.2, 1.2)}


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
    def __init__(self, episode, standing=None):
        """standing: None reads the episode manifest (input_patch homie_stand_v1)."""
        episode = Path(episode)
        manifest = json.loads((episode/'manifest.json').read_text()) if (episode/'manifest.json').exists() else {}
        self.standing = manifest.get('input_patch') == 'homie_stand_v1' if standing is None else bool(standing)
        self.scene = manifest.get('scene', '')
        build = manifest.get('sha256', {}).get('android/native/simulation.cpp', '')[:8]
        self.rest_arms = LEGACY_STANDING_REST.get(build, (0., 1.4))
        m = self.model = mujoco.MjModel.from_binary_path(str(episode/'learning-model.mjb'))
        self.data = mujoco.MjData(m); self.aux = mujoco.MjData(m)
        self.qa = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self.va = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.ai = np.array([m.actuator(n).id for n in JOINTS])
        self.torso = m.body('torso_link').id; self.pelvis = m.body('pelvis').id
        self.arm_gravity = np.zeros(14)
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
        if self.kind not in (1, 3) or m.nuserdata != 8: raise ValueError('Only the cup and bimanual collection tasks are ported')
        self.count = 2 if self.kind == 3 else 1
        fall = self._numeric('collection_fall')
        self.fall_height, self.fall_upright = (fall[0], fall[1]) if fall is not None else (.50, .707)
        self.require_hands = self._numeric('collection_bimanual_hands') is not None
        if self._numeric('collection_basket_sides') is not None: raise ValueError('basket sides not ported')
        names = ['cup'] if self.kind == 1 else [f'task_object_{i}' for i in range(2)]
        self.body = [m.body(n).id for n in names]
        self.joint = [m.joint(f'{n}_free').id for n in names]
        self.start = [self._numeric(f'collection_start_{i}') for i in range(self.count)]
        self.goal = [self._numeric(f'collection_goal_{i}') for i in range(self.count)]
        # Cup: optional sampled goal shown by the mocap marker (cup_placement_cue.h CupGoal).
        self.cup_bounds = self._numeric('collection_cup_goal_bounds'); self.cup_goal = self._numeric('cup_lab_goal')
        marker = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cup_goal_marker')
        self.cup_mocap = int(m.body_mocapid[marker]) if marker >= 0 else -1
        if self.cup_bounds is not None and (self.kind != 1 or self.cup_mocap < 0): raise ValueError('Invalid cup goal sampler')
        self.hand_body = np.array([2 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith('left_')
                                   else 1 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith('right_')
                                   else 0 for b in range(m.nbody)])

    def reset(self, seed):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[2] = .793; d.qpos[3] = 1; d.qpos[self.qa] = HOME
        d.qpos[self.qa[[18, 25]]] = 1.2; d.qpos[self.qa[16]] = .2; d.qpos[self.qa[23]] = -.2
        if self.standing:
            if self.scene in ('cup', 'bimanual', 'rubik'): d.qpos[0] = .05 if self.scene == 'rubik' else .18
            # Rest arms beside the body (pose of the recording build).
            d.qpos[self.qa[15:29]] = 0; d.qpos[self.qa[[16, 23]]] = [.2, -.2]
            d.qpos[self.qa[[15, 22]]] = self.rest_arms[0]; d.qpos[self.qa[[18, 25]]] = self.rest_arms[1]
        self.arm_gravity[:] = 0
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
        if self.cup_bounds is not None:
            g = self.cup_bounds; b = self.start[0]; band = 6 if rng.next() & 1 else 4
            for _ in range(4096):
                sx = rng.uniform(b[0], b[1]); sy = rng.uniform(b[2], b[3])
                gx = rng.uniform(g[0], g[1]); gy = rng.uniform(g[2], g[3])
                distance = np.hypot(sx-gx, sy-gy)
                if distance < g[band] or distance > g[band+1] or (band == 4 and sy*gy < 0): continue
                d.mocap_pos[self.cup_mocap] = [gx, gy, self.goal[0][2]]; break
            else: raise RuntimeError('Unreachable cup layout sampling bounds')
        for i in range(self.count):
            q = m.jnt_qposadr[self.joint[i]]; v = m.jnt_dofadr[self.joint[i]]; b = self.start[i]
            if self.cup_bounds is not None: x, y = sx, sy
            else:
                while True:
                    x = rng.uniform(b[0], b[1]); y = rng.uniform(b[2], b[3])
                    if self.kind != 1 or np.hypot(x-self.goal[i][0], y-self.goal[i][1]) >= .15: break
            yaw = rng.uniform(-b[5], b[5])
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
        if self.standing: self.command['velocity'] = np.zeros(3, np.float32); self._update_arm_gravity()
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

    def _update_arm_gravity(self):
        """Standing: arm gravity torques from current qpos with zero velocity (mj_rne)."""
        m, a = self.model, self.aux
        a.qpos[:] = self.data.qpos; a.qvel[:] = 0
        mujoco.mj_kinematics(m, a); mujoco.mj_comPos(m, a); mujoco.mj_comVel(m, a)
        bias = np.zeros(m.nv); mujoco.mj_rne(m, a, 0, bias)
        self.arm_gravity[:] = bias[self.va[15:]]

    def _step_decoupled(self):
        c = self.command = self._make_command()
        # Standing: the Balance actor gets zero navigation; height, torso and arms stay operated.
        if self.standing: c['velocity'] = np.zeros(3, np.float32)
        self.history[:-86] = self.history[86:]; self.history[-86:] = self._observation(c)
        walking = np.sqrt(float(np.sum(np.float64(np.float32(c['velocity']))**2))) >= .05
        s = self.walk if walking else self.balance
        action = s.run(None, {s.get_inputs()[0].name: self.history.reshape(1, -1)})[0].ravel()
        if not np.isfinite(action).all(): raise RuntimeError('Decoupled policy output is not finite')
        self.last_action = action.astype(np.float32)
        self.target[:15] = HOME[:15] + np.float32(.25)*self.last_action
        self.target[15:] = np.float32(c['arms'])
        if self.standing: self._update_arm_gravity()

    def step(self, grip, right_grip):
        m, d = self.model, self.data
        if self.steps % 20 == 0: self._step_decoupled()
        d.ctrl[self.ai] = self.torque(d.qpos, d.qvel)
        self.hand_grip += np.clip(np.array([grip, right_grip])-self.hand_grip, -.002, .002)
        d.ctrl[self.hands] = np.clip(self.hand_target*self.hand_grip[self.hand_side], self.hand_range[:, 0], self.hand_range[:, 1])
        mujoco.mj_step(m, d)
        self._task_finish()
        self.steps += 1
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < min(.35, self.fall_height): raise RuntimeError('G1 fell: reset required')

    def torque(self, qpos, qvel):
        """Joint PD of the decoupled backend (+ arm gravity when standing)."""
        tau = (np.float32(self.target)-qpos[self.qa])*KP - qvel[self.va]*KD
        if self.standing: tau[15:] += self.arm_gravity
        return tau

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
        for i in range(self.count):
            p = d.xpos[self.body[i]]; g = self.goal[i].copy()
            if self.kind == 1 and self.cup_goal is not None:
                g[:3] = d.mocap_pos[self.cup_mocap] if self.cup_mocap >= 0 else self.cup_goal[:3]
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
            if self.kind == 1: inside = np.hypot(p[0]-g[0], p[1]-g[1]) < g[3] and abs(p[2]-g[2]) < g[5]
            else: inside = all(abs(p[k]-g[k]) < g[3+k] for k in range(3))
            all_ok = all_ok and bool(s[LIFT0+i]) and inside and not touching and np.linalg.norm(vel[3:]) < .06 and np.linalg.norm(vel[:3]) < .25
            if self.kind == 1: all_ok = all_ok and d.xmat[self.body[i], 8] > .94
        s[SETTLED] = s[SETTLED]+m.opt.timestep if all_ok else 0
        if s[SETTLED] >= .5: s[OUTCOME] = 1
        elif s[ELAPSED] >= self.timeout: s[OUTCOME] = 3

    @property
    def outcome(self): return int(self.data.userdata[OUTCOME])
