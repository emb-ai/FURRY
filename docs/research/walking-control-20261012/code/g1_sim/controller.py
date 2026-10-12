"""CPU port of TWIST2's server_low_level_g1_sim.py (MIT, Yanjie Ze 2025).

Observation order, history, PD gains and action scaling follow upstream.
Joint addresses are resolved by name so articulated hands and props are safe.
"""
from collections import deque
from pathlib import Path
import time

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "TWIST2"
DEFAULT = np.array([
    -.2, 0, 0, .4, -.2, 0, -.2, 0, 0, .4, -.2, 0, 0, 0, 0,
    0, .4, 0, 1.2, 0, 0, 0, 0, -.4, 0, 1.2, 0, 0, 0,
], dtype=np.float64)
KP = np.array([100,100,100,150,40,40]*2 + [150]*3 + [40]*4+[4]*3 + [40]*4+[4]*3)
KD = np.array([2,2,2,4,2,2]*2 + [4]*3 + [5]*4+[.2]*3 + [5]*4+[.2]*3)
JOINTS = ([f"{side}_{part}_joint" for side in ("left", "right") for part in
          ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
          + [f"waist_{part}_joint" for part in ("yaw", "roll", "pitch")]
          + [f"{side}_{part}_joint" for side in ("left", "right") for part in
             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")])


def euler(q):
    w, x, y, z = q
    return np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def standing_command():
    return np.r_[0., 0., .8, 0., 0., 0., DEFAULT]


class Controller:
    def __init__(self, model, policy=None, frequency=100):
        self.model = model
        self.qadr = np.array([model.joint(n).qposadr[0] for n in JOINTS])
        self.vadr = np.array([model.joint(n).dofadr[0] for n in JOINTS])
        self.aids = np.array([model.actuator(n).id for n in JOINTS])
        self.hand_ids = np.array([i for i in range(model.nu)
                                  if "hand_" in model.actuator(i).name], dtype=int)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(policy or UPSTREAM / "assets/ckpts/twist2_1017_20k.onnx"),
                                           sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.period = round(1 / (frequency * model.opt.timestep))
        if not np.isclose(self.period * model.opt.timestep, 1/frequency):
            raise ValueError("Policy period must be a whole number of physics steps")
        self.reset()

    def reset(self):
        self.history = deque([np.zeros(127, dtype=np.float32) for _ in range(10)], maxlen=10)
        self.last_action = np.zeros(29)
        self.target = DEFAULT.copy()
        self.step_index = 0
        self.inference_ms = []
        self.grip = 0.

    def step(self, data, command=None, grip=0.):
        command = standing_command() if command is None else np.asarray(command)
        if command.shape != (35,) or not np.isfinite(command).all():
            raise ValueError("Expected a finite 35-value TWIST2 command")
        q, v = data.qpos[self.qadr], data.qvel[self.vadr]
        if self.step_index % self.period == 0:
            obs_v = v.copy()
            obs_v[[4, 5, 10, 11]] = 0
            proprio = np.r_[data.qvel[3:6]*.25, euler(data.qpos[3:7])[:2], q-DEFAULT, obs_v*.05, self.last_action]
            current = np.r_[command, proprio]
            obs = np.r_[current, np.concatenate(self.history), command].astype(np.float32)[None]
            self.history.append(current.copy())
            start = time.perf_counter()
            action = self.session.run(None, {self.input_name: obs})[0].reshape(29)
            self.inference_ms.append((time.perf_counter()-start)*1000)
            if not np.isfinite(action).all():
                raise RuntimeError("Non-finite policy output")
            self.last_action = action.copy()
            self.target = DEFAULT + .5*np.clip(action, -10, 10)
        data.ctrl[self.aids] = np.clip((self.target-q)*KP - v*KD, -KP, KP)
        self.grip += np.clip(grip-self.grip, -.002, .002)
        for aid in self.hand_ids:
            name = self.model.actuator(aid).name
            sign = 1 if name.startswith("left") else -1
            if "thumb_0" in name:
                target = 0
            elif "thumb" in name:
                target = sign * (1. if "_1_" in name else 1.74)
            else:
                target = -sign * (1.57 if "_0_" in name else 1.74)
            jid = self.model.actuator_trnid[aid, 0]
            data.ctrl[aid] = np.clip(target*self.grip, *self.model.jnt_range[jid])
        mujoco.mj_step(self.model, data)
        self.step_index += 1

    def initialize(self, data):
        mujoco.mj_resetData(self.model, data)
        data.qpos[:7] = [0, 0, .793, 1, 0, 0, 0]
        data.qpos[self.qadr] = DEFAULT
        data.qpos[self.qadr[[16, 23]]] = [.2, -.2]
        mujoco.mj_forward(self.model, data)
        self.reset()
