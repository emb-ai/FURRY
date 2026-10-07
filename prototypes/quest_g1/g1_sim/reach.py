"""Scripted right-hand reach toward the cup, and the shared episode loop.

The low-level tracker does not follow an arbitrary inverse-kinematics pose.
It does hold a smooth reference that lifts the right arm and aims it with
the shoulder roll. Wrist joints stay at the standing command: combined wrist
targets were not tracked. The hand approaches the cup and stops short of a
grasp. Grip stays open.

Demonstrations are ordinary desktop NPZ recordings. Training exports are a
separate ``state-v1`` file; the raw episode is not modified.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .controller import DEFAULT, JOINTS, Controller, standing_command
from .scene import make_model
from .state import convert_recording
from .tasks import cup_picked, fell

SCENE = "cup"
TIP = "right_hand_middle_finger_tip"
DURATION = 3.0
BLEND = 2.0
# Cup center on the table, inside the region this reference can approach.
CUP_X = (0.38, 0.46)
CUP_Y = (-0.22, -0.02)
CUP_Z = 0.754
RIGHT_ARM = JOINTS[22:29]


def sample_cup(rng):
    """One cup position. ``rng`` is a ``numpy`` Generator."""
    return np.array([rng.uniform(*CUP_X), rng.uniform(*CUP_Y), CUP_Z], dtype=np.float64)


def place_cup(model, data, cup):
    joint = model.joint("cup_free")
    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[qadr:qadr+3] = cup
    data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
    data.qvel[dadr:dadr+6] = 0
    mujoco.mj_forward(model, data)


def arm_target_delta(cup_y):
    """Right-arm offset from the standing pose, in ``JOINTS`` order.

    Shoulder roll is a line fit to tracked fingertip positions: a delta of
    0.45 put the fingertip near y=-0.15, and 0.65 put it near y=-0.063.
    Pitch and elbow are the largest lift that stayed put for the full hold.
    """
    y = float(np.clip(cup_y, *CUP_Y))
    shoulder_roll = float(np.clip(0.45 + (y - (-0.150)) / 0.435, 0.30, 0.90))
    return np.array([-0.95, shoulder_roll, 0.0, -0.55, 0.0, 0.0, 0.0], dtype=np.float64)


def _limits(model):
    return np.array([model.jnt_range[model.joint(name).id] for name in RIGHT_ARM], dtype=np.float64)


class ScriptedReach:
    """Smooth blend from standing to a cup-dependent arm reference."""

    def __init__(self, model, cup, blend=BLEND):
        self.blend = float(blend)
        self.stand = standing_command()
        self.goal = self.stand.copy()
        limits = _limits(model)
        self.goal[28:35] = np.clip(DEFAULT[22:29] + arm_target_delta(cup[1]), limits[:, 0], limits[:, 1])

    def reset(self):
        return None

    def __call__(self, model, data, controller):
        phase = 0.5 * (1 - np.cos(np.pi * np.clip(data.time / self.blend, 0, 1)))
        return (1 - phase) * self.stand + phase * self.goal, 0.0


class Standing:
    def reset(self):
        return None

    def __call__(self, model, data, controller):
        return standing_command(), 0.0


def tip_distance(data):
    return float(np.linalg.norm(data.body(TIP).xpos - data.body("cup").xpos))


def run_episode(seed, seconds=DURATION, record=None, source_factory=None):
    """Roll a reach episode. The default source is the scripted reference.

    ``source_factory(model, cup)`` may supply another command source. The
    source is called before every physics step and must return ``(command, grip)``.
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    model, xml = make_model(SCENE)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TIP) < 0:
        raise ValueError("The reach task needs articulated hands")
    data = mujoco.MjData(model)
    controller = Controller(model)
    controller.initialize(data)
    cup = sample_cup(np.random.default_rng(seed))
    place_cup(model, data, cup)
    source = ScriptedReach(model, cup) if source_factory is None else source_factory(model, cup)
    distances = [tip_distance(data)]
    min_height, max_tilt = float(data.qpos[2]), 0.0
    picked = cup_picked(model, data, controller.grip)
    fallen = fell(model, data)
    recording = []
    while data.time < seconds:
        command, grip = source(model, data, controller)
        command = np.asarray(command, dtype=np.float64)
        grip = float(grip)
        controller.step(data, command, grip)
        distances.append(tip_distance(data))
        picked = picked or cup_picked(model, data, controller.grip)
        fallen = fallen or fell(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        max_tilt = max(max_tilt, float(np.arccos(np.clip(data.xmat[model.body("pelvis").id, 8], -1, 1))))
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.35:
            raise RuntimeError(f"Robot fell or simulation became unstable at t={data.time:.3f}")
        if record is not None and controller.step_index % controller.period == 0:
            recording.append((data.time, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), command.copy(), grip))
    metrics = {
        "task": "reach",
        "scene": SCENE,
        "seed": int(seed),
        "seconds": float(data.time),
        "cup_xyz": cup.tolist(),
        "initial_tip_distance_m": distances[0],
        "minimum_tip_distance_m": float(min(distances)),
        "final_tip_distance_m": distances[-1],
        "minimum_pelvis_height_m": min_height,
        "maximum_pelvis_tilt_deg": float(np.degrees(max_tilt)),
        "mujoco_warnings": int(sum(w.number for w in data.warning)),
        "pickup": bool(picked),
        "fell": bool(fallen),
    }
    if record is not None:
        record = Path(record)
        record.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            record, time=[row[0] for row in recording], qpos=[row[1] for row in recording],
            qvel=[row[2] for row in recording], ctrl=[row[3] for row in recording],
            command=[row[4] for row in recording], grip=[row[5] for row in recording],
            scene_xml=xml, scene=SCENE, metadata=json.dumps(metrics))
        metrics["recording"] = str(record)
    return metrics


def generate(directory, demos=16, seconds=DURATION, seed=0):
    """Write raw episodes and ``state-v1`` exports. Raw files stay untouched."""
    if demos < 1:
        raise ValueError("demos must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(demos):
        episode_seed = seed + index
        raw = directory / f"reach_{index:03d}.npz"
        metrics = run_episode(episode_seed, seconds, record=raw)
        if metrics["final_tip_distance_m"] > metrics["initial_tip_distance_m"] - 0.05:
            raise RuntimeError(f"Reach demo {index} did not approach the cup: {metrics}")
        if metrics["mujoco_warnings"] or metrics["minimum_pelvis_height_m"] < 0.70:
            raise RuntimeError(f"Reach demo {index} was unstable: {metrics}")
        export = directory / f"reach_{index:03d}_state.npz"
        exported = convert_recording(raw, export)
        metrics["state"] = export.name
        metrics["state_rows"] = exported["rows"]
        metrics["state_dim"] = exported["state_dim"]
        rows.append(metrics)
        print(f"reach {index:03d} seed={episode_seed} cup_y={metrics['cup_xyz'][1]:.3f} "
              f"distance {metrics['initial_tip_distance_m']:.3f}->{metrics['final_tip_distance_m']:.3f}", flush=True)
    manifest = {"task": "reach", "scene": SCENE, "seed": seed, "demos": rows}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate scripted cup-reach demonstrations")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--demos", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=DURATION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = generate(args.directory, args.demos, args.seconds, args.seed)
    finals = [row["final_tip_distance_m"] for row in manifest["demos"]]
    print(json.dumps({"demos": len(finals), "final_tip_distance_m": {"min": min(finals), "max": max(finals)}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
