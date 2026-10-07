"""Success checks, timeouts, falls, and seeded starts for cup and push_t.

The TWIST2 paper describes a cup pickup and a kick of the T onto its mark,
and it does not publish a numeric tolerance. The thresholds below are the
FURRY rules used for every seeded rollout. A fall or a timeout stays that
outcome. It is not rewritten as a success.

Cup success is a pickup. The cup center must be at least 4 cm above the
0.748 m height it settles to on the table, the right middle fingertip must
be within 12 cm of that center, and the achieved grip must be at least
one half. Push-T success is at least half of the target footprint covered
by the T. Both are judged before the timeout, and a fall is checked first.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .controller import Controller
from .scene import make_model

SCHEMA = "task-v1"
TIMEOUT_S = 8.0
FALL_HEIGHT_M = 0.55
FALL_TILT_DEG = 45.0
TIP = "right_hand_middle_finger_tip"

# Measured resting height of the cup center on the table top, then the lift.
CUP_REST_Z = 0.748
CUP_LIFT_M = 0.04
CUP_SUCCESS_Z = CUP_REST_Z + CUP_LIFT_M
CUP_HOLD_M = 0.12
CUP_GRIP = 0.5
CUP_Z = 0.754
# Table center (0.65, 0), half-size (0.32, 0.45). Inset keeps a yawed cup on top.
CUP_EVAL_X = (0.43, 0.87)
CUP_EVAL_Y = (-0.35, 0.35)
CUP_EVAL_YAW = (-np.pi, np.pi)

T_Z = 0.101
TARGET_Z = 0.002
T_DEMO_XYZ = (0.65, 0.0, T_Z)
TARGET_DEMO_XYZ = (1.55, 0.0, TARGET_Z)
T_X = (0.55, 0.95)
T_Y = (-0.25, 0.25)
T_YAW = (-0.4, 0.4)
TARGET_AHEAD = (0.60, 0.90)
TARGET_LATERAL = (-0.20, 0.20)
TARGET_YAW = (-0.4, 0.4)
COVERAGE_MIN = 0.50
GRID_M = 0.02

ROOT_XY = (-0.05, 0.05)
ROOT_YAW = (-0.15, 0.15)

# Body-frame rectangles of the T: center x, center y, half-extent x, half-extent y.
_PARTS = ((0.12, 0.0, 0.09, 0.30), (-0.12, 0.0, 0.15, 0.09))


def _footprint_points(step=GRID_M):
    xs = np.arange(-0.27 + step / 2, 0.21, step)
    ys = np.arange(-0.30 + step / 2, 0.30, step)
    points = [(x, y) for x in xs for y in ys if _inside_t(x, y)]
    if not points:
        raise RuntimeError("T footprint grid is empty")
    return np.asarray(points, dtype=np.float64)


def _inside_t(x, y):
    return any(abs(x - cx) <= hx and abs(y - cy) <= hy for cx, cy, hx, hy in _PARTS)


_POINTS = _footprint_points()


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def _draw(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))


def criteria(task):
    """The numeric rule stored with every trial of this task.

    The episode horizon is ``timeout_s`` on the trial. The declared evaluation
    horizon is ``TIMEOUT_S``.
    """
    common = {"fall_height_m": FALL_HEIGHT_M, "fall_tilt_deg": FALL_TILT_DEG}
    if task == "cup":
        return common | {
            "success": "pickup",
            "cup_z_m": CUP_SUCCESS_Z,
            "tip_distance_m": CUP_HOLD_M,
            "grip": CUP_GRIP,
        }
    if task == "push_t":
        return common | {"success": "coverage", "coverage": COVERAGE_MIN}
    raise ValueError(f"Unknown task: {task}")


def sample_start(task, seed, distribution="eval"):
    """One initial pose. The same seed and distribution always return the same start.

    ``demo`` for the cup is the reach box used by the sixteen demonstrations,
    with the robot left at the standing reset. ``eval`` widens the cup across
    the inset table, draws a yaw, and shifts the root by a few centimeters.
    ``demo`` for push-T is the scene pose. ``eval`` moves the T and the mark.
    """
    if distribution not in ("demo", "eval"):
        raise ValueError(f"Unknown distribution: {distribution}")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if task == "cup":
        if distribution == "demo":
            from .reach import sample_cup
            cup = sample_cup(rng)
            yaw, root_x, root_y, root_yaw = 0.0, 0.0, 0.0, 0.0
        else:
            cup = np.array([_draw(rng, CUP_EVAL_X), _draw(rng, CUP_EVAL_Y), CUP_Z], dtype=np.float64)
            yaw = _draw(rng, CUP_EVAL_YAW)
            root_x, root_y, root_yaw = _draw(rng, ROOT_XY), _draw(rng, ROOT_XY), _draw(rng, ROOT_YAW)
        return {
            "task": "cup", "distribution": distribution, "seed": seed,
            "cup_xyz": [float(v) for v in cup], "cup_yaw": float(yaw),
            "root_x": float(root_x), "root_y": float(root_y), "root_yaw": float(root_yaw),
        }
    if task == "push_t":
        if distribution == "demo":
            t_xyz, t_yaw = T_DEMO_XYZ, 0.0
            target_xyz, target_yaw = TARGET_DEMO_XYZ, 0.0
            root_x, root_y, root_yaw = 0.0, 0.0, 0.0
        else:
            t_xyz = (_draw(rng, T_X), _draw(rng, T_Y), T_Z)
            t_yaw = _draw(rng, T_YAW)
            target_xyz = (t_xyz[0] + _draw(rng, TARGET_AHEAD), t_xyz[1] + _draw(rng, TARGET_LATERAL), TARGET_Z)
            target_yaw = _draw(rng, TARGET_YAW)
            root_x, root_y, root_yaw = _draw(rng, ROOT_XY), _draw(rng, ROOT_XY), _draw(rng, ROOT_YAW)
        return {
            "task": "push_t", "distribution": distribution, "seed": seed,
            "t_xyz": [float(v) for v in t_xyz], "t_yaw": float(t_yaw),
            "target_xyz": [float(v) for v in target_xyz], "target_yaw": float(target_yaw),
            "root_x": float(root_x), "root_y": float(root_y), "root_yaw": float(root_yaw),
        }
    raise ValueError(f"Unknown task: {task}")


def _place_free(model, data, joint_name, xyz, yaw):
    joint = model.joint(joint_name)
    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[qadr:qadr+3] = xyz
    data.qpos[qadr+3:qadr+7] = _yaw_quat(yaw)
    data.qvel[dadr:dadr+6] = 0


def _place_body(model, name, xyz, yaw):
    body = model.body(name).id
    model.body_pos[body] = xyz
    model.body_quat[body] = _yaw_quat(yaw)


def reset(model, data, controller, start):
    """Return the simulator and the tracker to ``start``. A second call matches the first."""
    controller.initialize(data)
    data.qpos[0] = float(start["root_x"])
    data.qpos[1] = float(start["root_y"])
    data.qpos[3:7] = _yaw_quat(float(start["root_yaw"]))
    data.qvel[:6] = 0
    task = start["task"]
    if task == "cup":
        _place_free(model, data, "cup_free", start["cup_xyz"], start["cup_yaw"])
    elif task == "push_t":
        _place_free(model, data, "t_box_free", start["t_xyz"], start["t_yaw"])
        _place_body(model, "t_target", start["target_xyz"], start["target_yaw"])
    else:
        raise ValueError(f"Unknown task: {task}")
    mujoco.mj_forward(model, data)
    return start


def fell(model, data):
    """Pelvis below 0.55 m, tilt past 45 degrees, or a non-finite pose."""
    if not np.isfinite(data.qpos).all():
        return True
    height = float(data.qpos[2])
    tilt = float(np.arccos(np.clip(data.xmat[model.body("pelvis").id, 8], -1, 1)))
    return bool(height < FALL_HEIGHT_M or tilt > np.radians(FALL_TILT_DEG))


def cup_picked(model, data, grip):
    """True when the cup is lifted and the right hand is closed on it."""
    grip = float(grip)
    if not np.isfinite(grip):
        raise ValueError("grip must be finite")
    cup = data.body("cup").xpos
    distance = float(np.linalg.norm(cup - data.body(TIP).xpos))
    return bool(float(cup[2]) >= CUP_SUCCESS_Z and distance <= CUP_HOLD_M and grip >= CUP_GRIP)


def coverage(data):
    """Fraction of the target footprint that lies inside the T. Identity overlap is 1."""
    origin = np.asarray(data.body("t_target").xpos, dtype=np.float64)
    rotation = np.asarray(data.body("t_target").xmat, dtype=np.float64).reshape(3, 3)
    local = np.column_stack((_POINTS, np.zeros(len(_POINTS))))
    world = origin + local @ rotation.T
    box_origin = np.asarray(data.body("t_box").xpos, dtype=np.float64)
    box_rotation = np.asarray(data.body("t_box").xmat, dtype=np.float64).reshape(3, 3)
    local_box = (world - box_origin) @ box_rotation
    inside = np.fromiter((_inside_t(x, y) for x, y in local_box[:, :2]), dtype=bool, count=len(local_box))
    return float(inside.mean())


def _verdict(model, data, task, grip):
    if fell(model, data):
        return "fell"
    if task == "cup" and cup_picked(model, data, grip):
        return "success"
    if task == "push_t" and coverage(data) >= COVERAGE_MIN:
        return "success"
    return None


def _signals(model, data, task, grip):
    tilt = float(np.arccos(np.clip(data.xmat[model.body("pelvis").id, 8], -1, 1)))
    row = {
        "pelvis_height_m": float(data.qpos[2]),
        "pelvis_tilt_deg": float(np.degrees(tilt)),
        "grip": float(grip),
        "verdict": _verdict(model, data, task, grip),
    }
    if task == "cup":
        row["cup_z_m"] = float(data.body("cup").xpos[2])
        row["tip_distance_m"] = float(np.linalg.norm(data.body("cup").xpos - data.body(TIP).xpos))
    else:
        row["coverage"] = coverage(data)
    return row


def _source(name, model, start):
    if name == "standing":
        from .reach import Standing
        return Standing()
    if name == "script":
        from .reach import ScriptedReach
        if start["task"] != "cup":
            raise ValueError("The scripted reach is only a cup source")
        return ScriptedReach(model, np.asarray(start["cup_xyz"], dtype=np.float64))
    raise ValueError(f"Unknown source: {name}")


def run(task, seed, seconds=TIMEOUT_S, distribution="eval", source="standing", record=None):
    """Roll one seeded episode and stop at success, a fall, or the timeout."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    model, xml = make_model(task)
    if task == "cup" and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TIP) < 0:
        raise ValueError("The cup task needs articulated hands")
    data = mujoco.MjData(model)
    controller = Controller(model)
    start = reset(model, data, controller, sample_start(task, seed, distribution))
    command_source = _source(source, model, start)
    command_source.reset()
    observed = [_signals(model, data, task, controller.grip)]
    recording = []
    outcome = observed[-1]["verdict"]
    while outcome is None and data.time < seconds:
        command, grip = command_source(model, data, controller)
        command = np.asarray(command, dtype=np.float64)
        grip = float(grip)
        controller.step(data, command, grip)
        observed.append(_signals(model, data, task, controller.grip))
        outcome = observed[-1]["verdict"]
        if record is not None and controller.step_index % controller.period == 0:
            recording.append((data.time, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), command.copy(), grip))
    if outcome is None:
        outcome = "timeout"
    success_time = float(data.time) if outcome == "success" else None
    metrics = {
        "schema": SCHEMA,
        "task": task,
        "distribution": distribution,
        "source": source,
        "seed": int(seed),
        "seconds": float(data.time),
        "timeout_s": float(seconds),
        "outcome": outcome,
        "success": outcome == "success",
        "success_time_s": success_time,
        "criteria": criteria(task),
        "start": start,
        "minimum_pelvis_height_m": min(row["pelvis_height_m"] for row in observed),
        "maximum_pelvis_tilt_deg": max(row["pelvis_tilt_deg"] for row in observed),
        "mujoco_warnings": int(sum(w.number for w in data.warning)),
    }
    if task == "cup":
        metrics["initial_tip_distance_m"] = observed[0]["tip_distance_m"]
        metrics["minimum_tip_distance_m"] = min(row["tip_distance_m"] for row in observed)
        metrics["final_tip_distance_m"] = observed[-1]["tip_distance_m"]
        metrics["maximum_cup_z_m"] = max(row["cup_z_m"] for row in observed)
    else:
        metrics["initial_coverage"] = observed[0]["coverage"]
        metrics["maximum_coverage"] = max(row["coverage"] for row in observed)
        metrics["final_coverage"] = observed[-1]["coverage"]
    if record is not None:
        record = Path(record)
        record.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            record, time=[row[0] for row in recording], qpos=[row[1] for row in recording],
            qvel=[row[2] for row in recording], ctrl=[row[3] for row in recording],
            command=[row[4] for row in recording], grip=[row[5] for row in recording],
            scene_xml=xml, scene=task, metadata=json.dumps(metrics))
        metrics["recording"] = str(record)
    return metrics


def _line(metrics):
    if metrics["task"] == "cup":
        detail = (f"tip {metrics['initial_tip_distance_m']:.3f}->{metrics['final_tip_distance_m']:.3f} "
                  f"cup_z {metrics['maximum_cup_z_m']:.3f}")
    else:
        detail = f"coverage {metrics['initial_coverage']:.3f}->{metrics['maximum_coverage']:.3f}"
    return (f"{metrics['task']} {metrics['distribution']} {metrics['source']} seed {metrics['seed']} "
            f"{metrics['outcome']} {detail}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score seeded cup and push-T rollouts")
    parser.add_argument("--task", choices=("cup", "push_t"), required=True)
    parser.add_argument("--distribution", choices=("demo", "eval"), default="eval")
    parser.add_argument("--source", choices=("standing", "script"), default="standing")
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=TIMEOUT_S)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args(argv)
    trials = []
    for seed in args.seed:
        record = None if args.record is None else args.record / f"{args.task}_{seed:03d}.npz"
        metrics = run(args.task, seed, args.seconds, args.distribution, args.source, record)
        trials.append(metrics)
        print(_line(metrics), flush=True)
    summary = {
        "schema": SCHEMA,
        "task": args.task,
        "distribution": args.distribution,
        "source": args.source,
        "outcomes": {name: sum(row["outcome"] == name for row in trials) for name in ("success", "timeout", "fell")},
        "trials": trials,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
