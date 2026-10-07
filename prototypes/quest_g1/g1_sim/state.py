"""Versioned low-dimensional state for desktop policy learning.

``state-v1`` reads a desktop NPZ by joint and body name. The raw episode is
left untouched. Export rows copy recorded rows, so the stored command keeps
the one-step lead described in the recording contract and is not interpolated.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .controller import JOINTS, euler

SCHEMA = "state-v1"
RECORD_HZ = 100
ACTION_HZ = 20
REQUIRED_FIELDS = ("time", "qpos", "qvel", "command", "grip", "scene", "scene_xml")
# Stable prefix. Scene bodies are appended in this order, then grip.
ROBOT_NAMES = (
    "root_z", "root_roll", "root_pitch",
    "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
    "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
    *JOINTS,
)
SCENE_BODIES = (("cup", "cup"), ("t_box", "t_box"), ("t_target", "t_target"))
COMMAND_NAMES = ("vx", "vy", "z", "roll", "pitch", "yaw_rate", *[f"q_{name}" for name in JOINTS])


def body_field_names(prefix):
    """Position and the first two columns of the pelvis-frame rotation, row-major."""
    return tuple(f"{prefix}_pos_{axis}" for axis in "xyz") + tuple(f"{prefix}_rot6d_{i}" for i in range(6))


def included_bodies(model):
    return tuple((body, prefix) for body, prefix in SCENE_BODIES
                 if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0)


def layout_names(model):
    names = list(ROBOT_NAMES)
    for _, prefix in included_bodies(model):
        names.extend(body_field_names(prefix))
    names.append("grip")
    return tuple(names)


def encode_state(model, data, grip):
    """One observation for the current ``qpos`` and ``qvel``.

    Root roll and pitch use the TWIST2 Euler pair. Linear velocity is the
    free-joint velocity rotated into the pelvis frame; angular velocity is
    already expressed there. Object and goal poses use that same frame.
    """
    grip = float(grip)
    if not np.isfinite(grip):
        raise ValueError("grip must be finite")
    root = model.joint("pelvis")
    if int(model.jnt_type[root.id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("pelvis must be a free joint")
    qadr, vadr = int(root.qposadr[0]), int(root.dofadr[0])
    mujoco.mj_forward(model, data)
    pelvis = model.body("pelvis").id
    rotation = data.xmat[pelvis].reshape(3, 3)
    roll, pitch, _ = euler(data.qpos[qadr+3:qadr+7])
    linear = rotation.T @ data.qvel[vadr:vadr+3]
    angular = np.asarray(data.qvel[vadr+3:vadr+6], dtype=np.float64)
    joints = [data.qpos[int(model.joint(name).qposadr[0])] for name in JOINTS]
    values = [data.qpos[qadr+2], roll, pitch, *linear, *angular, *joints]
    origin = data.xpos[pelvis]
    for body_name, _ in included_bodies(model):
        body = model.body(body_name).id
        relative = rotation.T @ (data.xpos[body] - origin)
        relative_rotation = rotation.T @ data.xmat[body].reshape(3, 3)
        values.extend(relative)
        values.extend(relative_rotation[:, :2].reshape(6))
    values.append(grip)
    state = np.asarray(values, dtype=np.float64)
    if state.shape != (len(layout_names(model)),) or not np.isfinite(state).all():
        raise ValueError("State is non-finite or does not match the layout")
    return state


def scalar_text(value):
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("Expected a scalar text field")
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise ValueError("Expected a text field")
    return value


def convert_recording(source, destination, action_hz=ACTION_HZ):
    """Write a derived ``state-v1`` export. The source NPZ is not opened for writing."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the raw episode with a training export")
    if destination.suffix != ".npz":
        raise ValueError("Training export must be an .npz file")
    if isinstance(action_hz, bool) or int(action_hz) < 1 or RECORD_HZ % int(action_hz):
        raise ValueError(f"action_hz must be a positive divisor of the {RECORD_HZ} Hz record")
    action_hz = int(action_hz)
    with np.load(source, allow_pickle=False) as episode:
        missing = [key for key in REQUIRED_FIELDS if key not in episode.files]
        if missing:
            raise ValueError("Recording is missing " + ", ".join(missing))
        time = np.asarray(episode["time"], dtype=np.float64).copy()
        qpos = np.asarray(episode["qpos"], dtype=np.float64).copy()
        qvel = np.asarray(episode["qvel"], dtype=np.float64).copy()
        command = np.asarray(episode["command"], dtype=np.float64).copy()
        grip = np.asarray(episode["grip"], dtype=np.float64).copy()
        scene = scalar_text(episode["scene"])
        xml = scalar_text(episode["scene_xml"])
        source_metadata = scalar_text(episode["metadata"]) if "metadata" in episode.files else ""
    rows = time.shape[0]
    if time.ndim != 1 or rows < 1 or not np.isfinite(time).all():
        raise ValueError("time must be a finite 1D vector")
    if rows > 1 and not np.allclose(np.diff(time), 1 / RECORD_HZ, rtol=0, atol=1e-8):
        raise ValueError("Recording time must advance in uniform 0.010 s steps")
    if command.shape != (rows, len(COMMAND_NAMES)) or grip.shape != (rows,) or not np.isfinite(command).all() or not np.isfinite(grip).all():
        raise ValueError(f"command must be (N, {len(COMMAND_NAMES)}) and grip (N,), both finite")
    model = mujoco.MjModel.from_xml_string(xml)
    if qpos.shape != (rows, model.nq) or qvel.shape != (rows, model.nv) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("qpos and qvel must be finite and match scene_xml")
    names = layout_names(model)
    index = np.arange(0, rows, RECORD_HZ // action_hz, dtype=np.int64)
    data = mujoco.MjData(model)
    state = np.empty((len(index), len(names)), dtype=np.float64)
    for row, sample in enumerate(index):
        data.qpos[:] = qpos[sample]
        data.qvel[:] = qvel[sample]
        state[row] = encode_state(model, data, grip[sample])
    metadata = {
        "schema": SCHEMA,
        "scene": scene,
        "record_hz": RECORD_HZ,
        "action_hz": action_hz,
        "rows": int(len(index)),
        "state_dim": len(names),
        "source": str(source.resolve()),
        "alignment": "Export rows copy recorded rows. command is the stored p_cmd and leads time by one physics step. No interpolation.",
        "frames": "Root roll/pitch are the TWIST2 Euler pair. Root linear velocity and object poses are in the pelvis frame. Root angular velocity is the free-joint angular velocity. Orientations are 6D: the first two rotation-matrix columns, row-major.",
        "source_metadata": source_metadata,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, state=state, state_names=np.array(names), command=command[index],
        command_names=np.array(COMMAND_NAMES), grip=grip[index], time=time[index],
        source_index=index, scene=np.array(scene), schema=np.array(SCHEMA),
        metadata=json.dumps(metadata))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a desktop NPZ recording into a state-v1 training export")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--action-hz", type=int, default=ACTION_HZ)
    args = parser.parse_args(argv)
    print(json.dumps(convert_recording(args.source, args.destination, args.action_hz), indent=2), flush=True)


if __name__ == "__main__":
    main()
