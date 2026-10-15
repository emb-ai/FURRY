"""GMR retargeting of a camera-only body pose (body2d.Pose) onto the G1, and its render.

Uses the pinned upstream GMR/Mink solver in prototypes/quest_g1/vendor/GMR
(the APK runs a native port checked against it by check_gmr_parity.py), with
its xrobot -> G1 task tables. Like the headset adapter (meta_retarget.cpp),
targets are aligned to the robot's own rest pose instead of the xrobot frame
conventions: each segment's frame is built from joint positions, and a task
target is the segment's rotation from a T pose applied to the robot link's
orientation in the same T pose (the headset's calibration pose: shoulder roll
+-90 deg, elbows 90 deg). Both T-pose frames are built by the same function,
so no frame convention has to match a human skeleton's.

Positions are the human's, scaled by robot/human limb length (legs and torso
by leg length, arms by arm length), with the floors aligned. World mapping:
human world (Y up, body2d) -> robot world (Z up) by HUMAN_TO_ROBOT, then that
scale. The camera that saw the operator maps the same way, so the robot can be
rendered from the same viewpoint.
"""
import importlib.util
import sys
import types
from pathlib import Path

import mujoco
import numpy as np

GMR_ROOT = Path(__file__).resolve().parents[1] / "prototypes/quest_g1/vendor/GMR"
HUMAN_TO_ROBOT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])  # (x, y, z)_W -> (-z, -x, y)
Z = np.array([0., 0., 1.])

# xrobot human body -> (robot task body, segment, human point).
TASKS = {
    "Pelvis": ("pelvis", "pelvis", "pelvis"),
    "Spine3": ("torso_link", "torso", "chest"),
    "Left_Hip": ("left_hip_yaw_link", "lthigh", "lhip"),
    "Right_Hip": ("right_hip_yaw_link", "rthigh", "rhip"),
    "Left_Knee": ("left_knee_link", "lshin", "lknee"),
    "Right_Knee": ("right_knee_link", "rshin", "rknee"),
    "Left_Foot": ("left_toe_link", "lfoot", "lankle"),
    "Right_Foot": ("right_toe_link", "rfoot", "rankle"),
    "Left_Shoulder": ("left_shoulder_yaw_link", "lupper", "lshoulder"),
    "Right_Shoulder": ("right_shoulder_yaw_link", "rupper", "rshoulder"),
    "Left_Elbow": ("left_elbow_link", "lfore", "lelbow"),
    "Right_Elbow": ("right_elbow_link", "rfore", "relbow"),
    "Left_Wrist": ("left_wrist_yaw_link", "lfore", "lwrist"),
    "Right_Wrist": ("right_wrist_yaw_link", "rfore", "rwrist"),
}
# Robot bodies standing in for the human joints at rest.
ROBOT_POINTS = {
    "pelvis": "pelvis", "lhip": "left_hip_pitch_link", "rhip": "right_hip_pitch_link",
    "lknee": "left_knee_link", "rknee": "right_knee_link",
    "lankle": "left_ankle_roll_link", "rankle": "right_ankle_roll_link",
    "lshoulder": "left_shoulder_pitch_link", "rshoulder": "right_shoulder_pitch_link",
    "lelbow": "left_elbow_link", "relbow": "right_elbow_link",
    "lwrist": "left_wrist_yaw_link", "rwrist": "right_wrist_yaw_link",
}
STANDING_PELVIS_Z = .793  # G1 pelvis above the floor, legs straight


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _orthogonal(v, axis):
    return v - (v @ axis) * axis


def _frame(x, y, z):
    return np.stack([x, y, z], 1)


def _hinge(proximal, distal, natural):
    """Hinge axis cross(proximal, distal), signed like `natural`.

    A straight limb has no bend plane: it takes `natural`, the axis of a
    slight flexion in its usual direction, so the twist about the bone stays
    continuous. A slightly overextended joint keeps the flexion sign.
    """
    natural = _unit(_orthogonal(natural, proximal))
    normal = np.cross(proximal, distal)
    size = np.linalg.norm(normal)
    if size < 1e-9:
        return natural
    normal = normal / size
    if normal @ natural < 0:
        normal = -normal
    w = min(1., size / np.sin(np.radians(20)))
    return _unit(w * normal + (1 - w) * natural)


def _arm_natural(upper, left, up, side):
    """Elbow axis of a slight flexion, continuous over the arm's range.

    In the sagittal plane it is the torso's lateral axis (pointing right for
    either arm); an arm raised sideways carries it about the forward axis to
    down (left arm) or up (right arm) in the T pose, where the lateral axis
    runs along the arm.
    """
    lateral = _orthogonal(-left, upper)
    w = min(1., np.linalg.norm(lateral) ** 2)
    vertical = _orthogonal(-up if side == "l" else up, upper)
    if np.linalg.norm(vertical) < 1e-9:
        return _unit(lateral)
    if w < 1e-12:
        return _unit(vertical)
    return _unit(w * _unit(lateral) + (1 - w) * _unit(vertical))


def segment_frames(p):
    """Rotation (columns x, y, z) of every segment from joint positions (Z up)."""
    out = {}
    up = _unit(p["chest"] - p["pelvis"])
    left = _unit(p["lhip"] - p["rhip"])
    pz = _unit(_orthogonal(Z, left))
    out["pelvis"] = _frame(np.cross(left, pz), left, pz)
    sl = _unit(p["lshoulder"] - p["rshoulder"])
    tz = _unit(_orthogonal(up, sl))
    out["torso"] = _frame(np.cross(sl, tz), sl, tz)
    foot_y = _unit(_orthogonal(left, Z))
    out["lfoot"] = out["rfoot"] = _frame(np.cross(foot_y, Z), foot_y, Z)
    for side in "lr":
        thigh = _unit(p[side + "knee"] - p[side + "hip"])
        shin = _unit(p[side + "ankle"] - p[side + "knee"])
        # The knee hinge is the pelvis's lateral axis (flexion moves the shin back).
        hinge = _hinge(thigh, shin, left)
        for name, a in ((side + "thigh", thigh), (side + "shin", shin)):
            y = _unit(_orthogonal(hinge, a))
            out[name] = _frame(np.cross(y, -a), y, -a)
        upper = _unit(p[side + "elbow"] - p[side + "shoulder"])
        fore = _unit(p[side + "wrist"] - p[side + "elbow"])
        hinge = _hinge(upper, fore, _arm_natural(upper, sl, tz, side))
        for name, a in ((side + "upper", upper), (side + "fore", fore)):
            y = _unit(_orthogonal(hinge, a))
            out[name] = _frame(np.cross(y, -a), y, -a)
    return out


def canonical_t_pose(body):
    """Human T pose with the operator's lengths, facing +X (Z up), pelvis at 0."""
    p = {"pelvis": np.zeros(3), "chest": Z * body.torso}
    for side, y in (("l", 1.), ("r", -1.)):
        p[side + "hip"] = np.array([0., body.hip_half * y, 0.])
        p[side + "knee"] = p[side + "hip"] - Z * body.thigh
        p[side + "ankle"] = p[side + "knee"] - Z * body.shin
        p[side + "shoulder"] = p["chest"] + np.array([0., body.shoulder_half * y, 0.])
        p[side + "elbow"] = p[side + "shoulder"] + np.array([0., body.upper_arm * y, 0.])
        p[side + "wrist"] = p[side + "elbow"] + np.array([0., body.forearm * y, 0.])
    return p


def load_gmr(height):
    """Upstream GeneralMotionRetargeting('xrobot', 'unitree_g1'), imported like check_gmr_parity.py."""
    name = "gmr_live_upstream"
    if name + ".motion_retarget" not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(GMR_ROOT / "general_motion_retargeting")]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + ".motion_retarget",
                                                      Path(package.__path__[0]) / "motion_retarget.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name + ".motion_retarget"]

    class AlignedTargets(module.GeneralMotionRetargeting):
        # Targets arrive already aligned to the robot links (task body -> (pos, R)):
        # upstream scaling and xrobot frame offsets are bypassed, the IK is not.
        def update_targets(self, targets, offset_to_ground=False):
            for table in (self.human_body_to_task1, self.human_body_to_task2):
                for human, task in table.items():
                    position, rotation = targets[human]
                    task.set_target(module.mink.SE3.from_rotation_and_translation(
                        module.mink.SO3.from_matrix(rotation), position))

    return AlignedTargets("xrobot", "unitree_g1", actual_human_height=height, verbose=False)


class Retargeter:
    def __init__(self, body, height=None):
        """body: body2d.Body (bone lengths); the robot is driven in its proportions."""
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):  # upstream prints its tables
            self.gmr = load_gmr(height or body.height)
        self.model = self.gmr.model
        self.data = mujoco.MjData(self.model)
        self.body = body
        self._calibrate()

    def _fk(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return ({k: self.data.body(v).xpos.copy() for k, v in ROBOT_POINTS.items()},
                {h: (self.data.body(t).xpos.copy(), self.data.body(t).xmat.reshape(3, 3).copy())
                 for h, (t, _, _) in TASKS.items()})

    def _calibrate(self):
        m = self.model
        q = np.zeros(m.nq)
        q[2], q[3] = STANDING_PELVIS_Z, 1.
        points, _ = self._fk(q)
        b = self.body
        leg = np.linalg.norm(points["lknee"] - points["lhip"]) + np.linalg.norm(points["lankle"] - points["lknee"])
        arm = (np.linalg.norm(points["lelbow"] - points["lshoulder"])
               + np.linalg.norm(points["lwrist"] - points["lelbow"]))
        self.leg_scale = leg / (b.thigh + b.shin)
        self.arm_scale = arm / (b.upper_arm + b.forearm)
        # Robot hip centre height for the human pelvis (hip line) on the floor.
        self.floor_offset = (points["lhip"][2] + points["rhip"][2]) / 2 - self.leg_scale * b.pelvis_height
        # Rest: arms out sideways, elbows bent (the headset's calibration pose).
        for side, sign in (("left", 1), ("right", -1)):
            q[m.jnt_qposadr[m.joint(side + "_shoulder_roll_joint").id]] = sign * np.pi / 2
            q[m.jnt_qposadr[m.joint(side + "_elbow_joint").id]] = np.pi / 2
        points, tasks = self._fk(q)
        # Segment frames of the matching human T pose: legs straight down, arms
        # straight out sideways, facing +X. Not the robot's own link points:
        # G1 links have offsets (the hip-to-knee line leans 9 deg outward).
        human = canonical_t_pose(self.body)
        rest = segment_frames(human)
        human = self._scaled(human)
        self.rest_q = q.copy()
        # Per task: rest link rotation, rest segment frame, and the link origin
        # in its own frame relative to the scaled human point, as on the headset
        # (meta_retarget.cpp): the difference between robot and human geometry
        # rides with the link.
        hips = (points["lhip"] + points["rhip"]) / 2
        self.rest = {}
        for name, (_, segment, point) in TASKS.items():
            position, rotation = tasks[name]
            self.rest[name] = (rotation, rest[segment], rotation.T @ (position - hips - human[point]))
        self.gmr.configuration.update(q)

    def _scaled(self, p):
        """Points relative to the pelvis, in robot proportions: legs and torso by
        leg length, arms by arm length about their shoulders."""
        out = {n: self.leg_scale * (v - p["pelvis"]) for n, v in p.items()}
        for side in "lr":
            shoulder = p[side + "shoulder"]
            for joint in ("elbow", "wrist"):
                out[side + joint] = out[side + "shoulder"] + self.arm_scale * (p[side + joint] - shoulder)
        return out

    def reset(self):
        self.gmr.configuration.update(self.rest_q.copy())

    def to_robot(self, p_world):
        """Human world point (Y up) to the scaled robot world."""
        r = self.leg_scale * (HUMAN_TO_ROBOT @ p_world)
        r[2] += self.floor_offset
        return r

    def targets(self, pose):
        """Task targets {xrobot body: (position, rotation)} for a body2d.Pose."""
        world = {n: HUMAN_TO_ROBOT @ v for n, v in pose.points.items()}
        frames = segment_frames(world)
        relative = self._scaled(world)
        root = self.to_robot(pose.points["pelvis"])
        out = {}
        for name, (_, segment, point) in TASKS.items():
            link_rest, segment_rest, local = self.rest[name]
            rotation = frames[segment] @ segment_rest.T @ link_rest
            out[name] = (root + relative[point] + rotation @ local, rotation)
        return out

    def solve(self, pose):
        """Robot qpos for a body2d.Pose (the solver warm-starts from the last one)."""
        targets = self.targets(pose)
        q = self.gmr.configuration.q
        position, rotation = targets["Pelvis"]
        current = np.zeros(9)
        mujoco.mju_quat2Mat(current, q[3:7])
        turn = rotation @ current.reshape(3, 3).T
        # The IK is local: a large turn (first frame, re-acquired operator)
        # starts from the rest pose at the target heading instead.
        if np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1))) > 60:
            start = self.rest_q.copy()
            start[:3] = position
            mujoco.mju_mat2Quat(start[3:7], rotation.ravel())
            self.gmr.configuration.update(start)
        return self.gmr.retarget(targets)

    def camera_view(self, camera):
        """MuJoCo free-camera parameters matching a body2d.Camera."""
        position = self.to_robot(camera.t)
        forward = HUMAN_TO_ROBOT @ camera.forward
        distance = 3. * self.leg_scale
        return dict(lookat=position + forward * distance, distance=distance,
                    azimuth=float(np.degrees(np.arctan2(forward[1], forward[0]))),
                    elevation=float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1)))),
                    fovy=float(np.degrees(2 * np.arctan(camera.cy / camera.fy))))


class RobotView:
    """Offscreen MuJoCo render of the retargeted robot from the operator's camera."""

    def __init__(self, model, width=640, height=360):
        self.model, self.data = model, mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, qpos, view):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.model.vis.global_.fovy = view["fovy"]
        self.camera.lookat[:] = view["lookat"]
        self.camera.distance = view["distance"]
        self.camera.azimuth = view["azimuth"]
        self.camera.elevation = view["elevation"]
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
