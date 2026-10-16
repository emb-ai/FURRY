"""Live viewer: GMR driven by one RGB camera's 2D keypoints, no depth, no headset.

Left: the colour frame with YOLO keypoints (dots) and the solved body
reprojected (lines). Right: the G1 after GMR, rendered from the same camera
viewpoint, so both panels show the same pose from the same place.

Pipeline per frame: YOLO11-pose -> person track -> undistorted 2D keypoints
-> body2d.BodySolver (bones from --height, depth axis regularised, floor from
the camera calibration) -> One-Euro -> gmr_live.Retargeter (upstream GMR/Mink)
-> MuJoCo render.

Camera placement (the floor is what fixes the depth axis):
  - default: D435i accelerometer for gravity, plus --camera-height (lens
    above the floor, m). Without an IMU reading, --pitch (deg down) is used.
  - --session session.json: the AprilTag camera->STAGE lock from pose_yolo.py.

Run on macOS (RealSense needs sudo there):
  sudo python gmr_view.py s --camera-height 1.20 --height 1.78
Without a camera, a synthetic operator checks the robot side:
  python gmr_view.py --source synthetic

Keys: q quit, r reset the solve, [ ] camera height -/+ 2 cm,
; ' pitch -/+ 1 deg (when not from the IMU or a session).
Stand 2.5-4 m away, whole body in view, nobody else in frame. A camera
30-45 deg off your front resolves forward/backward leg motion far better.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from body2d import COCO, INDEX, NAMES, Body, BodySolver, Camera, camera_from_gravity, camera_from_pitch
from keypoints2d import wire_keypoints

WINDOW = "GMR from 2D keypoints"
BONES = [("lhip", "rhip"), ("lshoulder", "rshoulder"), ("lhip", "lshoulder"), ("rhip", "rshoulder"),
         ("lhip", "lknee"), ("lknee", "lankle"), ("rhip", "rknee"), ("rknee", "rankle"),
         ("lshoulder", "lelbow"), ("lelbow", "lwrist"), ("rshoulder", "relbow"), ("relbow", "rwrist")]
LEFT, RIGHT, CENTRE = (80, 200, 60), (60, 120, 255), (235, 235, 235)  # BGR, as pose_yolo.py
LIMB_NAMES = ["L leg", "R leg", "L arm", "R arm"]


class OneEuro:
    """fusion.OneEuro without its pyrealsense2 import (the synthetic source runs without the SDK)."""

    def __init__(self, mincutoff, beta, dcutoff=1.):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.x = self.t = None
        self.dx = np.zeros(3)

    @staticmethod
    def _alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))

    def __call__(self, x, t):
        if self.x is None or not 0 < t - self.t <= .5:
            self.x, self.t, self.dx = x.copy(), t, np.zeros(3)
            return x.copy()
        dt = t - self.t
        self.dx = self.dx + self._alpha(self.dc, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.mc + self.beta * np.linalg.norm(self.dx)
        self.x, self.t = self.x + self._alpha(cutoff, dt) * (x - self.x), t
        return self.x.copy()


def colour(a, b):
    return LEFT if a[0] == "l" and b[0] == "l" else RIGHT if a[0] == "r" and b[0] == "r" else CENTRE


def draw_overlay(image, keypoints, pose, camera, min_conf):
    """Detected keypoints as dots, the solved body reprojected as lines."""
    if pose is not None:
        uv, z = camera.project(np.array([pose.points[n] for n in NAMES]))
        for a, b in BONES:
            if z[INDEX[a]] > .1 and z[INDEX[b]] > .1:
                cv2.line(image, tuple(np.int32(uv[INDEX[a]])), tuple(np.int32(uv[INDEX[b]])), colour(a, b), 3,
                         cv2.LINE_AA)
    for i, (u, v, c) in keypoints.items():
        if c >= min_conf:
            cv2.circle(image, (int(u), int(v)), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, (int(u), int(v)), 4, (255, 255, 255), -1, cv2.LINE_AA)


def text(image, lines, origin=(12, 24), scale=.5, colour_=(60, 255, 60)):
    """Lines on a dark band (an outline stroke would widen Hershey glyph spacing)."""
    step = int(24 * scale / .5)
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines)
    top = origin[1] - step + 6
    band = image[max(0, top):origin[1] + step * (len(lines) - 1) + 8, max(0, origin[0] - 6):origin[0] + width + 6]
    band[:] = (band * .35).astype(np.uint8)
    for n, line in enumerate(lines):
        cv2.putText(image, line, (origin[0], origin[1] + n * step), cv2.FONT_HERSHEY_SIMPLEX, scale, colour_, 1,
                    cv2.LINE_AA)


class Placement:
    """Camera in the world: gravity or pitch plus height, or a saved STAGE lock."""

    def __init__(self, args, intrinsics):
        self.args, self.intrinsics = args, intrinsics
        self.height, self.pitch, self.up = args.camera_height, args.pitch, None
        self.session = None
        if args.session:
            data = json.loads(Path(args.session).read_text(encoding="utf-8"))
            self.session = (np.asarray(data["R"], float), np.asarray(data["t"], float))

    def camera(self):
        fx, fy, cx, cy = self.intrinsics
        if self.session is not None:
            return Camera(self.session[0], self.session[1], fx, fy, cx, cy)
        if self.up is not None:
            return camera_from_gravity(self.up, self.height, fx, fy, cx, cy)
        return camera_from_pitch(self.pitch, self.height, fx, fy, cx, cy)

    def describe(self):
        if self.session is not None:
            return "camera: AprilTag STAGE lock"
        cam = self.camera()
        tilt = np.degrees(np.arcsin(-cam.forward[1]))
        source = "IMU" if self.up is not None else "--pitch"
        return f"camera: {self.height:.2f} m high, {tilt:.1f} deg down ({source})"


# ---- sources ------------------------------------------------------------------
class RealSense:
    def __init__(self, args):
        import pyrealsense2 as rs
        from camera_open import open_pipeline
        self.rs = rs
        self.accel = None
        for imu in (True, False):
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            if imu:
                config.enable_stream(rs.stream.accel)
            try:
                self.pipeline, profile = open_pipeline(rs, config)
                self.imu = imu
                break
            except Exception as exc:
                if not imu:
                    raise
                print(f"No accelerometer stream ({exc}); using --pitch for the camera tilt", flush=True)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = [intr.fx, intr.fy, intr.ppx, intr.ppy]
        self.dist = np.asarray(intr.coeffs, float)
        self.size = (intr.width, intr.height)
        self.samples = []
        print(f"USB: {profile.get_device().get_info(rs.camera_info.usb_type_descriptor)}, "
              f"colour {intr.width}x{intr.height}, fx {intr.fx:.0f}", flush=True)

    def read(self):
        from camera_open import latest_frames
        frames, _ = latest_frames(self.pipeline)
        if self.imu:
            for f in frames:
                if f.is_motion_frame():
                    a = f.as_motion_frame().get_motion_data()
                    self.samples = (self.samples + [(a.x, a.y, a.z)])[-60:]
        colour_frame = frames.get_color_frame()
        return None if not colour_frame else np.asanyarray(colour_frame.get_data()).copy()

    def gravity(self):
        """Up in camera axes from the accelerometer (it reads the reaction to gravity)."""
        if len(self.samples) < 10:
            return None
        a = np.mean(self.samples, 0)
        if not 8 < np.linalg.norm(a) < 11.5:
            return None  # moving, or a bogus reading
        up = a / np.linalg.norm(a)
        return -up if up[1] > 0 else up  # image y points down; the camera is not upside down

    def close(self):
        self.pipeline.stop()


class Detector:
    def __init__(self, args, dist):
        from person_track import PersonTrack
        from pose_yolo import load_model
        self.model, self.device = load_model(args.size, args.backend)
        self.track, self.args, self.dist = PersonTrack(), args, dist

    def __call__(self, image, intrinsics):
        res = self.model.predict(image, imgsz=self.args.imgsz, conf=.3, classes=0, verbose=False,
                                 **({"device": self.device} if self.device else {}))[0]
        if res.keypoints is None or not len(res.keypoints):
            return {}, None
        xy = res.keypoints.xy.cpu().numpy().reshape(-1, 17, 2)
        conf = (res.keypoints.conf.cpu().numpy().reshape(-1, 17) if res.keypoints.conf is not None
                else np.ones(xy.shape[:-1]))
        boxes = res.boxes.xyxy.cpu().numpy()
        pick = self.track.select(boxes, time.monotonic())
        if pick is None:
            return {}, (boxes, None)
        return wire_keypoints(xy[pick], conf[pick], intrinsics, self.dist), (boxes, pick)


class Synthetic:
    """An operator 3 m away lifting knees and arms, seen by a virtual camera."""

    def __init__(self, args):
        self.intrinsics = [910., 905., 640., 360.]
        self.size = (1280, 720)
        self.t0 = time.monotonic()
        self.solver = None

    def read(self):
        return np.full((self.size[1], self.size[0], 3), 40, np.uint8)

    def keypoints(self, camera, body):
        from test_body2d import directions
        t = time.monotonic() - self.t0
        lift = 45 * (1 - np.cos(t * 1.6)) * (np.sin(t * .4) > 0)
        right = 45 * (1 - np.cos(t * 1.6)) * (np.sin(t * .4) <= 0)
        arm = 60 * (1 - np.cos(t * .9))
        psi = np.radians(20 * np.sin(t * .3))
        d = directions(psi, (lift, lift * 1.2), arm)
        d[2], d[3] = directions(psi, (right, right * 1.2))[0:2]
        if self.solver is None or self.solver.camera is not camera:
            self.solver = BodySolver(camera, body)
        x = np.r_[0., 0., -3., psi, 0., 0., 0.]
        p = self.solver.points(x, d)
        x[1] = body.ankle_height - min(p[5, 1], p[6, 1])
        uv, _ = camera.project(self.solver.points(x, d))
        rng = np.random.default_rng(int(t * 30))
        uv = uv + rng.normal(0, 1.5, uv.shape)
        return {c: (uv[INDEX[n]][0], uv[INDEX[n]][1], .9) for n, c in COCO.items()}

    def close(self):
        pass


# ---- main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("size", nargs="?", choices=["n", "s", "m", "l", "x"], default="s", help="YOLO11-pose size")
    parser.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    parser.add_argument("--height", type=float, default=1.75, help="operator standing height, m (sets bone lengths)")
    parser.add_argument("--camera-height", type=float, default=1.2, help="lens above the floor, m")
    parser.add_argument("--pitch", type=float, default=0., help="camera tilt down, deg, when the IMU is unavailable")
    parser.add_argument("--no-imu", action="store_true", help="ignore the accelerometer; use --pitch")
    parser.add_argument("--session", help="camera->STAGE lock (session.json from pose_yolo.py) instead of height/tilt")
    parser.add_argument("--backend", choices=["auto", "torch", "cpu", "openvino"], default="torch")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--panel", type=int, default=640, help="width of each panel, px")
    parser.add_argument("--no-smooth", action="store_true", help="no One-Euro on the solved body")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--save", help="write the last composite frame to this image file")
    args = parser.parse_args()
    if sys.platform == "darwin" and args.source == "realsense" and hasattr(os, "geteuid") and os.geteuid() != 0:
        print("macOS opens the RealSense only under sudo: sudo <python> gmr_view.py ...", flush=True)

    from gmr_live import Retargeter, RobotView
    body = Body(args.height)
    source = Synthetic(args) if args.source == "synthetic" else RealSense(args)
    detector = None if args.source == "synthetic" else Detector(args, source.dist)
    placement = Placement(args, source.intrinsics)
    retargeter = Retargeter(body)
    width = args.panel
    height = int(round(width * source.size[1] / source.size[0]))
    view = RobotView(retargeter.model, width, height)
    camera = placement.camera()
    solver = BodySolver(camera, body)
    filters = {}
    timing, frames, fps, t_fps = {}, 0, 0., time.monotonic()
    started = time.monotonic()
    composite = None
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while not args.seconds or time.monotonic() - started < args.seconds:
            t = time.perf_counter()
            image = source.read()
            if image is None:
                continue
            if isinstance(source, RealSense) and not args.no_imu and placement.session is None:
                up = source.gravity()
                if up is not None and (placement.up is None or np.degrees(np.arccos(np.clip(up @ placement.up, -1, 1))) > .5):
                    placement.up = up
            new_camera = placement.camera()
            if not (np.allclose(new_camera.R, camera.R, atol=1e-4) and np.allclose(new_camera.t, camera.t)):
                camera = new_camera
                solver = BodySolver(camera, body)
            timing["camera"] = time.perf_counter() - t
            t = time.perf_counter()
            boxes = None
            if detector is None:
                keypoints = source.keypoints(camera, body)
            else:
                keypoints, boxes = detector(image, source.intrinsics)
            timing["yolo"] = time.perf_counter() - t
            t = time.perf_counter()
            now = time.monotonic()
            pose = solver.solve(keypoints, now) if keypoints else None
            timing["body"] = time.perf_counter() - t
            t = time.perf_counter()
            qpos = None
            if pose is not None:
                if not args.no_smooth:
                    for n, p in pose.points.items():
                        pose.points[n] = filters.setdefault(n, OneEuro(2., .8))(p, now)
                qpos = retargeter.solve(pose)
            timing["gmr"] = time.perf_counter() - t
            t = time.perf_counter()
            robot = view.render(qpos if qpos is not None else retargeter.gmr.configuration.q,
                                retargeter.camera_view(camera))[:, :, ::-1].copy()
            timing["render"] = time.perf_counter() - t

            left = image
            if boxes is not None:
                for i, b in enumerate(boxes[0]):
                    cv2.rectangle(left, tuple(np.int32(b[:2])), tuple(np.int32(b[2:])),
                                  (60, 220, 60) if i == boxes[1] else (60, 60, 220), 2)
            draw_overlay(left, keypoints, pose, camera, solver.o.min_conf)
            left = cv2.resize(left, (width, height))
            frames += 1
            if now - t_fps >= 1:
                fps, frames, t_fps = frames / (now - t_fps), 0, now
            status = ("no person" if not keypoints else "need both hips + 6 keypoints" if pose is None else
                      f"reproj {pose.reprojection_px:.1f} px")
            ambiguous = [LIMB_NAMES[k] for k, g in enumerate(pose.flip_gap) if 0 <= g < 1] if pose else []
            text(left, [f"{fps:4.1f} fps | {status}",
                        " ".join(f"{k} {1000 * v:.0f}" for k, v in timing.items()) + " ms",
                        placement.describe()], scale=.5)
            if ambiguous:
                text(left, ["front/back unsure: " + ", ".join(ambiguous)], (12, height - 14), .5, (60, 200, 255))
            text(robot, ["G1 after GMR, camera view", f"operator {args.height:.2f} m"], scale=.5)
            composite = np.hstack([left, robot])
            if args.headless:
                continue
            cv2.imshow(WINDOW, composite)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                solver.reset()
                retargeter.reset()
                filters.clear()
            elif key in (ord("["), ord("]")):
                placement.height += .02 if key == ord("]") else -.02
            elif key in (ord(";"), ord("'")) and placement.up is None:
                placement.pitch += 1 if key == ord("'") else -1
    finally:
        if args.save and composite is not None:
            cv2.imwrite(args.save, composite)
        source.close()
        view.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
