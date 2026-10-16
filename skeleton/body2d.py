"""Whole-body 3D pose from one camera's 2D keypoints, without depth or a headset.

Standalone counterpart of the headset's ray-based leg solve
(prototypes/quest_g1/android/native/camera_legs2d.h) for the desktop GMR
viewer (gmr_view.py). There is no Meta hip anchor here, so the pelvis is
solved too:

- Bone lengths are fixed from the operator's height (anthropometric ratios).
- Hips, shoulders, elbows, wrists, knees, ankles, nose and ears must reproject
  onto the detector's pixels (confidence-weighted, robust loss).
- The camera's depth axis is regularised, never measured: the lower ankle
  stands on the floor (the floor plane is known from the camera calibration),
  the torso's camera depth is pulled toward the previous frame, and each bone
  weakly toward a neutral standing pose as a tie-breaker.
- Soft anatomical limits as on the headset: no backward knee bend, knee varus
  within 5 deg, hip extension within 15 deg, thigh within 30 deg of the
  sagittal plane; the upper arm within 5 deg behind the torso and no
  backward elbow bend.

The solve alternates between the root (pelvis, heading, torso; limbs held)
and each limb (two bone directions; root held). Each bone has two ray-sphere
roots, so every limb starts from every root combination, the neutral pose and
the previous frame. The limb's forward/backward branch is chosen on the data
and anatomy alone, never on the temporal prior: a tracked limb must not stay
locked on the wrong branch because leaving it costs a depth jump. A small
hysteresis keeps the previous branch on near ties; the margin is flip_gap.

World frame: metres, Y up, floor at y = 0 (STAGE convention). Camera: OpenCV
axes, p_world = R p_camera + t.
"""
from dataclasses import dataclass, field

import numpy as np

UP = np.array([0., 1., 0.])

# Point order of the model. COCO indices observed for each point, if any.
NAMES = ["pelvis", "lhip", "rhip", "lknee", "rknee", "lankle", "rankle", "chest",
         "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "nose", "lear", "rear"]
INDEX = {n: i for i, n in enumerate(NAMES)}
COCO = {"nose": 0, "lear": 3, "rear": 4, "lshoulder": 5, "rshoulder": 6, "lelbow": 7, "relbow": 8,
        "lwrist": 9, "rwrist": 10, "lhip": 11, "rhip": 12, "lknee": 13, "rknee": 14, "lankle": 15, "rankle": 16}
OBSERVED = [INDEX[n] for n in COCO]
OBSERVED_COCO = [COCO[n] for n in COCO]
# Face points only fix the heading: weaker than the limbs.
POINT_WEIGHT = np.array([.5 if n in ("nose", "lear", "rear") else 1. for n in COCO])
# Limbs: (proximal point, middle point, distal point, bone keys).
LIMBS = [("lhip", "lknee", "lankle", "thigh", "shin"), ("rhip", "rknee", "rankle", "thigh", "shin"),
         ("lshoulder", "lelbow", "lwrist", "upper_arm", "forearm"),
         ("rshoulder", "relbow", "rwrist", "upper_arm", "forearm")]
NEUTRAL = np.array([[0, -1, 0]] * 8, float)  # standing, arms hanging
LIMB_POINTS = [[INDEX[m], INDEX[d]] for _, m, d, _, _ in LIMBS]
LIMB_OBSERVED = [[OBSERVED.index(i) for i in pts] for pts in LIMB_POINTS]
ROOT_POINTS = [INDEX[n] for n in ("pelvis", "lhip", "rhip", "chest", "lshoulder", "rshoulder")]
ROBUST_SCALE = 3.    # residuals beyond 3 sigma (outlier keypoints) grow linearly


@dataclass
class Body:
    """Segment lengths, m. Ratios of standing height (Drillis & Contini; Winter)."""
    height: float = 1.75

    def __post_init__(self):
        h = self.height
        self.thigh, self.shin = .245 * h, .246 * h
        self.upper_arm, self.forearm = .186 * h, .146 * h
        self.hip_half = .055 * h          # COCO hip keypoints, wider than joint centres
        self.shoulder_half = .115 * h     # COCO shoulder keypoints
        self.torso = .288 * h             # hip line to shoulder line
        self.pelvis_height = .530 * h
        self.ankle_height = .039 * h
        self.nose_up, self.nose_forward = .11 * h, .05 * h
        self.ear_up, self.ear_half = .10 * h, .04 * h

    def lengths(self):
        return np.array([self.thigh, self.shin] * 2 + [self.upper_arm, self.forearm] * 2)


@dataclass
class Camera:
    """Pinhole camera placed in the world: p_world = R p_camera + t."""
    R: np.ndarray
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float

    def to_camera(self, p):
        return (np.asarray(p, float) - self.t) @ self.R

    def project(self, p):
        """Pixels (N, 2) and camera depths (N,) of world points (N, 3)."""
        c = self.to_camera(p)
        z = c[..., 2]
        safe = np.maximum(z, 1e-3)
        return np.stack([self.fx * c[..., 0] / safe + self.cx, self.fy * c[..., 1] / safe + self.cy], -1), z

    def ray(self, u, v):
        d = self.R @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.])
        return d / np.linalg.norm(d)

    @property
    def forward(self):
        return self.R[:, 2]


@dataclass
class Options:
    pixel_sigma: float = 6.          # detector noise at confidence 1, px
    min_conf: float = .3             # keypoints below this are not used
    floor_sigma: float = .03         # lower ankle on the floor, m
    previous_depth_sigma: float = .08  # temporal pull along the camera axis, m
    neutral_sigma: float = 2.        # tie-breaker toward the standing pose (unit-vector distance)
    lean_sigma_deg: float = 20.
    twist_sigma_deg: float = 15.
    previous_turn_sigma_deg: float = 10.  # heading and torso twist per frame
    hinge_sigma_deg: float = 4.
    max_hip_extension_deg: float = 15.
    hip_extension_sigma_deg: float = 3.
    max_varus_deg: float = 5.
    varus_sigma_deg: float = 3.
    max_thigh_out_deg: float = 30.
    thigh_out_sigma_deg: float = 5.
    max_arm_extension_deg: float = 5.
    arm_extension_sigma_deg: float = 5.
    max_elbow_back_deg: float = 10.
    elbow_sigma_deg: float = 4.
    previous_max_age_s: float = .3
    reacquire_px: float = 20.        # start fresh next frame when the fit is this far off
    branch_hysteresis: float = .2    # keep the previous limb branch unless the other wins by this
    rounds: int = 2                  # root/limb alternations on a fresh start (1 while tracking)


@dataclass
class Pose:
    points: dict
    pelvis_forward: np.ndarray
    pelvis_left: np.ndarray
    torso_up: np.ndarray
    torso_forward: np.ndarray
    torso_left: np.ndarray
    reprojection_px: float          # median over used keypoints
    flip_gap: list = field(default_factory=list)  # per limb; -1 when unknown
    cost: float = 0.


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _tangent(d):
    """Two unit vectors orthogonal to unit d (..., 3)."""
    helper = np.where(np.abs(d[..., :1]) < .9, [1., 0, 0], [0., 1, 0])
    a = _unit(np.cross(d, helper))
    return a, np.cross(d, a)


def _robust(r, c=ROBUST_SCALE):
    """Residual whose square is the soft-L1 loss of r (outlier keypoints)."""
    return np.sign(r) * c * np.sqrt(2 * (np.sqrt(1 + (r / c) ** 2) - 1))


def _frames(X):
    """Pelvis forward/left and torso up/forward/left for root parameters (..., 7)."""
    psi, lean_f, lean_l, twist = X[..., 3], X[..., 4], X[..., 5], X[..., 6]
    zero = np.zeros_like(psi)
    forward = np.stack([np.sin(psi), zero, np.cos(psi)], -1)
    left = np.stack([np.cos(psi), zero, -np.sin(psi)], -1)  # up x forward
    torso_up = _unit(UP + lean_f[..., None] * forward + lean_l[..., None] * left)
    f0 = np.stack([np.sin(psi + twist), zero, np.cos(psi + twist)], -1)
    torso_forward = _unit(f0 - _dot(f0, torso_up)[..., None] * torso_up)
    return forward, left, torso_up, torso_forward, np.cross(torso_up, torso_forward)


def _deg(y, x):
    return np.degrees(np.arctan2(y, x))


def _lm(f, X, iterations=10, h=1e-6):
    """Batched Levenberg-Marquardt: f maps (B, n) to residuals (B, m).

    Every row is an independent start; rows never mix. Returns X, residuals.
    """
    B, n = X.shape
    r = f(X)
    cost = np.sum(r * r, -1)
    lam = np.full(B, 1e-3)
    eye = np.eye(n)
    for _ in range(iterations):
        before = cost.copy()
        Xp = (X[:, None, :] + h * eye[None]).reshape(B * n, n)
        J = (f(Xp).reshape(B, n, -1) - r[:, None, :]) / h      # (B, n, m): J transposed
        A = J @ np.swapaxes(J, 1, 2)
        g = np.einsum("bnm,bm->bn", J, r)
        diag = np.diagonal(A, axis1=1, axis2=2)
        improved = np.zeros(B, bool)
        for _ in range(3):
            M = A + (lam[:, None] * (1 + diag))[:, :, None] * eye[None]
            try:
                step = np.linalg.solve(M, -g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = np.zeros_like(X)
            Xn = X + step
            rn = f(Xn)
            cn = np.sum(rn * rn, -1)
            accept = (cn < cost) & ~improved
            X = np.where(accept[:, None], Xn, X)
            r = np.where(accept[:, None], rn, r)
            cost = np.where(accept, cn, cost)
            improved |= accept
            lam = np.where(accept, np.maximum(lam / 10, 1e-7), np.where(improved, lam, lam * 10))
            if improved.all():
                break
        if np.all(before - cost <= 1e-4 * (1 + before)):
            break  # every start has converged (or stalled)
    return X, r


class BodySolver:
    def __init__(self, camera, body=None, options=None):
        self.camera, self.body, self.o = camera, body or Body(), options or Options()
        self.lengths = self.body.lengths()
        self.reset()

    def reset(self):
        self.previous = None  # dict: x, directions, depth (per point), branch (per limb), time

    # ---- model --------------------------------------------------------------
    def points(self, X, directions):
        """Model points (..., 17, 3) for root parameters (..., 7) and bones (..., 8, 3)."""
        b = self.body
        X = np.asarray(X, float)
        _, left, up, tf, tl = _frames(X)
        pelvis = X[..., :3]
        chest = pelvis + up * b.torso
        head = chest + up * b.ear_up
        p = [None] * len(NAMES)
        p[0], p[7] = pelvis, chest
        p[1], p[2] = pelvis + left * b.hip_half, pelvis - left * b.hip_half
        p[8], p[9] = chest + tl * b.shoulder_half, chest - tl * b.shoulder_half
        p[14] = chest + up * b.nose_up + tf * b.nose_forward
        p[15], p[16] = head + tl * b.ear_half, head - tl * b.ear_half
        for k, (a, m, d, _, _) in enumerate(LIMBS):
            p[INDEX[m]] = p[INDEX[a]] + directions[..., 2 * k, :] * self.lengths[2 * k]
            p[INDEX[d]] = p[INDEX[m]] + directions[..., 2 * k + 1, :] * self.lengths[2 * k + 1]
        return np.stack(p, -2)

    def _pixels(self, P, observed):
        uv, z = self.camera.project(P)
        r = observed[:, 2:3] * (uv - observed[:, :2]) / self.o.pixel_sigma
        return _robust(r.reshape(*r.shape[:-2], -1)), np.where(z < .1, 1e2, 0.)

    def _floor(self, left_y, right_y):
        b, o = self.body, self.o
        return np.stack([(np.minimum(left_y, right_y) - b.ankle_height) / o.floor_sigma,
                         np.maximum(0., b.ankle_height - .01 - left_y) / .01,
                         np.maximum(0., b.ankle_height - .01 - right_y) / .01], -1)

    def _limits(self, k, frames, d1, d2):
        o = self.o
        forward, left, tu, tf, _ = frames
        if k < 2:
            axis = _unit(left - _dot(d1, left)[..., None] * d1)
            back = np.cross(axis, d1)
            flex = _deg(_dot(d2, back), _dot(d2, d1))
            ext = _deg(-_dot(d1, forward), -d1[..., 1])
            varus = np.degrees(np.arcsin(np.clip(_dot(d2, axis), -1, 1)))
            out = np.degrees(np.arcsin(np.clip(_dot(d1, left), -1, 1)))
            return np.stack([np.minimum(0., flex) / o.hinge_sigma_deg,
                             np.maximum(0., ext - o.max_hip_extension_deg) / o.hip_extension_sigma_deg,
                             np.maximum(0., np.abs(varus) - o.max_varus_deg) / o.varus_sigma_deg,
                             np.maximum(0., np.abs(out) - o.max_thigh_out_deg) / o.thigh_out_sigma_deg], -1)
        ext = _deg(-_dot(d1, tf), -_dot(d1, tu))
        # Elbow: flexes toward the front of the upper arm (forward when it
        # hangs, up when it points forward), never backward. Vanishes when
        # the upper arm points sideways along the hinge axis.
        _, tl = frames[3], frames[4]
        axis = tl - _dot(d1, tl)[..., None] * d1
        weight = np.linalg.norm(axis, axis=-1)
        back = np.cross(_unit(axis), d1)
        bend = _deg(_dot(d2, back), _dot(d2, d1))
        return np.stack([np.maximum(0., ext - o.max_arm_extension_deg) / o.arm_extension_sigma_deg,
                         weight * np.maximum(0., bend - o.max_elbow_back_deg) / o.elbow_sigma_deg], -1)

    # ---- root: pelvis, heading, torso with the limbs held --------------------
    def _root_residuals(self, X, directions, observed):
        o = self.o
        P = self.points(X, directions)
        pixels, behind = self._pixels(P[..., OBSERVED, :], observed)
        out = [pixels, behind, self._floor(P[..., 5, 1], P[..., 6, 1]),
               np.degrees(X[..., 4:6]) / o.lean_sigma_deg, np.degrees(X[..., 6:7]) / o.twist_sigma_deg]
        if self.previous is not None:
            z = self.camera.to_camera(P[..., ROOT_POINTS, :])[..., 2]
            out.append((z - self.previous["depth"][ROOT_POINTS]) / o.previous_depth_sigma)
        return np.concatenate(out, -1)

    def _fit_root(self, X, directions, observed):
        X, r = _lm(lambda v: self._root_residuals(v, directions, observed), np.atleast_2d(X), 12)
        return X, np.sum(r * r, -1)

    # ---- joint polish: root and every bone together, within the chosen branches
    def _joint_residuals(self, V, base, observed):
        """V (B, 7 + 16): root parameters and tangent steps from base (8, 3, 3)."""
        o = self.o
        X, T = V[:, :7], V[:, 7:].reshape(-1, 8, 2)
        D = _unit(base[:, 0] + T[..., :1] * base[:, 1] + T[..., 1:] * base[:, 2])
        P = self.points(X, D)
        pixels, behind = self._pixels(P[:, OBSERVED], observed)
        frames = _frames(X)
        out = [pixels, behind, self._floor(P[:, 5, 1], P[:, 6, 1]),
               np.degrees(X[:, 4:6]) / o.lean_sigma_deg, np.degrees(X[:, 6:7]) / o.twist_sigma_deg,
               (D - NEUTRAL).reshape(len(V), -1) / o.neutral_sigma]
        for k in range(4):
            out.append(self._limits(k, frames, D[:, 2 * k], D[:, 2 * k + 1]))
        if self.previous is not None:
            # Root only: on the limbs the pull would drag a bone back across
            # to the previous frame's branch.
            z = self.camera.to_camera(P[:, ROOT_POINTS])[..., 2]
            out.append((z - self.previous["depth"][ROOT_POINTS]) / o.previous_depth_sigma)
            turn = np.degrees(X[:, 3:7:3] - self.previous["x"][3:7:3])
            out.append((turn + 180) % 360 - 180)
            out[-1] = out[-1] / o.previous_turn_sigma_deg
        return np.concatenate(out, -1)

    def _fit_joint(self, x, directions, observed):
        a, b = _tangent(directions)
        base = np.stack([directions, a, b], 1)
        V, r = _lm(lambda v: self._joint_residuals(v, base, observed), np.r_[x, np.zeros(16)][None], 10)
        T = V[0, 7:].reshape(8, 2)
        return V[0, :7], _unit(directions + T[:, :1] * a + T[:, 1:] * b), float(np.sum(r * r))

    # ---- limbs: two bone directions with the root held -----------------------
    def _limb_residuals(self, T, k, base, start, frames, other_y, observed):
        """T (B, 4) tangent steps from base (B, 6, 3): start dirs and tangents."""
        d1 = _unit(base[:, 0] + T[:, :1] * base[:, 2] + T[:, 1:2] * base[:, 3])
        d2 = _unit(base[:, 1] + T[:, 2:3] * base[:, 4] + T[:, 3:4] * base[:, 5])
        mid = start + d1 * self.lengths[2 * k]
        end = mid + d2 * self.lengths[2 * k + 1]
        pixels, behind = self._pixels(np.stack([mid, end], 1), observed[LIMB_OBSERVED[k]])
        out = [pixels, behind,
               np.concatenate([d1 - NEUTRAL[2 * k], d2 - NEUTRAL[2 * k + 1]], -1) / self.o.neutral_sigma,
               self._limits(k, frames, d1, d2)]
        if k == 0:
            out.append(self._floor(end[:, 1], np.full(len(T), other_y)))
        elif k == 1:
            out.append(self._floor(np.full(len(T), other_y), end[:, 1]))
        return np.concatenate(out, -1), d1, d2

    def _fit_limb(self, k, x, directions, observed, kp):
        """Directions of limb k with the root held, its branch and the flip gap.

        Fits use data and anatomy only. The temporal prior never decides the
        branch: leaving a wrong branch would cost a depth jump, so it would
        lock in. It only keeps the previous branch on near ties (hysteresis).
        """
        p = self.points(x, directions)
        frames = _frames(x)
        a, m, d, _, _ = LIMBS[k]
        start = p[INDEX[a]]
        other = p[6, 1] if k == 0 else p[5, 1]
        starts = [directions[2 * k:2 * k + 2], NEUTRAL[2 * k:2 * k + 2]]
        if self.previous is not None:
            starts.append(self.previous["directions"][2 * k:2 * k + 2])
        cm, cd = kp[COCO[m]], kp[COCO[d]]
        ray = None
        if cm[2] >= self.o.min_conf:
            ray = self.camera.ray(cm[0], cm[1])
            mids = _ray_sphere(self.camera.t, ray, start, self.lengths[2 * k])
            for mid in mids:
                ends = ([mid + NEUTRAL[2 * k + 1] * self.lengths[2 * k + 1]] if cd[2] < self.o.min_conf else
                        _ray_sphere(self.camera.t, self.camera.ray(cd[0], cd[1]), mid, self.lengths[2 * k + 1]))
                for end in ends:
                    starts.append(np.stack([_unit(mid - start), _unit(end - mid)]))
        S = np.stack(starts)
        a1, b1 = _tangent(S[:, 0])
        a2, b2 = _tangent(S[:, 1])
        base = np.stack([S[:, 0], S[:, 1], a1, b1, a2, b2], 1)
        args = (k, base, start, frames, other, observed)

        def f(T):
            rows = len(T) // len(base)
            return self._limb_residuals(T, k, np.repeat(base, rows, 0), start, frames, other, observed)[0]
        T, r = _lm(f, np.zeros((len(base), 4)), 8)
        _, d1, d2 = self._limb_residuals(T, *args)
        cost = np.sum(r * r, -1)
        if ray is None:
            branch = np.zeros(len(T), int)
        else:
            split = (start - self.camera.t) @ ray
            branch = (((start + d1 * self.lengths[2 * k]) - self.camera.t) @ ray < split).astype(int)
        best = {}
        for i in np.argsort(cost):
            best.setdefault(int(branch[i]), i)
        chosen = min(best, key=lambda b: cost[best[b]])
        if self.previous is not None and ray is not None:
            kept = self.previous["branch"][k]
            if kept in best and cost[best[kept]] <= cost[best[chosen]] + self.o.branch_hysteresis:
                chosen = kept
        gap = float(cost[best[1 - chosen]] - cost[best[chosen]]) if len(best) == 2 else -1.
        i = best[chosen]
        return np.stack([d1[i], d2[i]]), chosen, gap

    # ---- initialisation -----------------------------------------------------
    def _root_guess(self, kp):
        """Pelvis and heading from the hips, ankles on the floor and the face."""
        b, cam = self.body, self.camera
        mid = (kp[11][:2] + kp[12][:2]) / 2
        depth = None
        feet = []
        for i in (15, 16):
            if kp[i][2] >= self.o.min_conf:
                d = cam.ray(kp[i][0], kp[i][1])
                if d[1] < -1e-3:
                    feet.append(cam.t + (b.ankle_height - cam.t[1]) / d[1] * d)
        if feet:
            depth = float(np.mean([cam.to_camera(f)[2] for f in feet]))
        elif min(kp[5][2], kp[6][2]) >= self.o.min_conf:
            torso_px = np.linalg.norm((kp[5][:2] + kp[6][:2]) / 2 - mid)
            if torso_px > 5:
                depth = cam.fy * b.torso / torso_px
        if depth is None or not np.isfinite(depth) or depth < .3:
            depth = 3.
        ray_camera = np.array([(mid[0] - cam.cx) / cam.fx, (mid[1] - cam.cy) / cam.fy, 1.])
        pelvis = cam.t + cam.R @ (ray_camera * depth)
        to_camera = cam.t - pelvis
        to_camera[1] = 0
        to_camera = _unit(to_camera)
        # Facing the camera when the nose is seen.
        f = to_camera if kp[0][2] >= self.o.min_conf else -to_camera
        return np.r_[pelvis, np.arctan2(f[0], f[2]), 0., 0., 0.]

    # ---- solve --------------------------------------------------------------
    def solve(self, keypoints, now):
        """keypoints: {coco index: (u, v, conf)} undistorted pixels. Pose or None."""
        kp = np.zeros((17, 3))
        for i, value in keypoints.items():
            if 0 <= int(i) < 17:
                kp[int(i)] = value
        if self.previous is not None and now - self.previous["time"] > self.o.previous_max_age_s:
            self.reset()
        observed = kp[OBSERVED_COCO].copy()
        used = observed[:, 2] >= self.o.min_conf
        observed[:, 2] = np.where(used, np.sqrt(np.clip(observed[:, 2], 1e-3, 1)) * POINT_WEIGHT, 0.)
        if not (kp[11][2] >= self.o.min_conf and kp[12][2] >= self.o.min_conf) or used.sum() < 6:
            return None
        if self.previous is not None:
            starts = [(self.previous["x"].copy(), self.previous["directions"].copy())]
        else:
            guess = self._root_guess(kp)
            mirrored = guess.copy()
            mirrored[3] += np.pi  # front/back when the face is not decisive
            starts = [(guess, NEUTRAL.copy()), (mirrored, NEUTRAL.copy())]
        best = None
        for x, directions in starts:
            branches, gaps = [0] * 4, [-1.] * 4
            for _ in range(self.o.rounds if self.previous is None else 1):
                X, _ = self._fit_root(x, directions, observed)
                x = X[0]
                for k in range(4):
                    pair, branches[k], gaps[k] = self._fit_limb(k, x, directions, observed, kp)
                    directions[2 * k:2 * k + 2] = pair
            x, directions, cost = self._fit_joint(x, directions, observed)
            if best is None or cost < best[0]:
                best = (cost, x, directions, list(branches), list(gaps))
        cost, x, directions, branches, gaps = best
        p = self.points(x, directions)
        uv, _ = self.camera.project(p[OBSERVED])
        err = np.linalg.norm(uv - kp[OBSERVED_COCO, :2], axis=1)[used]
        median = float(np.median(err))
        if median > self.o.reacquire_px:
            self.reset()  # lost: start fresh next frame
        else:
            self.previous = dict(x=x.copy(), directions=directions.copy(), branch=branches,
                                 depth=self.camera.to_camera(p)[:, 2], time=now)
        forward, left, tu, tf, tl = _frames(x)
        return Pose({n: p[i] for i, n in enumerate(NAMES)}, forward, left, tu, tf, tl, median, gaps, cost)


def _ray_sphere(origin, direction, centre, radius):
    oc = origin - centre
    b = oc @ direction
    c = oc @ oc - radius * radius
    disc = b * b - c
    if disc < 0:
        return [origin + direction * max(0., -b)]
    s = np.sqrt(disc)
    out = [origin + direction * t for t in (-b - s, -b + s) if t > 0]
    return out or [origin + direction * max(0., -b)]


def camera_from_gravity(up_camera, height, fx, fy, cx, cy):
    """Camera at `height` m above the floor, levelled by gravity.

    up_camera: the world's up direction in camera axes (e.g. the D435i
    accelerometer at rest). World heading follows the camera: world -Z is
    the camera's forward direction projected on the floor.
    """
    up = _unit(np.asarray(up_camera, float))
    forward = np.array([0., 0., 1.]) - up[2] * up
    if np.linalg.norm(forward) < 1e-3:
        raise ValueError("Camera looks straight up or down")
    forward = _unit(forward)
    back = -forward
    right = np.cross(up, back)
    # Rows: world axes in camera coordinates; R maps camera to world.
    R = np.stack([right, up, back])
    return Camera(R, np.array([0., height, 0.]), fx, fy, cx, cy)


def camera_from_pitch(pitch_deg, height, fx, fy, cx, cy):
    """Camera with no roll, tilted down by pitch_deg, at height m."""
    a = np.radians(pitch_deg)
    # Up in camera axes (y down, z forward): a downward tilt sees up partly along -z.
    return camera_from_gravity([0., -np.cos(a), -np.sin(a)], height, fx, fy, cx, cy)
