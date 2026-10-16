# Camera Legs From 2D Keypoints

[Home](Home.md) | [Camera stream](../prototypes/quest_g1/android/CAMERA_STREAM.md) | [DexWeave](https://github.com/emb-ai/FURRY/pull/20)

Status: **proposed** method with **locally reproduced** synthetic checks
(2026-10-14). No headset or operator measurement backs it yet.

## Question

Quest 3 with hand controllers observes no leg joints. TWIST2 adds two ankle
trackers to its PICO input; FURRY uses one external RealSense D435i instead.
D435i depth on legs is noisy (4–5 cm, mostly along the camera axis), and
lifting 2D keypoints with it gave poor skeletons. Can the robot's legs follow the
operator's legs from **2D keypoints only**, with the camera's depth axis
regularised rather than measured?

## Method

Inputs per camera frame:

| Quantity | Source | Status |
| --- | --- | --- |
| Raw 2D keypoints `(u, v, conf)` | YOLO11-pose on the colour image, before One-Euro, undistorted | Implemented: `skeleton/keypoints2d.py` |
| Intrinsics `[fx, fy, cx, cy]` | RealSense colour profile | Implemented: packet field `intr` |
| Camera to STAGE, 6D, fixed per camera place | AprilTag + controller touch (`skeleton/marker_frame.py`), or the existing shoulder fit when depth is on | Implemented: packet field `stage_from_camera` |
| Hip position and bone lengths | Meta body tracking, driven by headset and controllers | Existing |

The transform that must be known is **camera to STAGE**, not camera to body.
Camera to body follows per frame from Meta's pelvis. Leaving the body pose free
per frame would make a 2D-only fit underdetermined.

Per leg, the headset (`android/native/camera_legs2d.h`) solves thigh and shin
directions from Meta's hip with Meta's bone lengths, minimising:

- the confidence-weighted **reprojection error** of knee and ankle (6 px sigma);
- a **depth-axis regulariser**: a weak pull of knee and ankle depth (camera z)
  toward Meta's time-matched pose (1 m sigma, a tie-breaker only) and toward
  the previous solution (8 cm sigma);
- soft **anatomical limits**: no backward knee bend, knee varus/valgus within
  5°, hip extension within 15°, thigh within 30° of the sagittal plane.

Every bone has two ray–sphere roots for any camera placement. The solver starts
from both roots of each bone, Meta's pose and the previous solution, and keeps
the lowest cost. The rejected branch's cost margin is reported (`flipGap`).
The result enters the existing camera fusion as thigh and shin directions, so
smoothing, fading, the person gate, recording, GMR retargeting and the TWIST2
policy are unchanged.

GMR is not given the pixel residual directly. Robot and human leg proportions
differ, so fitting robot links to human camera rays would bend the joint angles.
The 2D residual is solved against human bone lengths and GMR tracks the result.

## Synthetic Results

`tests/check_camera_legs2d.cpp`, camera 3 m away, 1.2 m high, operator
facing it (frontal) or rotated 40° (oblique). Locally reproduced on Linux.

| Case | Frontal | Oblique 40° |
| --- | --- | --- |
| Exact pixels, standing and knee lifts to 90° | Recovered (< 2 cm) | Recovered (< 2 cm) |
| Exact pixels, backward swing (hip −15°, knee 30°) | Wrong branch | Recovered |
| 5 px noise, random poses, mean worst-joint error | 13.5 cm | 5.9 cm |

A frontal camera sees a backward leg swing purely along its depth axis, so no
2D method can recover it from that view alone. Placing the camera 30–45° off
frontal is the single largest improvement. AprilTag registration allows that.
The shoulder fit does not: it accepts only frames facing the camera.

## Evaluating Without Ground Truth

There is no motion capture. Recorded sessions (`camera.jsonl` on the headset
with `kp2d`, and the PC `--log` with downlink and HMD/controller uplink) make
every method run on the same frames:

1. **Controller probes** (cheap ground truth): touch a knee or ankle with a
   controller and hold it; touch floor marks, then step on them; stand straight.
2. **Held-out evidence**: predict a keypoint the solve did not use; reproject
   onto the RGB video.
3. **Physical plausibility**: foot sliding in stance, floor penetration, jitter,
   knee flips (`flipGap` near zero marks frames decided by the regulariser).
4. **Downstream**: replay through GMR and TWIST2 (`replay_gmr`) against Meta's
   generated legs as the baseline.

Agreement between methods shows ambiguity, not truth; rank on the probes.

## Alternatives Considered

| Option | Verdict |
| --- | --- |
| D435i depth per leg joint | Existing path; kept for A/B (`use2d=false`, or server `--no-kp2d`). Too noisy along the axis that matters. |
| Heavy monocular 3D (RTMW3D-X, HybrIK, CLIFF, HMR2) | Not lightweight; RTMW3D depth is model-relative ([experiments](../skeleton/EXPERIMENTS.md)). |
| Temporal 2D→3D lifter (VideoPose3D, MotionAGFormer-XS class) | Proposed follow-up: only the sign of each bone's depth is needed, as an extra regulariser. |
| Second RGB camera, triangulation | Rejected for setup weight. |
| Pixel residual inside GMR's robot IK | Rejected: robot proportions differ from the human's. |

From [DexWeave](https://github.com/emb-ai/FURRY/pull/20): infer stance from foot
motion and pin the stance foot to the floor. That constrains exactly the depth
axis 2D leaves open. Proposed follow-up, not implemented.

## Running It

On the PC with the RealSense (`skeleton/`):

```sh
python pose_yolo.py s --ws --no-depth --session session.json --log logs/run.jsonl
```

Press `c` with the right controller touching the printed AprilTag to lock the
camera in STAGE. Without `--no-depth`, packets carry both 3D and 2D, and the
shoulder fit can register the camera too. On the headset the leg row reads
`Ноги: камера 2D` when 2D keypoints drive the legs, and `нужна связка
AprilTag` when RGB-only packets arrive without a registration.

## Desktop Viewer Without The Headset

`skeleton/gmr_view.py` shows the same idea on a Mac or PC alone: the
RealSense colour frame with the YOLO keypoints and the solved body on the
left, and the G1 after GMR on the right, rendered from the same camera
viewpoint so both panels show one pose from one place.

Without Meta there is no hip anchor, so `skeleton/body2d.py` solves the whole
body from the 2D keypoints:

- Bone lengths come from `--height` (anthropometric ratios).
- The pelvis, heading and torso are solved, and the floor fixes the depth axis:
  the lower ankle stands on the floor, located by the D435i accelerometer
  (gravity) and `--camera-height`, or by an AprilTag `--session`.
- Limbs use the same ray-sphere branches and anatomical limits as the headset,
  plus no backward elbow bend. The branch is chosen on data and anatomy only,
  never on the temporal prior: tracked limbs must not lock onto the wrong
  branch. Limbs decided by the regulariser alone show as "front/back unsure".

`skeleton/gmr_live.py` feeds the pinned upstream GMR/Mink solver (the one the
native port is checked against) with targets aligned to the robot's T pose, as
`meta_retarget.cpp` does on the headset.

```sh
pip install -r skeleton/requirements-gmr-view.txt
sudo python skeleton/gmr_view.py s --camera-height 1.20 --height 1.78   # macOS needs sudo for the camera
python skeleton/gmr_view.py --source synthetic                          # no camera
```

Synthetic checks (`skeleton/test_body2d.py`, `skeleton/test_gmr_live.py`,
locally reproduced): exact keypoints recover standing, tracked knee lifts to
90° and back on frontal and 40° oblique cameras (median leg error under 3 cm),
and GMR maps them to the expected G1 hip and knee angles (knee lift 60°/70° to
hip pitch −60°, knee 70°, within 4°). With 3 px keypoint noise the frontal
leg error is bounded at a 12 cm median. Near standing, the two ray-sphere
roots of a frontal leg are close and both anatomical, so small lifts can lag.

## Limitations

- Synthetic checks only; no live operator session yet.
- Hip anchor is Meta's: a pelvis error shifts both legs.
- COCO hips, knees and ankles are visual landmarks, not joint centres.
- Foot pitch and yaw stay Meta's; the camera has no toe or heel keypoints.
