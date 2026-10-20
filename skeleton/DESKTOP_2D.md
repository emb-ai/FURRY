# Experimental desktop 2D-to-GMR viewer

This optional viewer reconstructs a human pose from camera keypoints and displays
the retargeted G1 next to the RGB image. It does not send commands to Quest or
change its leg fusion, policy, recording or calibration defaults.

The source comes from `origin/claude/project-thread-ei5tbk` at
`1b26515` (full revision and the complete branch patch are recorded in the
[preservation manifest](../docs/research/project-preservation-20261021/runtime/manifest.json)).
The original headset integration remains an archived experiment.

## Setup and use

Prepare the pinned GMR assets used by the
[desktop GMR oracle](../prototypes/quest_g1/android/GMR_PORT.md).
The viewer expects `prototypes/quest_g1/vendor/GMR`, including its G1 model.
From the repository root, install the camera dependencies and then
`skeleton/requirements-gmr-view.txt` into the chosen Python environment.
On macOS, the RealSense Python binding comes from the separately built SDK;
see [camera setup](AGENT.md).

```sh
python -m pip install -r skeleton/requirements-gmr-view.txt
python skeleton/gmr_view.py --source synthetic
python skeleton/gmr_view.py s --camera-height 1.20 --height 1.78
```

Use the actual camera and operator heights. The RealSense mode needs its SDK,
YOLO weights and camera access. The synthetic mode needs OpenCV, MuJoCo, Mink
and the pinned GMR assets, but no camera or headset. An existing AprilTag
`--session` can supply camera placement; otherwise height and gravity/pitch
establish the floor. No new calibration is inferred from old recordings.

## Checks and limits

On 2026-10-21, six body reconstruction tests, four GMR retargeting tests and two
keypoint conversion tests passed on the local Mac. They check synthetic
standing, forward arms, tracked knee lifts, camera projection, noisy pixels
and missing hips. These are geometry and retargeting checks, not a new camera
recording or physical policy rollout.

```sh
cd skeleton
python -m unittest test_body2d test_gmr_live test_keypoints2d -v
```

One image cannot resolve every front/back limb ambiguity. The floor and assumed
body proportions constrain depth; a frontal view remains weak for backward
motion. COCO landmarks are not anatomical joint centres. The original branch
reported synthetic error growth with noisy pixels and has no accepted live
operator session. Do not treat the viewer's skeleton as measured ground truth.

## Camera diagnostics retained from local work

`pose_yolo.py --ws --debug-arms` optionally adds the four elbow/wrist landmarks
to the existing packet. Without the flag, the nine-joint contract is unchanged.
Timestamp, timing diagnostics and `session_id` are preserved in both modes.
These landmarks support alignment inspection; controller centres and wrist
landmarks are different physical points.

For a separate raw camera/Quest collection workflow, see
[inference logger](inference_logger/README.md). Its proposed Quest sync patch
is included for review and remains unapplied.
