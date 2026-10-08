# RGB-D replay and human leg fitting

These tools are an experimental comparison path. They do not change the robot
policy or send fitted poses to Quest. Keep recordings, measurements, model
weights and derived participant data outside Git.

## Capture with the existing server

Add `--record-rgbd recordings/<new-session> --record-rgbd-seconds 60` to
`pose_yolo.py --ws`. The normal stream continues. A new directory is required;
existing recordings are never overwritten. On macOS use the existing SDK/USB
launcher and stop any previous camera process first.

Stand still for the first ten seconds, then step, turn, squat and raise each leg.
Keep the whole body in the image. Avoid another person crossing the operator.

The bounded background writer saves at most ten processed frames per second:
unannotated BGR, unfiltered depth aligned to color (metres), intrinsics,
distortion, source epoch time and the same sequence number as the outgoing
pose. It is not a complete 30 Hz sensor recording. Queue drops and write errors
are reported in `manifest.json`. Disk/compression load may affect live timing;
use offline replay for comparing inference time. Empty capture is not evidence
of a successful camera recording even if shutdown was clean.

## RTMW3D comparison

Install optional `requirements-experiments.txt`. RTMLib's `Wholebody3d.MODE`
documents the YOLOX detector and RTMW3D-X ONNX downloads. Place the extracted
detector at `models/rtmw3d/detector.onnx` and pose model at
`models/rtmw3d/pose.onnx`. Model licenses and exports are separate from this code.

```
python rtmw_probe.py data/frame.npy --models models/rtmw3d \
  --provider CoreMLExecutionProvider --output recordings/probe.json
python compare_rgbd.py recordings/<session> yolo --output recordings/yolo.jsonl
python compare_rgbd.py recordings/<session> rtmw --output recordings/rtmw.jsonl
```

Run backends sequentially to avoid GPU contention. `rtmw_probe` also accepts a
captured NPZ. CPU is its default; the replay script defaults to CoreML for Mac.
The replay's YOLO backend uses MPS. No claim of portable acceleration is made.

Both replays use identical conservative depth sampling and a foreground box
selected by area initially, then overlap. `--min-area-fraction` can explicitly
restrict acquisition to the foreground; record that setting in comparisons. This is not a robust identity tracker:
crossing/occlusion needs separate validation. Scores from different networks
are not calibrated probabilities; compare threshold sensitivity, not just one
dropout percentage. No keypoint predictions are ground truth.

RTMLib's `RTMPose3d` output mixes crop-pixel X/Y with model-relative Z. It must
not be interpreted as metric camera XYZ. The probe exports RGB pixels and
relative Z separately. Depth provides metric coordinates for observed points;
nonzero lens distortion currently requires SDK deprojection and is rejected by
the replay. The experimental `monocular_prior` anchors nominal model Z with a robust
torso depth offset and rejects inconsistent anchors. Its output is always a
prediction. It can supply missing current-frame observations to human IK; it is
not a measured foot position, and nominal model scale still needs validation.

## Human constraints before robot GMR

`human_fit.py` optimizes six lower-body landmarks with fixed operator hip width,
thigh and shin lengths, robust observation error, and a maximum knee-flexion
constraint. Lengths can come from a separate calibration interval or an explicit
profile JSON `{"lengths_m": [hip_width, left_thigh, right_thigh, left_shin,
right_shin]}`. Camera-calibrated surface distances are provisional estimates,
not measured anatomical joint-centre distances. No robot-height scaling is used.

```
python evaluate_human_fit.py <calibration-camera.jsonl> <evaluation-camera.jsonl> \
  recordings/fit-output [--profile <operator-profile.json>]
```

For a `compare_rgbd.py` result, use an explicitly chosen static interval:

```
python fit_rgbd.py recordings/rtmw.jsonl --calibration-start 10 \
  --calibration-end 20 --use-prior --output recordings/rtmw-fitted.jsonl
```

Times are relative to capture start. Verify that the operator is actually in
frame and still during this interval. Evaluation starts after calibration ends.

Calibration is frozen before evaluation. The fit has no future frames, temporal
smoothing or indefinite hold. Missing observations require an explicit current
metric prior or the fit rejects the frame. Output marks corrected observations
`fitted` and prior-only points `predicted`; raw input stays intact. Large required
adjustments and solver failures are rejected. This does not yet model knee bend
plane, twist, feet contact or the whole torso. It is not the robot's GMR solver.

Constant output lengths are imposed by construction; they cannot prove better
pose accuracy or robot stability. Before enabling control, validate position,
orientation, latency, person identity, occlusion and physical robot rollouts.

```
python -m unittest test_human_fit test_rtmw_probe test_rgbd_recording test_camera_open
```
