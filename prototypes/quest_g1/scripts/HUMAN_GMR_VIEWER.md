# Offline human GMR viewer

`render_human_gmr.py` fits an articulated human MJCF to metric camera landmarks
using upstream GMR tasks and Mink IK. This is a diagnostic kinematic animation,
not a physically validated robot target and not a replacement for the Quest pipeline.

Requires the project's Python MuJoCo/Mink/SciPy environment and `vendor/GMR`.
Use JSONL from `skeleton/compare_rgbd.py` and its original RGB-D recording:

```sh
python scripts/render_human_gmr.py camera.jsonl local-output --rgbd rgbd-recording --height 1.75 --calibration 12 20
```

`--profile` reuses another export's proportions for a controlled comparison.
Lengths are bilateral medians from the calibration interval, normalized to the
supplied height. They are approximate surface-derived proportions, not anatomical
measurements. Floor orientation is estimated from lower-image depth points using
RANSAC and SVD. The transform is a proper rotation; camera vertical is not assumed
to be gravity. Inspect `floor.json` before trusting a new scene.

Middle-joint depths are rejected if both adjoining bone lengths are inconsistent
while the endpoints remain feasible. The unmeasured joint is then inferred by IK,
and shown hollow. This conservative check does NOT resolve all occlusions or
incorrect high-confidence 2D landmarks. There is no support/contact constraint,
balance model, or dynamics. A lower ankle above 18 cm is labelled unconfirmed
support in the viewer, not automatically classified as a tracking failure or
forced to the floor. Actual jumps also produce that label.

Build an inline viewer from two exports with matching source timestamps:

```sh
python scripts/build_human_viewer.py yolo/animation.json rtmw/animation.json local-viewer.html
```

The fragment uses the Codex visualization runtime's theme and control utilities.
Playback interpolates local joint rotations with SLERP and evaluates forward
kinematics, preserving bone lengths. It includes play/pause, scrub, speed, camera
orbit/zoom, detector comparison and optional unfiltered observations.

Keep recordings, personal proportion profiles, generated JSON/MJCF and populated
viewers local and ignored. Only these generic tools belong in Git.
