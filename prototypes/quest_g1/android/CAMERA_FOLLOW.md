# First-person camera and HMD objective

The observer view retains its existing world placement. Left stick click toggles
first person. Calibrate with A while standing upright and looking forward.

First-person rendering now maps the robot camera centre to the **current** HMD
centre on every frame, using a fixed, level calibration rotation. Head rotation
remains responsive even while the robot is delayed. The entire virtual scene
gets a uniform visual scale equal to standing HMD height divided by robot camera
height. Physics dimensions, inertia, controller gains and policy weights stay
unchanged. This fixes the previous one-time anchor and robot-height mismatch.
The camera's previous 30-degree downward mounting pitch is removed.

In first person only, GMR also receives an optional camera SE(3) objective. Its
position and orientation costs are 10 and 3; foot position cost remains 100.
Pelvis translation still drives locomotion. Head displacement relative to pelvis
acts on this separate objective, rather than translating all planted feet.
The camera orientation is relative to the A-calibration pose. G1 has no actuated
neck in this model, so the goal is soft and competes with torso and foot goals.
This extends GMR's loss; it does **not** retrain the pretrained TWIST2 policy.
Velocity commands alone do not guarantee that the physical robot catches up to
an absolute camera pose. HUD/CSV show the remaining physical camera error.

The HUD uses larger glyphs and shorter lines. CSV retains the detailed metrics,
with `camera_error_m`, `camera_error_deg` and `visual_scale` appended. Camera
errors are -1 while the objective is off. Recording formats remain compatible.

## Verification

- 28 Python tests pass.
- Native GMR without the optional objective matches pinned upstream GMR across
  70 frames at each height 1.30, 1.75 and 1.95 m (maximum tangent error < 2e-4;
  observed errors approximately 1e-13).
- Existing leg mapping and independent head sway checks pass.
- A synthetic 8 cm / 0.15 rad camera goal reduces combined position/rotation
  error from 0.230000 to 0.195109; maximum foot displacement is 0.000099 m.
  Joint limits and camera-target reset are checked.
- Stereo panel projection, live camera centre under HMD translations/crouching,
  uniform scale and level world up pass native transform checks.
- Android APK builds successfully. These checks do not establish improved live
  walking or full head-pose correspondence on a user recording.
