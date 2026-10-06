# Walking and arm ablation — 2026-10-07

Status: locally reproduced offline; the new APK still needs a Quest trial.
Baseline: `f7aa850edd983ed9720d7aa1620fc4914ddf19d4` (corrected Y-up,
HMD room translation, primitive colliders). Question: which recent changes
contribute to poor walking and arm correspondence?

## Decision

Use the tracked **pelvis** for room translation, with the existing anatomical
scale and A-calibration origin. HMD orientation still sets forward at calibration.
When the entire person moves, the displacement remains; moving the head relative
to a planted body no longer translates every foot target. This also puts body
positions, orientations and their derivatives on the returned body timestamp.

The original adapter combined `rootXY = scaled head displacement` with limb
positions relative to the pelvis. It therefore added this common error to all
limbs, including planted feet:

```text
scale × [(head − pelvis)now − (head − pelvis)at calibration]
```

In the private walking interval, horizontal head-relative-pelvis spans are
12.9 / 18.3 cm. Relative speed median / p95 is 0.147 / 0.364 m/s; at the
recorded scale this adds approximately 0.126 / 0.312 m/s of spurious translation.
The old head/limb samples also differed in time by a median 29.75 ms.
Pinned upstream GMR uses `Pelvis` as the human root and scales that root directly:
[GMR config](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/general_motion_retargeting/ik_configs/xrobot_to_g1.json),
[GMR source](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/general_motion_retargeting/motion_retarget.py).

A regression with a stationary body and 20 cm head sway changes robot root by
16 cm with the old adapter; the new adapter gives below 0.001 mm of drift.
The independent 501-frame anatomical leg check still preserves 0.4 m of robot
translation, alternating foot lifts, and joint mapping within 0.000004 rad.
Duplicate/older body timestamps, pause/resume and source-coordinate changes pass.

## Protocol and limits

- One complete private full-body episode: 36 seconds, two resets/calibrations.
  Dynamics start at each reset and rebuild policy history; all variants use the
  same recorded input schedule and grips. No participant trajectory is published.
- A public walking trajectory (`0807_yanjie_walk_002`), 14 seconds including a
  two-second initial hold; nine other pinned public walking clips for full-duration
  validation. Clips are separate trajectories from the same collection, not
  independent operators or devices.
- A 22-second arm exercise uses smooth robot joints, then robot FK → synthetic
  Meta landmarks → GMR → policy. Both arms, elbow bends and wrist rotations are
  included. This checks geometry and execution against known robot motion; it
  **does not validate Meta's inferred human skeleton**. All generated joint angles
  are within model limits. Direct-to-policy controls bypass GMR on identical motion.
- Each factor is changed alone from baseline. Startup blending and head/body time
  alignment variants affect only the private replay; their synthetic cells are
  unchanged controls, marked N/A below.
- Tracking/roughness metrics compare only matching surviving time intervals in
  each segment, after one second of active warmup. Every row includes its matched
  baseline in [aggregate JSON](gait_ablation_results.json); falls and full outcome
  durations are retained separately. An early fall cannot improve the score by
  truncating the difficult portion.
- Roughness is RMS joint velocity above 6 Hz at the 100 Hz control snapshots;
  it can contain intended motion harmonics. Loaded-foot slip uses ground contact
  points with normal force >1 N, force-weighted per frame, evaluated when total
  foot force >20 N. Wrist position/orientation errors use each pose's pelvis frame.
  XY trajectory error removes initial translation only, retaining heading drift.
- Original hardware dynamics are not fully reproduced on this Mac. The original
  adapter reproduces the second fall to below 9e-8 state error; the first recorded
  fall is not reproduced. Counterfactual results are bounded by the recording end.
  Confidence 1 on estimated feet does not establish correct human contact timing.

Host: MacBook Air M4; MuJoCo 3.3.7, ONNX Runtime 1.23.2, NumPy 2.2.6.
Policy and PD gains are unchanged. Physics is 1 kHz, policy 100 Hz; source clips
are approximately 30 Hz, synthetic arms 60 Hz. Numerical GMR parity uses the
pinned upstream GMR/Mink oracle. There are 13 main configurations, two direct
controls and 18 additional full walking rollouts.

During the independent audit, two fixture errors were found and corrected before
these final results: the HMD landmark must be `head_mocap` (the `head_link` origin
is near the torso), and generated left elbow angles must remain below 1.7 rad.
Earlier exploratory percentages are superseded by this rerun.

## Main one-factor results

Private replay tilt is p95 in degrees. Arm errors are physical wrist RMS relative
to the known robot arm exercise, in centimetres / degrees. Fall counts refer to
complete rollouts, not just metric windows.

| Configuration / changed factor | Private falls | Private tilt p95 | Arm wrist cm / degrees | Decision |
|---|---:|---:|---:|---|
| Baseline | 0/2 | 9.61 | 2.20 / 8.10 | comparison |
| Roll back Y-up correction | 1/2 | 17.46 | 4.66 / 12.59 | retain corrected axes |
| Unit scales for root and arms | 0/2 | 9.44 | 2.37 / 6.95 | no consistent benefit; public XY error increases |
| Upstream coefficients at explicit 1.75 m | 0/2 | 9.43 | 2.06 / 7.21 | no consistent walking benefit; source-specific scaling matters |
| Pelvis room translation | 0/2 | 9.43 | 2.20 / 8.10 | shipped; removes demonstrated geometric error |
| Roll back ankle landmark scale | 0/2 | 9.68 | 2.21 / 8.11 | retain ankle mapping |
| Split proportion correction from rigid foot offset | 0/2 | 11.25 | 2.20 / 8.10 | worse recorded tilt/tracking; not shipped |
| 40 ms causal filter on root velocities | 0/2 | 9.43 | 2.20 / 8.10 | recorded XY error increases; not shipped |
| Restore collision meshes | 0/2 | 10.07 | 1.97 / 7.12 | affects arms, no consistent gait improvement |
| Remove restored drivetrain parameters | 2/2, before tracking | N/A | fails during exercise | restored parameters are necessary |
| Disable robot self-contact, retain floor/objects | 0/2 | 10.04 | 1.67 / 6.41 | diagnostic control only |
| Startup blend 0.5 → 2 s | 0/2 | 9.68 | N/A | no sustained improvement |
| Interpolate HMD position to returned body time | 0/2 | 9.97 | N/A | no consistent improvement; pelvis avoids mixed-time translation |

The collider changes preserve body masses/inertias, joint/actuator parameters and
foot pads. Removing drivetrain restoration intentionally reverts the hand model's
joint armature/damping/friction/force-limit parameters. These are distinct factors.
Restoring meshes reduces wrist error in the arm exercise, but increases arm
roughness in the private replay (0.292 → 0.432 rad/s). It does not explain the
remaining walking instability as a single cause. Disabling self-contact reduces
arm-exercise wrist error to 1.67 cm / 6.41°, confirming a contact contribution;
physical tracking error remains, and this diagnostic configuration is not shipped.

## Pelvis correction: walking and arms

| Measure | HMD baseline | Pelvis correction |
|---|---:|---:|
| Private replay: falls | 0/2 | 0/2 |
| Private replay: leg roughness, rad/s | 0.777 | 0.762 |
| Private replay: wrist position / angle | 2.88 cm / 14.98° | 2.73 cm / 14.46° |
| Primary public walk: root command velocity RMSE vs direct control | 0.0770 m/s | 0.00335 m/s |
| Primary public walk: leg roughness | 1.653 rad/s | 1.533 rad/s |
| Primary public walk: loaded-foot slip p95 | 0.499 m/s | 0.378 m/s |
| Primary public walk: physical XY trajectory RMSE | 0.847 m | 0.643 m |
| Primary public walk: tilt p95 | 9.22° | 8.65° |
| Arm exercise: physical wrist error | 2.20 cm / 8.10° | 2.20 cm / 8.10° |

Final primary-walk improvements are approximately 7% in leg roughness, 24% in
loaded-foot slip and 24% in XY error. The root-command error drops about 96%.
The private replay's XY score uses the old HMD-derived shared reference: it changes
0.490 → 0.517 m, so it cannot independently establish human pelvis accuracy.

For the legal synthetic arm exercise, GMR reproduces the known orientations to
below 0.001°. Bypassing GMR still gives 2.18 cm / 8.00° physical wrist error,
versus 2.20 cm / 8.10° through GMR. Most residual error in this exercise is in
policy/physical execution. Actual human arm accuracy still requires a measured
reference; the synthetic roundtrip alone cannot prove correspondence on Meta.

## Additional walking trajectories

| Public clip | HMD fall | Pelvis fall | Result |
|---|---:|---:|---|
| 001 | no | no | roughness slightly worse (~3%) |
| 003 | no | no | roughness slightly worse (~6%); XY error improves |
| 004 | yes | yes | startup challenge; fails before walking window |
| 005 | no | no | roughness improves; XY error worsens |
| 006 | no | no | roughness and XY error improve; substantial tilt remains |
| 007 | yes | no | avoids baseline fall; matched tilt p95 43.78° → 8.07° |
| 008 | yes | yes | walking instability remains in both variants |
| 009 | yes | yes | startup challenge; fails before walking window |
| 010 | yes | yes | startup challenge; fails before walking window |

Four ordinary-start clips complete with either configuration; one improves from
fall to completion; one still falls. Three others start in deep/tilted poses and
fail during entry into that pose, before there is a warmed walking interval.
All nine are reported; none is silently excluded. These results support fixing
head-relative target drift, but do not establish reliable walking in general.

## Reproduction

Export the exact baseline adapter from the FURRY Git checkout into ignored output
storage (the current production adapter already contains the pelvis correction):

```sh
mkdir -p prototypes/quest_g1/outputs/ablation
# Run from the FURRY checkout, or adjust the output path to your prototype folder.
git show f7aa850:prototypes/quest_g1/android/native/meta_retarget.cpp > prototypes/quest_g1/outputs/ablation/baseline_frozen.cpp
```

From `prototypes/quest_g1`, with local pinned dependencies provisioned:

```sh
./android/check_gmr_mac.sh
.venv/bin/python scripts/run_gait_ablation.py \
  --baseline-source outputs/ablation/baseline_frozen.cpp \
  --episode /private/path/to/episode \
  --walk /path/to/robot-walk.txt
.venv/bin/python scripts/summarize_gait_ablation.py outputs/ablation
.venv/bin/python scripts/check_gait_holdouts.py outputs/ablation --prepare
```

The text motion format is source FPS followed by 36 qpos values per row, wxyz
quaternion. Normalize the initial XY/yaw consistently for both variants. Place
additional normalized public clips in `outputs/ablation/holdouts/*.txt` for the
holdout command, or use `--prepare` to normalize the nine pinned public fixtures. Experimental source copies and per-frame results remain under
ignored `outputs/`; they are not build options in the Quest app. Aggregated
results are committed in `gait_ablation_results.json`. Native replay CSVs contain
compact robot qpos/qvel, GMR references, segment/applying flags and foot contacts.

Next unresolved work is support/contact consistency of Meta's estimated legs and
policy tracking during turns/leaning; startup from deep poses is a separate
failure. Do not treat an axis correction or a no-fall short replay as validation
of generated legs against real foot tracking. A fresh Quest recording is needed
to verify the new translation behavior on the device.
