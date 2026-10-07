# Walking and arm ablation — 2026-10-07

Status: locally reproduced offline; the new APK still needs a Quest trial.
Baseline: `f7aa850edd983ed9720d7aa1620fc4914ddf19d4` (corrected Y-up,
HMD room translation, primitive colliders), for the historical LAB cohort.
The additional unobstructed cohort below preserves full motion durations.
Question: which recent changes
contribute to poor walking and arm correspondence?

## Decision

Use the tracked **pelvis** for room translation, with the existing anatomical
scale and A-calibration origin. HMD orientation still sets forward at calibration.
When the entire person moves, the displacement remains; moving the head relative
to a planted body no longer translates every foot target. This also puts body
positions, orientations and their derivatives on the returned body timestamp.
Further collision-geometry changes are candidates under evaluation; the current
production decision remains pelvis translation with the existing model.

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

The historical 13-configuration comparison and its nine public holdouts use the
physical **lab scene with props**. These outcomes remain lab-task results. The
additional 20-rollout, ten-motion unobstructed comparison is reported separately;
its results must not be mixed with this scene or the historical 14-second cap.

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
controls and 18 additional full LAB walking rollouts. The subsequent unobstructed
cohort adds 20 full rollouts on ten motions, retaining the same physical hand model.

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
| 007 | yes | no | LAB fall avoided; both complete without props, so this is not a flat-floor fall benefit |
| 008 | yes | yes | direct LAB control trips on the T-object; both hand-model variants also fall without props |
| 009 | yes | yes | startup challenge; fails before walking window |
| 010 | yes | yes | startup challenge; fails before walking window |

Four ordinary-start clips complete with either configuration; one improves from
fall to completion; one still falls. Three others start in deep/tilted poses and
fail during entry into that pose, before there is a warmed walking interval.
All nine are reported; none is silently excluded. Head-relative target drift is
a demonstrated geometric error, but changes in lab paths may also change prop
contacts. These fall counts do not establish reliable unobstructed walking.

## Unobstructed full-duration walking comparison

Twenty additional rollouts remove only `table`, `cup`, `t_box` and `t_target`
from copied assets: ten public motions through the original HMD adapter and the
pelvis adapter. The physical G1/Dex hand model, robot and floor parameters,
primitive colliders, policy, gains and floor friction 0.6 remain unchanged.
All ten motions run to completion or a valid fall. Their aggregate full outcomes
and matched-window metrics are retained in the separate `unobstructed_walking`
section of [the JSON report](gait_ablation_results.json).

Both adapters fall in **4/10** motions: 004, 009 and 010 during entry into their
deep/tilted initial pose, and 008 during walking. Seven motions have usable paired
tracking intervals, including 008's common surviving prefix.
Clip 007 completes with both adapters once props are removed. Its previous LAB
fall avoidance therefore does not establish a fall-rate benefit on a flat floor.

| Full unobstructed primary motion 002 | HMD adapter | Pelvis adapter |
|---|---:|---:|
| Falls | 0 | 0 |
| Leg roughness | 1.598 rad/s | 1.423 rad/s |
| Loaded-foot slip p95 | 0.483 m/s | 0.441 m/s |
| Physical XY trajectory RMSE | 0.958 m | 0.840 m |

These changes are approximately 11%, 9% and 12% improvements respectively.
This is the **full 15.36-second** rollout, including the two-second initial hold;
it has a different horizon from the 14-second LAB exercise above. Improvements
are not uniform: clip 003's roughness increases, clip 005's roughness/slip/XY error
increase, and clip 006 still comes close to falling (pelvis minimum height
0.3985 m, maximum tilt 32.1°). Clip 008 falls at approximately 6.78 s through the
pelvis adapter and 6.90 s through the HMD adapter. Correcting target translation
does not establish robust walking or accurate human foot contact inference.

For reproduction, use the separate-results options and a directory containing
all ten normalized public clips, including primary motion 002:

```sh
.venv/bin/python scripts/check_gait_holdouts.py outputs/ablation \
  --clips-dir /path/to/ten-normalized-public-clips \
  --empty-scene --results-dir outputs/unobstructed-reproduction
```

This preserves the historical LAB results and writes copied clean assets under
the new results directory's `_assets/<variant>` folders. The default `--prepare`
set contains nine holdouts; include 002 explicitly for this ten-motion comparison.

## Changes outside this comparison

The former wrist-only IK has no full-body walking reference, so it is not scored
as a locomotion baseline. GMR is tested against the pinned numerical oracle and
against direct-to-policy robot-motion controls; this does not constitute a
paired human comparison of old wrist IK and Meta full-body tracking.
Visual mesh simplification, passthrough and HUD placement do not enter offline
physics state; their device timing still requires a separate Quest measurement.

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
Saved-command mode has no full GMR qpos: `ref_available=0` and NaN reference
columns explicitly mark that limitation; paired pose metrics require retarget mode.

Next unresolved work is support/contact consistency of Meta's estimated legs and
policy tracking during turns/leaning; startup from deep poses is a separate
failure. Do not treat an axis correction or a no-fall short replay as validation
of generated legs against real foot tracking. A fresh Quest recording is needed
to verify the new translation behavior on the device.

## Separate floor-friction sensitivity

An additional comparison in the current **lab scene, including its props**, uses
the current **pelvis** adapter in both runs and
changes only the floor's tangential friction from 0.6 to 1.6. It is separate from
the 13-configuration historical-HMD ablation above; the latter is unchanged.
Foot pads, inertias, drivetrain, contact dimensions and solver parameters match.

| Measure | Pelvis, friction 0.6 | Pelvis, friction 1.6 |
|---|---:|---:|
| All nine public motions in the lab scene: falls | 4/9 | 7/9 |
| Primary walk: leg roughness | 1.533 rad/s | 1.970 rad/s |
| Primary walk: physical XY trajectory RMSE | 0.643 m | 1.133 m |
| Private replay: leg roughness | 0.762 rad/s | 0.970 rad/s |

The increased friction introduces falls in clips 001, 005 and 007. The three
startup challenges remain included in both totals. Each metric comparison uses
the same surviving time interval with its matched baseline; full fall outcomes
remain separate. This experiment rejects increasing friction as a walking fix
for the tested lab rollouts. It does not establish the effect in an empty walking
environment; prop contacts and changed paths can affect these outcomes.
Production retains 0.6.

The pinned [recommended sim2sim launcher](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/sim2sim.sh)
selects the [no-hand model with floor friction 1.6](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/g1/g1_sim2sim_29dof.xml).
The [hand model](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/g1/g1_sim2sim_29dof_with_hands.xml)
uses 0.6. The available training configuration specifies terrain friction 1.0
and robot-shape friction randomization over [0.1, 2.0], through
[HumanoidCfg](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/legged_gym/legged_gym/envs/base/humanoid_config.py)
and [G1MimicPrivCfg](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py).
These source settings do not establish the published checkpoint's resolved
training configuration; this is not evidence that it was trained at friction 1.6.

Reproduce as an explicitly selected, separate cohort, using the current
production adapter rather than the historical HMD source:

```sh
.venv/bin/python scripts/run_gait_ablation.py \
  --output outputs/floor-ablation --variants baseline floor_1_6 \
  --episode /private/path/to/episode --walk /path/to/robot-walk.txt
.venv/bin/python scripts/summarize_gait_ablation.py outputs/floor-ablation
.venv/bin/python scripts/check_gait_holdouts.py outputs/floor-ablation \
  --candidate floor_1_6 --prepare
```

`floor_1_6` is opt-in and does not enter the default experiment list. The holdout
runner uses each variant's generated assets when present. Existing normalized
clips can instead be supplied with `--clips-dir outputs/ablation/holdouts`.
The separate `floor_sensitivity` section of `gait_ablation_results.json` contains
aggregate comparisons only; source traces remain in ignored `outputs/`.

## Direct walking controls beyond GMR

Four public walks were also sent directly to the policy, bypassing both the
synthetic Meta adapter and GMR. The direct reference matches all 36 source robot
qpos values. Both paths use the current physical lab model, including its props,
and floor friction 0.6.

| Public clip | Direct control | Pelvis GMR control |
|---|---|---|
| 001 | completes | completes |
| 006 | completes, minimum height 0.3995 m and maximum tilt 32.2° | completes, substantial tilt remains |
| 007 | completes | completes |
| 008 | falls at approximately 5.92 s | falls at approximately 6.47 s |

Clip 008's planned rollout is 9.31 s, including the initial hold. On the common
3.00–5.91 s interval, GMR leg reference RMSE is 0.00000166 rad and root XY
reference RMSE is 0.000000108 m. Physical leg RMSE is 0.184 rad through GMR and
0.243 rad through direct control. These are robot-motion comparisons, not human
tracking ground truth.

The direct clip 008 contact audit found robot-to-T-object contacts starting at
4.73 s, including approximately 391 N at the right ankle and 534 N at the right
knee before the fall. The direct LAB failure is therefore collision-confounded.
The subsequent prop-disabled controls still fall with the hand model, as detailed
below. The LAB contact event and the clean-scene failure are separate evidence;
neither identifies a defect in the neural network alone. Policy execution, model
and playback remain possible contributors. Completion of clip 006
also leaves little height margin. Aggregate paired metrics and full outcomes are
stored separately under `direct_walk_holdouts` in `gait_ablation_results.json`.

## Model controls and numerical validity

The official no-hand G1 model completes direct motion 008 for its full 9.31-second
rollout with floor friction 0.6 and 1.6. The current hand model still falls after
removing all props: direct control at approximately 6.59 s and pelvis GMR at
approximately 6.78 s. Restoring collision meshes or disabling self-contact in
that empty hand-model scene also leaves a valid fall, at approximately 6.82 s
and 6.74 s respectively. Those controls do not isolate one collider as the cause.

Compiled model comparison finds identical shared joint/actuator parameters, foot
pads and solver settings. Robot masses are 33.342142 kg without hands and
34.3952352 kg in the current Dex hand model, a 1.0530932 kg difference including
changed wrist inertias and the finger bodies. All named robot body masses,
inertias and inertia frames in the current model match the pinned official
**with-hands** model exactly. Primitive replacement did not create the extra
mass. Reverting only wrist inertias to the no-hand values delays motion 008's
fall to approximately 8.02 s but still does not complete it; this is a diagnostic,
not a physically faithful production fix. The separate collision fidelity audit below confirms geometry errors, but its
replacement candidates are not shipped.

An exploratory finger-inertia multiplier of 0.001 appeared to complete a rollout,
but MuJoCo emitted BADQACC warnings and automatically reset its state repeatedly.
Its apparent no-fall result is invalid and excluded from gait conclusions.
Both offline harnesses now reject numerical warnings, nonfinite state and
unexpected simulation-time reset with exit code 2. Tests of the deliberately
unstable model fail in both motion and private replay harnesses; healthy controls
exit successfully. A retrospective scan of the original 13-configuration cohort's
39 logs/CSVs finds no numerical warning or within-segment time reset/non-10 ms
interval. Numerical validity is therefore distinguished from successful walking.

The separate `model_diagnostics` and `numerical_validation` JSON sections store
compact outcomes, parameter comparisons and validation provenance. They omit
per-contact records and participant frames. A successful no-hand control does
not validate the hand model or Meta's human foot estimates, and these single-motion
diagnostics do not establish a reliable walking fix.

## Primitive collision fidelity audit

The compiled mesh-to-body transform is retained correctly. The fitting heuristic
uses the mesh bounding box, however: the hip-roll capsule extends up to 33.25 mm
beyond the source convex hull, and wrist-roll/yaw housing boxes have about
1.67 / 1.64 times its volume. This creates contacts that the source mesh lacks.

A reproducible static comparison uses 1,024 legal small joint perturbations around
the actual reset pose. Wrist-roll versus wrist-yaw contacts appear in 788 / 813
left/right poses with the current boxes and in zero with the source meshes.
At the exact reset pose, thumb-to-hip contact already exists with source meshes:
7.27 / 7.37 mm penetration becomes 27.70 / 25.36 mm with the primitives. Thus the
reset-pose finding is an amplified contact, not evidence that all source contact
should be disabled. The existing no-intersection check exercises `qpos0`, which
differs from the actual reset pose; the balanced-reset check does not establish
geometric fidelity. Fingers are fixed at zero in this static comparison.

Rounded wrist alternatives remove the demonstrated false wrist pairs. Inscribed
compound alternatives for the hips also reduce false contacts, but lose real
mesh contacts: on this synthetic pose set, hip contact recall changes from about
90.8% with current primitives to 81.2% / 85.6% with three/six fitted ellipsoids.
Six hip ellipsoids cover approximately 83% of the source hull volume. These are
approximation tradeoffs, not validated replacements for external manipulation.

Dynamic controls remain mixed. The combined three-capsule hip/four-rounded-wrist
candidate still falls on empty-scene 008: direct control around 6.70 s and GMR
around 6.67 s, against 6.58 / 6.77 s before that geometry change. Wrist-only
rounding reduces private wrist orientation error from 14.46 to 9.01 degrees,
but position error increases from 2.73 to 2.78 cm and arm roughness increases.
The synthetic arm exercise changes 2.20 cm / 8.10 degrees to 2.17 cm / 8.24 degrees.
No alternative is shipped as a walking fix, and masses, gains and contact masks
are unchanged in the APK. The remaining geometry defect and hand-model policy
robustness stay open for a faithful replacement and further controlled evaluation.

Reproduce the independent static audit without recordings or a headset:

```sh
.venv/bin/python scripts/check_collision_fidelity.py
```

This writes only ignored aggregate output. Mesh contact is a geometric reference,
not independent human or hardware ground truth. Small independent joint
perturbations are not a distribution of human walking. The `collision_fidelity`
JSON section preserves the aggregate findings, source hashes and limitations.
