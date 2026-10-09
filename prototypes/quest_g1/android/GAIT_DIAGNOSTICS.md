# Quest gait diagnosis — 2026-10-07

The 2026-10-06 diagnostic bundle contains one complete full-body GMR episode
(36.0 seconds, 2,679 source frames, 3,355 control snapshots, two calibrations and
two falls), plus two older wrist-IK episodes. Only the full-body episode is used
for this gait comparison. Raw tracking, logs and identifying device data remain
local under ignored `outputs/`; this report contains aggregate results only.

This report describes the first Y-up correction. The subsequent paired
[walking/arms ablation](GAIT_ABLATION.md) changes room translation from HMD
to the body pelvis after identifying head-relative drift of planted feet.

## Findings

- Physics was keeping up during the recorded tracking intervals: median cycle
  6.47 ms, p95 7.46 ms, maximum 8.43 ms against a 10 ms budget. RTF was
  0.995–1.006. GMR median 0.398 ms, p95 0.516 ms. Contact count 9–21.
  These samples do not support collision overload as the cause of these falls.
- All recorded body samples are valid, confidence is 1, and source input has
  no gaps longer than 13.9 ms. Body timestamps lag predicted XR display time
  by about 29.7 ms. Confidence is not evidence of accurate foot contact timing.
- Policy, scene, GMR model, config and adapter source hashes match the baseline.
  Replaying its source poses reproduces all 2,747 applied commands with maximum
  absolute difference 1.82e-12. Native GMR computation is not being corrupted
  on Android.
- The bind-coordinate adapter incorrectly used pelvis→chest as world up.
  In this source T-pose the chest is about 3.8 cm behind the pelvis, producing
  about 5° of false reference tilt. Anatomical spine shape is not gravity.
  The fix uses Y-up and the horizontal shoulder direction for neutral heading.
  It retains head-based room displacement, existing scales, GMR costs and the
  original policy/PD gains.

The new regression supplies a 4 cm chest-depth offset independently of an
otherwise identical anatomical leg fixture. Before the fix, maximum leg error
is 0.085785 rad and height error 6.738 mm; after it, 0.000004 rad and below
0.001 mm. It also tests 501 frames of alternating leg lifts, translation,
duplicate/out-of-order timestamps and resume. Shoulder fixture positions use
the anatomical pitch pivot, while orientations follow the upper-arm yaw link.

## Recorded-motion experiments

Dynamic replay starts at each recorded reset, rebuilding policy history from
there. Saved commands and recomputation with the original adapter agree. The
second recorded fall reproduces at simulator time 4.76 s; its state discrepancy
before falling is below 9e-8. The first segment diverges over time and does not
reproduce its hardware fall on this Mac. Therefore the following is a bounded
counterfactual comparison, not a claim of identical Quest dynamics.

| Adapter variant | First segment | Second segment |
|---|---|---|
| Original | no offline fall, min pelvis 0.743 m | falls at 4.76 s |
| Unit scales for both arms and legs | falls at 8.582 s | falls at 4.76 s |
| Upstream coefficients at an explicit height of 1.75 m, otherwise our adapter | no offline fall, min 0.753 m | falls at 4.736 s |
| Y-up bind basis (the shipped correction) | no offline fall, min 0.764 m | no offline fall, min 0.771 m |

With the corrected basis, the largest physical tilt over both available
segments is 13.63°. Median reference pelvis pitch in the first walking segment
changes from −11.80° to −6.86°. No global scaling coefficient was tuned to
obtain this result. Hardware walking still requires verification on the Quest.
The recording ends shortly after the original second fall, so its corrected
continuation is not evidence of long-term stability.

Other experiments moved the root anchor from HMD to pelvis or split foot
proportion correction from the rigid ankle→toe offset. Neither consistently
improved this recording: some combinations fell earlier in the first segment.
They are **not shipped**. In the original adapter a source ankle separation of
13 cm can yield a toe target below the floor; its interpretation depends on
foot pitch and offsets. The adapter now clamps only the toe position to the
flat-foot level (see "Foot trips and toe targets"); it still generates no
artificial gait.

## What scaling actually does

`qpos[2] = 0.8` initializes the robot bind pelvis height in metres; it is not a
fixed scale applied to every human. The episode's anatomical ratios are 0.85719
for root/legs and 0.77869 for arms. Scaled targets go directly to `SetTargets`;
there is no second upstream scaling pass.

Pinned upstream GMR uses a table of 0.9 for root/legs and 0.8 for arms,
multiplied by `actual_human_height / 1.8`. TWIST2 passes that parameter and its
CLI default is 1.5 m. At an explicitly supplied 1.75 m those coefficients are
0.875 and 0.77778. Copying just the coefficients into our Meta adapter is not
an end-to-end reproduction of the authors' PICO/XRobot input processing.

Sources: [GMR scaling](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/general_motion_retargeting/motion_retarget.py),
[XRobot→G1 config](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/general_motion_retargeting/ik_configs/xrobot_to_g1.json),
[TWIST2 teleoperation](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/deploy_real/xrobot_teleop_to_robot_w_hand.py).

## Reproduction with a local recording

```sh
.venv/bin/python scripts/analyze_gait_recording.py /path/to/episode /path/to/runtime_stats.csv
./android/check_gmr_mac.sh
G1_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
export DYLD_LIBRARY_PATH="$G1_PACKAGES/mujoco:$G1_PACKAGES/onnxruntime/capi"
# Original saved commands, and a counterfactual using the currently built adapter:
android/build/replay_gmr_dynamics android/assets /path/to/episode saved outputs/saved.csv
android/build/replay_gmr_dynamics android/assets /path/to/episode retarget outputs/retarget.csv
```

Use the baseline adapter to check exact historical command parity with
`replay_gmr`. A corrected adapter intentionally produces different commands.
The dynamics helper requires recorded resets, matching model/policy assets,
and no focus/reference-space-change events. It skips any prefix before the
first reset and stops each segment on a simulated fall or recorded reset/end;
its exit code indicates valid replay, not that the robot never fell.
The statistics parser reports the incomplete final runtime CSV row, which
was captured while the app was writing. Episode streams are complete and
are validated strictly.

## Foot trips and toe targets — 2026-10-09

Replay CSVs from `replay_gmr_dynamics` and `ablation_motion` now add per-foot
`floor_force_*`, `floor_brake_*` and `toe_target_*` columns. `toe_target` is the
estimated GMR toe-task height relative to a flat grounded foot, before the
floor clamp; it exists only where the Meta adapter solved the frame and is NaN
in `saved`/`direct` rows.

```sh
.venv/bin/python scripts/analyze_foot_trips.py outputs/retarget.csv --output outputs/trips.json
```

A trip is a physical touchdown while the commanded foot is still in swing:
more than 1 cm above the other commanded foot and moving faster than 0.3 m/s.
A braking trip also meets more than 50 N of horizontal floor force against the
foot's travel. The command, not the GMR qpos, defines the reference, so every
replay mode is scored the same way. Trips are not falls.

Baseline without a headset: nine pinned public walks in the production scene
with props removed, MJX sole capsules on the floor, policy and gains
unchanged. Clip 004 is rejected by the upright-calibration check in both
modes; 009 and 010 fall during startup and have no scored window. The remaining seven give 91 s of scored walking.

| Input path | Trips/s | Braking trips/s | Toe target below −5 mm | Trips with toe below in previous 100 ms |
|---|---:|---:|---:|---:|
| `direct` (source robot motion) | 1.58 | 1.09 | N/A | N/A |
| `meta` (synthetic Meta from robot FK) | 1.66 | 1.23 | 14.4% of foot-rows | 1 / 145 |

Trips occur with exact source motion, so they are not created by Meta or GMR.
With exact synthetic landmarks, below-floor toe targets come from source feet
pitched toe-down by about 4–6° (median); 44–76% of such rows are on the foot
with the lower ankle, which the grounded adapter places at ankle height.
They almost never precede a trip here. Meta's estimated foot pitch is not
modelled by this fixture; compare a real Quest recording against this table.

### Sole pads and toe clamp

Two changes followed. TWIST2's four 5 mm sole spheres per foot replace the MJX
sole capsules, as in the source sim2sim model; foot boxes keep contact with
props only. The Meta adapter raises a toe-task position below the flat-foot
level to that level; the toe orientation task is unchanged. The optional
`G1_ENABLE_TWIST_GROUNDING` build regrounds every frame instead and does not
clamp. Same seven walks and scored window:

| Floor contact | Toe clamp | `direct` trips/s (braking) | `meta` trips/s (braking) | `meta` falls in scored clips |
|---|---|---:|---:|---|
| MJX capsules | no | 1.58 (1.09) | 1.66 (1.23) | 005 at 15.8 s, 008 |
| MJX capsules | yes | 1.58 (1.09) | 1.39 (1.10) | 008 |
| TWIST2 pads | no | 1.35 (0.90) | 1.31 (0.94) | 008 |
| **TWIST2 pads** | **yes (shipped)** | 1.35 (0.90) | **1.27 (0.83)** | 008 |

The clamp affects only the Meta adapter, so `direct` rows repeat. Replays are
deterministic but chaotic: single clips move by ±20% between variants, and
with about 120 trips the aggregate's counting noise is about ±0.12 trips/s.
Pads reduce trips in both input paths beyond that noise. The clamp's further
effect on pads (1.31 → 1.27) is within it. Requested toe targets stay below
−5 mm in 14.2% of foot-rows, so the clamp is active that often. The 008 fall
(about 6.6 s) and the 009/010 startup falls occur in every variant. The
synthetic fixture does not model Meta's estimated foot pitch; real Quest
skeletons follow.

### Real Quest skeletons: toe clamp

The 2026-10-08 capture in
[`datasets/quest-virtual-legs-v2`](../../../datasets/quest-virtual-legs-v2/2026-10-08/README.md)
is `human_skeleton_only`: the robot never ran, so it has no resets or commands
for `replay_gmr_dynamics`. `replay_meta_skeleton` drives the current adapter,
GMR and policy from its Meta body frames instead. Each of the 17 accepted
segments is cut into windows of at most 30 s; every window resets the robot,
calibrates on its first usable frame and blends in over 0.5 s. Stand scene,
default build (no floor guard, swing clearance or TWIST2 grounding):

| Toe clamp | Scored s | trips/s | braking trips/s | touchdowns/s | falls |
|---|---:|---:|---:|---:|---|
| no | 854 | 0.71 | 0.37 | 3.05 | stage 6 at 16 s |
| **yes (shipped)** | 849 | **0.49** | **0.27** | 3.04 | stage 13 at 12 s |

The clamp lowers trips in 13 of the 14 segments that contain walking; stage 13
is unchanged. The largest drops are variable-pace walking (stage 2, 1.52 →
0.57) and turns with a carried object (stage 10, 1.76 → 0.90), where 37% of
foot-rows ask for the toe below −5 mm. Over the whole capture that share is
17%, mostly on the supporting foot. One operator and one session; windows
restart the robot, unlike a continuous session. Rates are comparable only
between variants on the same input, not with the public walks above.

```sh
python3 -c "import json,sys; print('stage,first,last'); [print(f\"{s['stage']},{s['first_sequence']},{s['last_sequence']}\") for s in json.load(open(sys.argv[1]))['accepted_segments']]" \
  ../../datasets/quest-virtual-legs-v2/2026-10-08/audit/audit.json > outputs/segments.csv
unzip -q ../../datasets/quest-virtual-legs-v2/2026-10-08/raw/source-episode.zip -d outputs
for mode in clamp noclamp; do
  android/build/replay_meta_skeleton android/assets stand outputs/episode-1791467626074995 \
    outputs/segments.csv $mode outputs/quest-$mode.csv
done
.venv/bin/python scripts/analyze_foot_trips.py outputs/quest-*.csv
```

Running each segment as a separate file gives identical counts.
