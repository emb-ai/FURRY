# Virtual Legs swing ankle +10 cm

`virtual_leg_lift.h` implements an optional causal input adapter. It operates on
raw 14-point Meta input before GMR, independently of camera leg corrections.
The separate APK catalogue entry `quest-111586-500-lift10` uses the unchanged
TWIST2 +500 weights and activates this adapter. Other modes remain unchanged.

The target is 10 cm above the **opposite raw ankle's support level**, as chosen by
the user. It is not ankle Y=0.10 in STAGE, not a +10 cm additive offset at every
frame, and not a physical sole-clearance constraint. Already higher feet are not
lowered. The original support ankle and all upper-body positions are unchanged.

Causal swing estimate: ankle height difference >12 mm plus filtered horizontal
speed >0.12 m/s or upward speed >0.04 m/s; stop below 4 mm while descending/slow,
or after 1.2 s. No forced left/right alternation and no use of future frames.
A smoothstep height envelope from 4–40 mm and 60 ms weight ramp remove an instant
height discontinuity. Thus lift ramps during takeoff/landing rather than imposing
10 cm throughout every estimated swing frame. The detector is heuristic, not a
measured or ground-truth contact label. Long gaps, invalid tracking, calibration
and pause reset its history; duplicate source timestamps are idempotent.

Two-segment analytic IK preserves each input frame's thigh and shin lengths,
keeps the hip fixed and ankle horizontal position fixed, and chooses the original
knee bending side. Hip/knee orientations rotate with their new segment directions;
ankle orientation stays unchanged. Unreachable targets reduce vertical lift rather
than stretching a limb. Immutable full-body raw SDK data remains untouched.

The policy catalogue and episode manifest store `input_patch`. Menu selection
resets physics/history/calibration and is blocked during recording. Native replay
reads the same flag. Runtime telemetry contains estimated side and added lift.
The Meta debug overlay continues to show raw input; GMR target overlay shows the
retargeted result.

## Reproduce the diagnostic comparison

Run from `prototypes/quest_g1` in the isolated project environment:

```sh
python experiments/virtual_leg_lift/run.py OUTPUT_DIR EPISODE_DIR
python experiments/virtual_leg_lift/analyze.py OUTPUT_DIR
python experiments/virtual_leg_lift/render.py OUTPUT_DIR
```

Native GMR is exported on both variants. After one second of GMR warmup, both
fresh physical rollouts use 50 Hz TWIST2 +500, 1000 Hz MuJoCo and the same camera
catch-up goal from the unmodified trajectory. No physical state is teacher-forced.
Both use the current APK's 14-point GMR, not the historical recording's separate
full-foot retargeter. The evaluated duration is 33.5 seconds. Source/GMR clips have
100 Hz samples; robot states/metrics have 50 Hz samples.

The report uses the unmodified GMR reference for both trajectory and upper-body
tracking metrics. Slip is computed at real sphere-sole contact points during the
original forward runs, weighted by normal load. Clearance uses the exact lowest
floor-colliding sphere, with the same source-estimated swing mask for both modes.
An upward/braking contact candidate is not a confirmed stumble.

The generated video has three rows (Meta input / GMR / physical robot) and two
columns (original / patched). This is offline animation, not Quest screen capture.
Results on one previously examined clip are diagnostic, not an independent test.
