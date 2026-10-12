# SONIC VR3 on Quest

Default packaged mode `sonic-vr3`: encoder (1762 → 64), decoder (994 → 29),
neural planner (4 context frames → up to 64 poses). Models are Git LFS; use
`git lfs pull` before building. Exact source and model revisions/hashes are in
`provenance.json`. Original source/model licenses and modification notice are included.

The same GMR targets used for the desktop comparison supply robot-scaled local
wrist/head poses. Encoder teleop mode is 1. Incoming Meta/GMR leg joint targets do
not control SONIC's legs; the neural planner supplies ten leg references at 100 ms
spacing. Catch-up remains selectable and supplies planar velocity/yaw commands.
This is not a direct raw HMD/controller-only retargeter: the current app's Meta/GMR
calibration and tracking requirements still apply.

Native C++ executes encoder/decoder at 50 Hz and motor PD/physics at 1000 Hz, with
the author joint ordering, scales and motor profile. Quest requests a plan every
200 ms on a separate CPU thread; seed is 11. The initial plan is synchronous.
Completed trajectories retain their request simulation time, so latency advances
the read position instead of playing stale first frames. One request may be in
flight; reset drains/discards it. Exhausting a trajectory raises a runtime fault
instead of silently freezing the leg reference. This asynchronous schedule differs
from the synchronous desktop benchmark. Default native offline checks use the
synchronous schedule for reproducibility.

Select **Политика → SONIC: автономные ноги** (last page), then calibrate with A.
HOMIE is the adjacent autonomous mode. Baseline TWIST2 remains the default.
Selections persist; switching resets physics/history/calibration and is blocked
while recording. Loading validates all models and pinned motor parameters before
replacing the active mode.

Recordings preserve backend, three model identities and copies, motor profile,
seed/schedule, full raw 84-joint Meta data and actual simulated states. Planner
latency and plan age are in runtime telemetry. Asynchronous internal plan history
is not captured: state playback is supported, exact dynamic restart is not.
Copying the planner adds about 774 MB to each SONIC recording.

Build: `G1_CAPTURE_APP=1 android/build_apk.sh` produces an update to the separate
`org.g1lab.quest.capture` app. No device install is part of building.

Checks: `python scripts/check_sonic.py OUTPUT_DIR` compares 100 native updates
against the evaluated Python implementation, tests leg-input invariance, loading
failure, reset, switching TWIST2/HOMIE/SONIC, stand/cup/push-T scenes, recording
model identities, and asynchronous reset. `python -m unittest discover -s tests`
covers menu, packaging, capture and existing runtime contracts.

Android ARM64 compilation and desktop checks do not establish Quest performance.
The planner is substantially larger than TWIST2/HOMIE; latency, memory and live
behavior on the headset remain unvalidated.
