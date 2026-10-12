# HOMIE v2 / SONIC VR3 desktop simulation comparison

This experiment adds `g1_sim.autonomous_legs.Homie` and `.Sonic` to the desktop
MuJoCo simulator, with a common clip runner. It does not modify the Quest APK.

- HOMIE v2: official Balance/Walk ONNX, 15 lower/waist outputs and direct 14 arm
  targets. Port matches the existing native runtime on a 100-step rollout.
- SONIC: official default encoder/decoder, encoder **teleop mode 1**, local
  left-wrist/right-wrist/head targets and the **released neural planner** for
  leg references. Recorded leg targets are never supplied to its encoder.
  `check_contracts.py` tests this independence.
- TWIST2 comparison: `quest-111586-500`, explicitly 50 Hz.

Upstream source: NVIDIA GR00T-WholeBodyControl
`b042411fae38ee4d1af9aac82a37a1f8d14d6dd0` (Apache-2.0). Weights:
<https://huggingface.co/nvidia/GEAR-SONIC> (NVIDIA Open Model License).
HOMIE weight/license provenance is in `policies/homie-v2`.
SONIC constants are exported from the pinned `policy_parameters.hpp`;
`sonic_motor_profile.txt` comes from its deployed `g1_29dof_old.xml`.
Source/model/data hashes and environment versions are in the result receipt.

## Interface and adaptations

`update(data, command35, reference_qpos36)` is called at 50 Hz, then
`apply(data)` at 1000 Hz. Keep one model per controller: author motor inertia,
friction, ranges, force limits and PD differ. Call `initialize` after construction
so `mj_setConst` refreshes derived dynamics. The robot geometry, floor and
sphere soles are shared with the FURRY native scene.

All inputs are robot-scaled GMR references. SONIC's three targets are obtained
from the same reference FK with the author's wrist/head marker offsets. These
are not new raw human measurements. Its upper body is constrained by three
points, so elbow/waist joint tracking is diagnostic, not a directly commanded
quantity. HOMIE retains the fuller upper-body command.

The SONIC adapter replans synchronously at 5 Hz, carrying forward four generated
30 Hz frames as context. It interpolates quaternions, samples ten generated
leg reference frames 100 ms apart, and passes world movement/facing commands.
Facing integrates the corrected yaw command over a 0.3 s lookahead. It retains
all 11 official planner inputs and their default token mask. This is a new
synchronous navigation adapter, not timing parity with upstream asynchronous
VR deployment. There is no library-of-recorded-steps substitution.

Dataset runs derive feedforward velocity from the current and next 20 ms GMR
pose. Root catch-up uses current simulated pose, caps correction at 0.4 m/s and
total speed at 0.8 m/s; yaw correction uses gain 3, cap 1 rad/s and total cap
1.2 rad/s. Live exports use recorded source-only feedforward and the native
camera catch-up v2 formula with the current rollout camera. Reset is fresh,
aligned once at the beginning, with a shared 0.5 s activation blend. No saved
robot states are teacher-forced into an evaluation rollout.

## Running

Use the project's isolated Python environment with NumPy, SciPy, MuJoCo,
ONNX Runtime, Matplotlib and Pillow. Models are external inputs; never commit
LFS pointers as if they were usable weights. `--models` contains the default
`model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx` and matching
`observation_config.yaml` from the official release. `--assets` points at
prepared FURRY Android assets including the HOMIE models and motor profile.

From `prototypes/quest_g1`:

```sh
.venv/bin/python experiments/homie_sonic/evaluate.py \
  --backend sonic --models /path/to/sonic-models \
  --dataset /path/to/dataset/validation --output /path/to/results
```

Use `--backend homie` or `--backend twist2`; for the reported baseline add
`--twist-policy policy-quest-111586-500.onnx`. `--seed` controls SONIC's planner
seed, not trained checkpoint initialization. `--seconds` limits diagnostic
probes. Each run saves qpos/qvel, applied ctrl, references, commands, per-frame
metrics and a summary. `prepare_live.py` converts native GMR source CSVs to the
same input format and preserves camera goals. `report.py` produces the paired
report and plots; `render.py` produces the three-panel offline replay.

## Interpretation

All clips are compared at identical lengths. Fall and numerical failure stop a
rollout and are explicit failures; do not compare aggregate surviving-prefix
errors if future runs have unequal lengths. The current set is repeatedly used
diagnostic validation (12 clips), plus two recorded live sequences, not an
independent acceptance test. Contact events are heuristic candidates. In
particular, braking touchdown is often a normal step, and zero upward-contact
candidates does not establish zero stumbles. See REPORT.md for exact thresholds.
