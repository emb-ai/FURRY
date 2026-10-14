# Quest contact and Virtual Legs input work — 2026-10-12

Status: **no long runs launched**. The user changed the priority to aligning
training with Virtual Legs inference before the six A/B/C runs. The queue and
trainer reject long runs without a checked input contract. This package is an
experimental implementation, not an accepted gait policy.

## Implemented and checked

- Nonlooping completion uses the remaining clip duration after RSI; safety
  failure takes precedence over simultaneous natural end. Source PPO bootstrap
  remains unchanged. Natural end and time limit use the existing timeout mask;
  safety failure does not. Failed completion is capped at 0.5 and non-natural
  completion at 0.949 before the existing curriculum calculation. These are
  explicit adapter choices, not upstream TWIST2 behavior.
- Contact sliding uses all four unchanged sole spheres and angular velocity.
  Loaded per-foot COM offsets determine the velocity origin. Isaac net foot
  load is shared between near-plane sphere points; individual contact forces
  are unavailable in this approximation. The reward is load-normalized and
  bounded. Its support-loss term requires a valid reference support mask.
- Clearance is a bounded two-sided error, masked when reference phase is
  unavailable or shortly after a source push. The phase construction is a
  heuristic on authored TRAIN motions, **not measured human contact labels**.
  Quest phase is untrusted. Condition C is not ready for long training.
- Native command exports preserve `MetaRetargeter::Solve` output before offline
  floor correction/filtering. They support the historical 14-marker mode, with
  explicit camera-tracking selection. They do not implement full-foot,
  application startup blending, or closed-loop catch-up.
- A separate command bank replaces only the actor's 35 command values. Source
  reward/termination references and privileged critic observations are kept
  separately. This separation prevents accidental target corruption; it does
  **not** establish that the old filtered targets are correct human leg motion.
- Command lookup is causal zero-order hold at original body-event times.
  Every original Quest TRAIN clip and its paired mirror is required. Author
  TRAIN motions retain their source commands. Validation is rejected by the
  loader. Dataset membership and file/metadata hashes are checked.
- Actor history now starts at zero after reset and includes only preceding
  actor observations; the final 35 values duplicate the current command.
  Native C++ history/tail parity: 14 inference ticks, two resets, max error 0.
  This check does not claim Isaac/MuJoCo dynamics or proprioception parity.
- Full evaluation traces contain the actual held actor input/history, a policy
  tick flag, pre-input/post-state times, physical/reference poses, contacts and
  braking impulse. A short CPU smoke produced identical states and observations
  with and without the recorder (20 steps, 70 contact rows, max error 0).
- Twelve CPU unit tests passed, including pinned source termination/curriculum,
  MuJoCo geometry/Jacobian velocity, causal command lookup, validation exclusion,
  reset history and unchanged critic/reward references.

## Available data and current limitations

Two reconstructed banks cover 66 Quest TRAIN clips (33 originals + 33 mirrors)
within the unchanged 1,090-motion mixture. Camera tracking is OFF in one bank
and ON in the other. Authored clean motions still dominate the source sampling
mass; these banks do not create a Virtual Legs error distribution for them.

The original human-only capture has a zero-filled `mimic.csv`. Exporting raw
Meta through native GMR reconstructs retargeter commands, not a recorded policy
consumption schedule. The native solver was fed body updates continuously; the
original renderer/policy scheduling and application blend are not reconstructed.
Previous filtering/curation exclusions are also not restored as observed faults.

For the camera-OFF reconstruction, original TRAIN leg-joint differences from
the filtered target are mean absolute 0.00554 rad, p95 0.02367 rad, maximum
0.18578 rad. These differences alone do not establish the cause of falls.
The camera-ON reconstruction has p95 0.02448 rad and maximum 0.18954 rad.
Neither comparison used validation or counted mirrors as independent data.

The existing training capture stores 14 Meta markers. Recent diagnostic live
records use `full_foot_retarget=true`, which needs additional raw full-body
joints. Those joints cannot be recovered from the historical training capture.
Diagnostic live records have not been reassigned to TRAIN. The target runtime
choice (new full-foot capture versus historical 14-marker training) is pending
the user's answer.

Catch-up must be recomputed from each candidate robot's current camera/root
state. A correction recorded with another policy is not a valid substitute.
Its velocity and displacement reward goals also need to match the applied
command. The new input bank does not yet implement this closed loop.

No paired measured human feet are available, so delay/freeze amplitudes from
older experiments must not be called measured Virtual Legs errors. Reliable
intent for Kick-T is not supplied by this reconstruction.

## GPU evidence and remaining work

Before the input change, the real 4,096-env B smoke finished two critic updates
and three actor updates on GPU UUID
`GPU-68d1fffd-2ea2-3f95-f100-38f39855bcdf`. A separate PhysX probe identified
COM-referenced linear velocity (RMS 0.000123 m/s versus 0.131733 m/s under the
link-origin assumption). These earlier remote results do not validate the new
command-bank adapter. After a lost SSH session, three reconnection attempts
timed out during banner exchange; the new GPU smoke has **not run**.

Required before long runs:

1. Resolve the target runtime mode and supply missing full-body TRAIN capture
   if full-foot remains the target. Keep diagnostic/test recordings separate.
2. Match scheduler, startup/reset behavior, camera-tracking configuration and
   catch-up as applicable; agree reward goals for that input.
3. Validate actual actor/proprio assembly and run a 20-update Isaac GPU smoke
   with this bank. Check finite losses, actual Quest-command activation and
   unchanged privileged/reward references.
4. Reassess the common A/B/C input contract and phase coverage. Freeze input,
   source, data, geometry and config hashes only after these checks.
5. Run and evaluate the six 5,000-update jobs only then. Turn/recovery tasks,
   closed-loop catch-up evaluation, event/video review and D/E/F remain pending.

## Dependencies and reproduction

The bundled Python files include the actual local dependency closure, including
`reference_faults.py`; unused copied launchers were removed. Upstream TWIST2 /
Isaac Gym remain pinned external dependencies, not silently vendored replacements.
The source revision is `b06178f19a22f2138cbd31f60c6d494bc263f67d`.
`g1_sim` is supplied by the pinned prototype snapshot or the existing remote
root; include that directory on `PYTHONPATH` when running CPU evaluation.
The native exporter requires the explicitly selected native source/assets
snapshot, MuJoCo and ONNX Runtime. `build_native_export.py` records source,
header, asset and library hashes. The build helper supports macOS and Linux.

Use the existing isolated uv environments on the shared server. Do not install
system packages, change drivers, or touch another user's processes. One GPU per
independent run; at most three H100 GPUs including smoke/evaluation.

Example CPU tests (the source path must refer to the pinned upstream checkout):

```sh
export PYTHONPATH="$FURRY_CODE:$FURRY_ROOT:$PYTHONPATH"
export TWIST_SOURCE="$FURRY_ROOT/TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py"
uv run --no-project --python "$FURRY_ROOT/eval-env/bin/python" \
  -m unittest test_contact_cycle test_native_command_bank -v
```

`build_command_bank.py --help` exposes explicit dataset, continuous capture
manifest, raw episode, native export receipt, model and mode arguments. Outputs
use fresh paths and are intentionally marked `long_run_ready=false`.
`audit_command_bank.py` compares only original Quest TRAIN clips with their
unchanged filtered targets. `audit_native_history.py` instruments a copy of
native `Simulation::Step`, never the runtime source in place.

The remote smoke overlay is generated alongside the local evidence. Merge it
into the existing pilot config and use `cycle_runner.py --pilot B --updates 20`
with an explicitly checked free GPU. That is a smoke, not critic readiness or
permission to start the long queue. Full-fidelity physical/RNG continuation is
not implemented: resume restores learning/curriculum state but resets physics
and history, as explicitly reported by `resume_state.py`.

## Camera-leg clips (2026-10-13)

Goal: fine-tune on real leg motion (push-T kicks/pushes) from the camera
skeleton. Actor input is the recorded live `mimic.csv` (camera fusion as run on
the headset); the reward target is the offline-cleaned camera legs
(`prototypes/quest_g1/training/camera_targets.py`, clean3) re-solved by native GMR.

1. `prototypes/quest_g1/training/camera_dataset.py OUT --assets android/assets
   --pair EPISODE_DIR:PREFIX ... --validation EPISODE_ID --mirror` writes
   quest-g1-motion-v1 clips (one stage per episode, from 101) and
   `command_bank.npz` (quest-camera-command-bank-v1, mirrors included).
Published inputs (Git LFS; `git lfs pull --include="datasets/**"`):
- `datasets/quest-camera-legs/2026-10-13/dataset-v1`: step 1 output with its
  camera command bank.
- `datasets/quest-virtual-legs-v2/2026-10-08/native-bank-camera-off`: native
  command bank for the night-v3 Meta TRAIN clips (camera tracking OFF, as in
  the observer-view camera episodes), rebuilt on Linux (2026-10-14).

2. `camera_mixture.py combine QUEST night-v3 dataset-v1` merges the camera
   clips with the Meta Quest clips. Files are byte copies; stages must not collide.
3. `author_dataset.py --quest QUEST ...` builds the mixture as before.
   Run `build_phase.py` on the new mixture; Quest phase stays masked.
4. `camera_mixture.py bank --dataset MIXTURE --banks native-bank-camera-off/bank.npz
   dataset-v1/command_bank.npz --output BANK.npz` writes one
   quest-native-command-bank-v1 covering every Quest TRAIN clip. Both banks
   match by original NPZ hash, so no old mixture is needed. It is loaded by the
   unchanged `native_command_bank.py` (`--actor-command-bank`).

Rebuilding the native bank (no macOS needed):
`meta_epochs.py EPISODE OUT` recreates prepare.py's calibration-epoch sources
and run origins from the raw night-v3 episode (origins and exclusions match the
night-v3 manifest exactly); `build_native_export.py` compiles the exporter on
macOS or Linux; `export_native_commands` runs per epoch (camera tracking 0);
`build_command_bank.py --dataset VIEW` uses a view of night-v3 with
`source=quest`/`npz_file` added (`night-v3-quest.manifest.json`). The native
source in main has no full-foot mode, so the exporter no longer selects it.
`audit_command_bank.py` reproduces the original camera-OFF bank statistics to
every printed digit (legs mean 0.00554, p95 0.02367, max 0.18578 rad).

Checked locally on dataset-v1 (5 episodes, 187 s train + 187 s mirrors, 89 s
validation): the unchanged author_dataset accepts the combined Quest set, bank
samples equal live `mimic.csv` exactly on all 40 camera clips, and `bank`
merges the published native and camera banks over all 106 Quest TRAIN clips
(1426 s Meta + 374 s camera, no unused entries). No GPU smoke has run.
Family weighting is unchanged: camera families get about 38% of the Quest mass,
i.e. roughly 1.4% of motion draws before curriculum.
