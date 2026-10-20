# Stance narrowing A/B — 2026-10-13

Authorized experiment: additional 1,000 PPO updates per run, paired seeds 11/22,
4 independent single-H100 runs. Checkpoint indices 750/1000/1250/1500 correspond
to +250/+500/+750/+1000 after the user-selected quest-111586-500.

## Exact intervention and data

Only B shifts foot targets by ±0.06 m along the target pelvis yaw lateral axis,
immediately before `gmr.SetTargets(targets)`. The exact block is copied from
`../stance-width-audit-20261013/meta_retarget.cpp`. Native adapter is pinned to
3d0164f, the version used to create the initial checkpoint's Quest references.
No later wrist/full-foot/camera fixes are mixed into this controlled experiment.
Raw Meta and bind poses, knee targets, target Z/quaternions and GMR weights are
unchanged. Common historical smoothing and sole-floor preprocessing is retained.
The solved pose may change height/orientation through IK; those are outcomes,
not additional edits of targets.

Both variants were rerun continuously from original calibration events in
`../dataset-audit-20261008/episode/episode-1791467626074995` through adapter + GMR.
A reproduced all original qpos to <1e-10 and regenerated all mirror arrays exactly;
its old reference files are retained byte-for-byte. Same source cuts: 33 train
originals +33 train mirrors, 12 validation clips/stages 2/9/14. No clip removed
based on rollout success. B has recomputed full FK, root-oriented PKL body fields,
and velocity diagnostics. Original MotionLib recomputes its standard velocities
from each new PKL at load time. Derivative sidecars are diagnostics, not new inputs.

All 1,024 author non-PICO motions, ordering and weights are byte-identical. Quest
sampling mass remains 730/20083 before source curriculum. This is the existing
single-session validation, not a new independent test set.

## Initialization / unchanged training

Source checkpoint SHA256:
`13e93d837b014317ca1179303bc2b580c1a7940f2a5d868dae20bd36774d3a21`.
Published 50 Hz ONNX SHA256:
`57d6ac4b2461ac8884afeac86d32bcc9a7b4b72f376e15b898af7ac3d8a5241d`.
1024-sample ONNX/Torch parity max error 3.3378601e-6 (`initial/parity.json`).

Restore actor, critic, exploration std, Adam and both normalization transforms.
Recover learning rate and PPO algorithm counter from the checkpoint and its
original metrics. Noise/DR clocks were neither saved nor logged in111586 and
retain fresh source-environment initialization in both branches. Physical simulator/history/RNG reset; legacy motion curriculum
was not saved and starts over in both branches. This is NOT an exact resume.
No critic warmup is added when restoring the existing critic.

Pinned TWIST2 b06178f DaggerPPO, original adaptive KL learning rate (not a stop),
4096 environments, 50 Hz actions/history, 500 Hz training physics. Preserve the
existing one-second displacement/yaw reward, 1743-dimensional critic, sphere-sole
URDF and all source PPO/DR settings of111586. No new reward, reference adapter
network, low-clearance cost, leg faults or nonlooping change. No validation or
whole-run KL early stop. Nonfinite states/failure halt the affected job and new
scheduling; a failed job is not silently retried or declared complete.

## Evaluation

Initial actor and each saved checkpoint run against both A/B references, 12 held-out
clips × evaluation seeds0/1. All use ONNX Runtime CPU and 50 Hz actor/history,
1000 Hz MuJoCo. Candidate exports are private and checked against Torch.
Both reference modes start at the SAME original A physical qpos/qvel with identical
seed noise and zero actor history. Commands are naturally derived from each
reference; no catch-up servo is added. Original A trajectory remains the common
XY/yaw goal. Store all per-tick traces.

Report falls and executed/requested duration first. Error/slip/step comparisons
use a common survival prefix across completed conditions, after1s warmup; a
separate cohort contains only trials completed by every condition. No shorter
failed rollout qualifies as an improvement. XY in cm, yaw in degrees, contact
slip and slide rate, root path and pelvis-relative horizontal step amplitude
(p95-p05 in full1s windows) are retained. Raw gait diagnostics remain available.

## Server and commands

Server: andrei_gpu.
Shared root: `/home/jovyan/vlad/FURRY/twist2-distributed-20261010`.
Experiment: `<shared-root>/stance-finetune-20261013`.
`source <shared-root>/remote_environment.sh` before the saved commands.

- Exact launcher: `source/launch_remote.py`; setup checks all data/package hashes.
- Baseline commands: server `baseline-{A,B}.command.json`.
- Queue: server `env/bin/python <experiment>/code/stance_queue.py` (single launch;
  existing queue-start receipt prevents accidental duplicate scheduling).
- Worker commands/PID/device receipts: `<experiment>/*.launch.json`.
- Exact PPO/eval/export commands: `<experiment>/runs/<label>/*.command.json`.
- State: `<experiment>/queue-state.json`, `runs/<label>/status.json`.
- Metrics/checkpoints: `runs/{A,B}-s{11,22}/training/`.
- Evaluation: `evaluation/`, aggregate `summary.json` when complete.

Finite queue: baseline → A/B resume smokes →4 long runs. Smokes verify bitwise
restoration and all20 original PPO minibatch updates, reward/PPO/control equality.
Allowed GPUs0..4, at most4 active jobs. Before each launch require memory<100 MiB,
utilization0 and no compute PID, then recheck in the child. No system-level atomic
reservation is available. Foreign jobs are never signalled. Scheduling horizon24h;
`STOP_SCHEDULING` prevents further starts while existing jobs finish.

No APK installation, default runtime switch or publication is part of this run.

## Verified launch status

Baseline finished: A24/24 and B24/24 completed, zero falls,348.44s each.
Queue PID3372450 is confirmed alive, phase `waiting_free_gpu`; GPU0..4 remain
occupied by other training. No GPU smoke or new PPO update has run yet.
After a device frees, the queue performs both smoke gates before long training.
`current-status.json` and `remote-final/` hold dated evidence; do not treat this
snapshot as a permanent live status.

Before starting GPU work, NumPy2→NumPy1.23.5 PKL serialization compatibility
was fixed in the isolated B dataset. All78 payloads were independently reloaded
and every array compared exactly; no numeric data changed. Final server file
hashes are in `remote-final/package-manifest.json`; conversion evidence is in
`pickle-compatibility.json`. Only the first empty queue was replaced; baseline
and other training processes were preserved.

For a fresh reproduction, upload `package.tar.gz` as
`stance-finetune-20261013-package.tar.gz` and `execution-final.tar.gz` as
`stance-finetune-20261013-execution.tar.gz` under the shared root, then execute
`source/launch_remote.py` under its existing training environment. It validates
inputs, canonicalizes the new PKLs using training NumPy, checks identical arrays,
sets up the isolated roots, and starts baseline + finite queue. Existing experiment
directories are deliberately rejected. Artifact hashes: `reproduce-transfer.json`.
The actually running queue received the same serialization conversion separately;
its exact final deployment manifests are retained in `remote-final/`.

The real-model CPU resume test passed: actor/critic/std/Adam/normalizations restore
bitwise; the next complete original PPO update matches an independent direct
restore bitwise. PPO counter1050 and LR3.375e-5 recovered. Environment noise/DR
clocks are explicitly reset, not reconstructed; legacy111586 did not log them.
This compatibility change touches only restore-state handling, not PPO equations,
reward, observation dimensions or frequency. GPU rollout smoke remains pending.
