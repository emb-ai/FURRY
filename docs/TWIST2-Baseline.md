# TWIST2 Baseline

[Home](Home.md) | [Literature](Literature-Review.md) | [Architecture](Architecture-and-Decisions.md)

**Status: inspected, not executed.** Audit date: 2026-10-05. Pin:
`b06178f19a22f2138cbd31f60c6d494bc263f67d` in
[amazon-far/TWIST2](https://github.com/amazon-far/TWIST2/tree/b06178f19a22f2138cbd31f60c6d494bc263f67d).
This is a reproducibility starting point, not a claim that the released stack
works on our machine. Track execution under the [baseline umbrella](https://github.com/emb-ai/FURRY/issues/2).

## Paper Versus Release

The paper's sensing setup uses PICO 4 Ultra and two calf trackers. Its Dex3-1
hand interface interpolates grasp presets from controller input rather than
imitating all human finger joints. These are useful baselines, but differ from
FURRY's Quest optical-hand and no-worn-tracker targets.
[Paper, Sections III-C and III-D](https://arxiv.org/html/2511.02832v1)

| Inspected release artifact | What it establishes | What it does not establish |
| --- | --- | --- |
| `assets/ckpts/twist2_1017_20k.onnx` and `twist2_1017_25k.onnx` | Two supplied policy files; `sim2sim.sh` selects the 20k file. | Compatibility or achieved performance in our environment. |
| `assets/example_motions/0807_yanjie_walk_001.pkl` through `_010.pkl` | Ten bundled walking clips are present. | Released task scenes for every manipulation demonstration. |
| `sim2sim.sh` | Selects `g1_sim2sim_29dof.xml`, CUDA and 100 Hz policy configuration. | A measured 100 Hz end-to-end system. |
| `server_low_level_g1_sim.py` | 29 body actions, 35 reference values, history-based policy input, PD torques and MuJoCo stepping. | A complete dexterous-hand simulation loop. |
| `g1_sim2sim_29dof_with_hands.xml` | A hand-equipped model is supplied. | That swapping the XML preserves checkpoint ordering or adds hand control. |
| `server_data_record.py` | Records visual/state/action streams through the upstream collector. | Atomic sensor synchronization or a complete reproducible simulator snapshot. |

Sources: [asset tree](https://github.com/amazon-far/TWIST2/tree/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets),
[launch script](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/sim2sim.sh),
[simulation loop](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/deploy_real/server_low_level_g1_sim.py),
[collector](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/deploy_real/server_data_record.py).

## Hand Integration Gap

The simulation loop reads hand commands but applies a 29-element body torque
vector and publishes zero-filled hand states. These placeholders must not become
valid training observations. Before using the hand-equipped model, map joints
and actuators by name, preserve the body's policy contract, add hand control,
and log actual achieved hand state. Test collision/contact behavior as well as
joint motion. A model file alone does not establish grasping capability.

## Runtime And Portability

The [README](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/README.md)
describes two Conda environments and external GMR/PICO dependencies. The
[offline reference server](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/deploy_real/server_motion_lib.py)
imports Isaac Gym, uses CUDA inside `build_mimic_obs`, and sets a 0.02-second
reference interval. The low-level loop exposes `--device cpu`, but that does not
make the reference server CPU-compatible. Audit both processes independently.

The simulator uses a 0.001-second physics timestep and a configurable policy
interval. The script requests 100 Hz while README discussion mentions 50 Hz.
Record which configuration matches the checkpoint, then measure actual rates;
do not silently pick one value as an established performance result.

## Proposed Reproduction Sequence

1. Record host, drivers, environment, source revision and asset/checkpoint hashes.
2. Inspect the motion loader before opening `.pkl` files; only load trusted,
   pinned artifacts. Pick the first bundled walking clip rather than a missing
   external dataset path.
3. Audit reference layout, quaternion conversion, joint ordering and the
   checkpoint input/output sizes. Verify the simulator's initial pose and gains.
4. Bring up a loopback reference source and simulator. Capture a standing/default
   reference interval before the walking trial; do not assume startup scripts
   provide a complete automated session.
5. Record actual control timing, tracking behavior, falls, and reset behavior.
   Repeat with the remaining supplied walking clips.
6. Only after the body baseline is understood, integrate the hand model and
   document any body-control regression separately.

This sequence is **unverified runtime guidance**, not a tested runbook. Publish
the commands that actually work in the M0 report rather than presenting guessed
install commands as ready to run. No policy training is required merely to
inspect the supplied checkpoint.

## Reuse Boundaries

Keep upstream software separate and record patches. FURRY retains Apache-2.0;
TWIST2's root license is MIT, but dependency, model and motion-data terms need
their own provenance checks. Do not redistribute external datasets simply
because code is public. Review available high-level learning code separately
from teleoperation; the upstream README describes that release separately.

Never copy instructions that delete existing environments, expose Redis without
protection, disable a firewall, or launch robot-hardware control. They are not
needed for a simulation-only foundation.
