# Common Ground

[Home](Home.md) | [Architecture](Architecture-and-Decisions.md) | [Data and evaluation](Data-and-Evaluation.md)

## Five Different Problems

| Term | Meaning in FURRY | Frequent mistake |
| --- | --- | --- |
| Pose estimation | Estimate human/device state from incomplete observations. | Calling every SDK joint directly measured. |
| Inverse kinematics (IK) | Solve joint configurations for selected geometric goals. | Assuming an IK solution is dynamically executable. |
| Retargeting | Map motion across different bodies, limits, proportions and contacts. | Copying human joint angles into a robot by index. |
| Motion tracking control | Use feedback to make simulated robot dynamics follow a reference. | Confusing target joint positions with achieved motion. |
| Trajectory collection | Preserve observations, decisions, execution and outcomes over time. | Saving only a pretty skeleton animation. |

The chain is observation -> estimate -> robot reference -> controller action ->
achieved robot state. Each arrow can introduce error, delay or a modeling
assumption. Evaluate them separately before assigning credit to an algorithm.

## Sparse Observations And Ambiguity

Multiple full-body motions can produce similar head and hand trajectories.
Priors can choose plausible completions, but cannot guarantee recovery of an
unobserved foot motion. Meta describes generated legs as plausible estimates
from upper-body signals. Keep their provenance distinct from tracked poses.
[Meta's explanation](https://developers.meta.com/vr/blog/inside-out-body-tracking-and-generative-legs/)

Two different objectives are legitimate: reconstruct the operator's motion, or
infer an executable robot behavior that achieves the operator's intent. Choose
one explicitly per experiment. Task success alone cannot establish human-pose
accuracy; a pose error alone does not describe manipulation success.

## Proposed Shared Conventions

These conventions guide FURRY interfaces and tests; they are not an implemented
wire format. Record source conventions before conversion and preserve raw data.

| Quantity | Convention |
| --- | --- |
| Length, time, angles | Meters, seconds, radians; name any stored integer timestamp unit. |
| Canonical world | Right-handed, Z-up, aligned to the calibrated MuJoCo floor; document initial heading. |
| Frames | Name `quest_stage`, `human_root`, `sim_world`, `robot_base`, and each camera frame; never use ambiguous `world`. |
| Transform | `T_A_B` maps coordinates in frame B into frame A: `p_A = T_A_B p_B`. Compose `T_A_C = T_A_B T_B_C`. |
| Quaternion | Canonical `wxyz`, unit norm; declare active/passive rotation and source order at each adapter. Treat `q` and `-q` as the same orientation. |
| Joints | Use ordered joint-name manifests with model hashes; do not assume skeleton, policy, `qpos`, `qvel` and actuator indices coincide. |
| Missing data | Explicit validity and provenance, not zeros or an unmarked held-last value. |

MuJoCo free joints have seven position coordinates but six velocity coordinates.
Quaternion derivatives are not interchangeable with angular velocity. Use the
model's named/indexed access and quaternion operations rather than arithmetic
on presumed joint-array layouts. [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)

Do not hard-code a Quest-to-MuJoCo axis permutation without checking the chosen
SDK frame convention. Handedness changes require a basis transformation, not
just reordering four quaternion components.

## Calibration And Clocks

Calibration records include floor and heading alignment, human scale, sensor
extrinsics where applicable, procedure/version and residual error. Recenter is
a new frame epoch: emit an event, update transforms and interrupt or segment the
episode. Do not let a coordinate jump masquerade as human acceleration.

Retain device capture time, desktop receive time, sequence ID and clock-domain
ID. Use monotonic clocks for intervals. Comparing two devices requires an
estimated offset/drift mapping and uncertainty; wall-clock UTC is useful for
session metadata, not a substitute for synchronization. Keep simulator time
separate, because it can pause, reset or run slower than real time.

Sampling rate, controller rate, physics rate and render rate are different.
End-to-end latency includes capture, buffering, transport, estimation, control
and feedback. A 100 Hz process does not prove 10 ms total latency. Any future
reference lookahead consumes an explicit prediction or buffering budget.

## Minimum Adapter Tests

- Identity, translation and 90-degree rotations about each axis; round-trip transforms.
- Left/right hand and foot mapping, neutral standing height and floor contact.
- Quaternion order, normalization, sign equivalence and shortest-path interpolation.
- Joint-name mapping against the pinned model and policy manifests.
- Recenter, dropout, stale/reordered samples, clock drift and simulator reset.

Use synthetic fixtures first. Live headset results supplement these tests; they
do not replace deterministic convention checks.
