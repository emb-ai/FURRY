# Data And Evaluation

[Home](Home.md) | [Common Ground](Common-Ground.md) | [Contributing](Contributing.md)

**Status: proposed requirements, not a file format or measured benchmark.**
Implementation belongs to [data #8](https://github.com/emb-ai/FURRY/issues/8) and
[evaluation #9](https://github.com/emb-ai/FURRY/issues/9).

## Keep The Streams Distinct

| Stream | Preserve | Do not confuse it with |
| --- | --- | --- |
| Source observations | Device poses, controllers/buttons or hand signals, source clock, sequence, validity and source convention. | Complete measured human motion. |
| Human estimates | SDK/completion/fusion outputs, method/version and observed/inferred/fused provenance. | Ground truth for the same estimator. |
| Robot references | Retargeted body/hand targets, joint manifest, frame and timestamp. | Achieved robot state. |
| Controller outputs | Policy output and actual applied actuator controls, gains/configuration and checkpoint identifier. | A desired pose recorded before filtering or limiting. |
| Simulation state | Root/body/hand state, simulator time, contacts, object/task state and relevant runtime state. | Placeholder zero arrays or rendered skeleton poses. |
| Visual streams | Camera identity, source frame time, encoding, intrinsics/extrinsics if relevant. | The time a recorder happened to copy an image buffer. |
| Session events | Start, pause, resume, recenter, calibration change, reset, abort and outcome annotation. | Unexplained gaps in timestamps. |

The episode manifest records schema/config version, task, input mode, optional
camera mode, participant pseudonym/consent reference, seed, code/model/policy
hashes, dependency versions, calibration IDs and clock mappings. Record unavailable
fields explicitly; do not manufacture confidence values that an SDK does not
provide. A zero hand array from upstream is missing information, not a relaxed
measured hand.

Retain raw timestamps and streams alongside any resampled/aligned product. Latest
values from different processes are not necessarily simultaneous. Store the
alignment rule, interpolation policy, maximum accepted age and clock uncertainty
as provenance of derived data.

## Recording And Replay

Choose storage through ADR-005 after testing throughput and interrupted writes.
Keep schema versioning explicit, but do not commit to HDF5, Zarr, LeRobot, ROS bags
or another wire/storage format in this foundation.

A collector must distinguish finalized, aborted and incomplete episodes. Never
silently discard failed trials. Quality filtering produces a traceable selection
over raw data, with reasons and segmentation boundaries; it must not overwrite
the originals or turn failures into successes.

Kinematic replay displays recorded poses. Dynamic replay re-executes a controller
or actions through physics. They answer different questions. Dynamic replay may
need controller history, integrator/runtime state, object state and resets in
addition to `qpos` and `qvel`; audit the chosen controller and
[MuJoCo state API](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)
before declaring a complete snapshot. Specify numerical tolerances and measure
divergence rather than promising cross-machine bitwise determinism.

## Benchmark Ladder

| Stage | Initial scenarios | Evidence |
| --- | --- | --- |
| M0 | Standing/default reference and bundled walking clips. | Pinned baseline, achieved timing, tracking error and failures. |
| M1 | Neutral pose, reach, crouch, turn/step; add walking after stable integration. | Both Quest input modes, session control, dropout behavior and replayable episodes. |
| M2 | Simple grasp/release, carry, crouch-to-pickup; selected ambiguous/dynamic leg motions. | Task success and physical feasibility, hand fidelity, matched sensing ablations. |

Test the two primary modes, controllers and optical hands, with camera-free
operation first. Add RGB-D-enabled cells when available. Report missing cells
and additional equipment rather than treating an incomplete matrix as a full
comparison. RGB-D fusion must not be evaluated against its own estimates as if
they were an independent reference.

## Metrics And Fair Comparisons

- **Human reconstruction:** joint/root/orientation error only against suitable independent or synthetic full-body references; report masking and observability assumptions.
- **Retargeting:** end-effector/relative-pose error, joint-limit violations, foot sliding and contact consistency before control.
- **Physical execution:** command-following error, falls, unintended contacts, object outcomes and body-hand coordination under a fixed simulator/controller configuration.
- **Systems:** capture-to-actuation and feedback latency distributions, clock uncertainty, effective rates, stale/drop fractions and recovery behavior.
- **Collection:** task-success rate, completion time, accepted demonstration yield, calibration/setup burden and operator interventions.

Fix controller/checkpoint, embodiment, scene, seeds and task conditions for a
sensing comparison. Controller adaptation is a separate ablation. Account for
lookahead and filtering latency. Split training and evaluation by participant,
session and motion family as appropriate; prevent neighboring frames or repeated
clips from leaking across the split. Declare trial counts, exclusions and success
criteria before collecting results. Show all trials, variability and failures.

## Required Adverse Scenarios

| Scenario | Required evidence |
| --- | --- |
| Tracking loss or occluded hands | Invalidity is visible; control and recording follow the declared dropout policy. |
| Headset recenter | A new calibration/frame epoch is recorded; no silent pose jump. |
| Stale, duplicate or reordered packets | Age/sequence checks and a reproducible transport test. |
| Camera loss | Explicit camera-free fallback or pause; fused estimates are not mislabeled. |
| Simulator reset | Reset event, new simulator-time epoch and known object/controller state. |
| Interrupted recording | Recoverable finalized data and an explicitly incomplete episode. |
| Replay divergence | A measured tolerance check with a diagnostic, not an unqualified replay-success label. |

## Privacy And Release

Do not put identifiable footage, participant names, headset identifiers, network
details or consent records in Git or public issues. Use approved private storage,
minimum necessary recording, pseudonymous manifests and an agreed retention
policy. Publish synthetic fixtures or explicitly cleared samples, with data and
model licenses checked separately from code. Do not upload any participant data
to a service merely to make an experiment convenient.
