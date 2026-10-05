# FURRY Knowledge Base

Research foundation, reviewed **2026-10-05**. Start with this page, then
[Common Ground](Common-Ground.md) and the [TWIST2 audit](TWIST2-Baseline.md).

## Research Question

How much useful, faithful, physically executed humanoid behavior can we collect
from sparse Quest 3 observations, and when does one external RGB-D view justify
its calibration and setup cost?

The target is **MuJoCo simulation only**, first G1 with the bundled Dex3-1 hands.
Both Quest controllers and optical hands are first-class input modes. Other
humanoids are possible later, not prerequisites. No extra worn body trackers are
required for the target system, and no physical-robot deployment is planned.

Whole-body motion and object interaction are both goals. We first reproduce what
the released TWIST2 code actually provides, then extend it. A plausible animated
body is not evidence that unobserved human leg motion was recovered or that a
robot can execute the reference under contact and balance constraints.

## Evidence Vocabulary

| Label | Meaning |
| --- | --- |
| Author-reported | A cited paper or project makes the claim; we have not reproduced it. |
| Inspected | A specific source revision supports the statement; execution is unverified. |
| Proposed | A FURRY hypothesis, contract, or experiment awaiting implementation and validation. |
| Locally reproduced | Linked commands, environment, inputs and results support a repeatable local run. |

This foundation includes source inspection and documentation-tool tests only.
**There are no locally reproduced robotics results yet.** Promote claims between
these categories only when the corresponding evidence is linked.

## Milestones And Workstreams

The [program roadmap](https://github.com/emb-ai/FURRY/issues/1) links the eight
workstreams: baseline simulation, Quest acquisition, desktop/VR runtime, sparse
motion reconstruction, dexterous hands, optional RGB-D, trajectories, and
evaluation. Umbrellas coordinate work; students create smaller linked tasks.

| Workstream | Umbrella |
| --- | --- |
| Released TWIST2 baseline | [#2](https://github.com/emb-ai/FURRY/issues/2) |
| Quest acquisition | [#3](https://github.com/emb-ai/FURRY/issues/3) |
| Desktop runtime and VR feedback | [#4](https://github.com/emb-ai/FURRY/issues/4) |
| Sparse reconstruction and retargeting | [#5](https://github.com/emb-ai/FURRY/issues/5) |
| G1 dexterous hands and interaction | [#6](https://github.com/emb-ai/FURRY/issues/6) |
| Optional RealSense fusion | [#7](https://github.com/emb-ai/FURRY/issues/7) |
| Recording and replay | [#8](https://github.com/emb-ai/FURRY/issues/8) |
| Benchmarking and evidence | [#9](https://github.com/emb-ai/FURRY/issues/9) |

| Milestone | Exit evidence |
| --- | --- |
| [M0: Released TWIST2 Baseline](https://github.com/emb-ai/FURRY/milestone/1) | Pinned standing/walking reproduction, environment and joint/tensor mapping, measured timing, documented gaps. |
| [M1: Quest-to-G1 Collection](https://github.com/emb-ai/FURRY/milestone/2) | Both input modes, calibrated references, closed-loop simulation, VR feedback and inspectable/replayable episodes. |
| [M2: Dexterous Loco-Manipulation And Sensing Comparisons](https://github.com/emb-ai/FURRY/milestone/3) | Actual hand simulation and object tasks; matched input-mode and optional RGB-D comparisons. |

Optional hardware must not block M0 or the camera-free M1 path. Benchmark design
starts during M0; the final comparison belongs to M2. Owners and dates remain
unset until participants agree to a bounded task.

## Reading Routes

| Page | What it answers |
| --- | --- |
| [Getting Started](Getting-Started.md) | What should I install now, and what should I leave for my workstream? |
| [Literature Review](Literature-Review.md) | What ideas, baselines and implementations should we compare? |
| [TWIST2 Baseline](TWIST2-Baseline.md) | What is actually released, and what still needs reproduction or extension? |
| [Architecture And Decisions](Architecture-and-Decisions.md) | Where are the component boundaries and unresolved design choices? |
| [Common Ground](Common-Ground.md) | What do our terms, transforms, clocks and representations mean? |
| [Data And Evaluation](Data-and-Evaluation.md) | What makes a trajectory auditable and a comparison fair? |
| [Contributing](Contributing.md) | How do I turn an investigation into a reviewed contribution? |

For a first session, read Home -> Common Ground -> TWIST2 Baseline. Then read the
literature rows relevant to your workstream and write a small experiment using
the contribution template. Do not start by installing every upstream project.

## What Is Not Decided

The Quest engine, desktop transport, storage format, reconstruction model and
RGB-D estimator are not selected. See the [decision register](Architecture-and-Decisions.md#decision-register).
No runtime API is introduced by these docs. No performance target, novelty
claim, or paper result should be presented as an achieved FURRY capability.
