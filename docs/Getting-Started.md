# Getting Started

[Home](Home.md) | [Contributing](Contributing.md) | [Baseline audit](TWIST2-Baseline.md)

## Work On The Foundation

Reading the Markdown needs only a browser or editor. Documentation checks use
Git and **Node.js 22 or newer**. Node is documentation tooling, not a choice of
desktop application language.

For a new checkout:

```bash
git clone https://github.com/emb-ai/FURRY.git
cd FURRY
git switch main
npm ci --ignore-scripts
npm run docs:lint
npm run docs:check
npm test
git diff --check
```

Start new work from the repository's current default branch. In the
owner's existing workspace, do not clone again. Inspect `git status --short
--branch` and the current worktree before editing.

Choose an umbrella from the [roadmap](https://github.com/emb-ai/FURRY/issues/1),
then propose one bounded task, an expected artifact and a test. Read the
[evidence vocabulary](Home.md#evidence-vocabulary) before reporting results.

## Prerequisites By Workstream

The table below is a planning guide, **not a validated FURRY runtime setup**.

| Workstream | Likely tools or hardware | Gate before installation |
| --- | --- | --- |
| Docs, source audit, experiment design | Git, Node 22+ for checks | None; macOS or Linux is sufficient for documentation. |
| Released TWIST2 reproduction | Linux/NVIDIA reference environment, upstream Python environments, MuJoCo, ONNX Runtime, Redis | Read the pinned audit, select a host, and record versions. |
| Quest acquisition | Quest 3, developer access, Android build/deploy tools; SDK/engine depends on reuse result | Check device capability and licensing; do not preselect an engine. |
| Controller-free hands | Optical-hand and body-tracking APIs where available | Record permissions, SDK/runtime versions, supported fields and invalidity signals. |
| RGB-D experiment | Available RealSense device and compatible driver/pose estimator | Inventory the actual camera/host before choosing packages. |
| Policy training | Separate GPU environment and licensed motion datasets | Only after baseline reproduction shows a concrete training need. |

The upstream README splits Python 3.8 training/deployment and Python 3.10+
retargeting environments because of Isaac Gym dependencies. The released
offline motion server itself imports Isaac Gym and constructs CUDA tensors.
This prevents assuming that a CPU-capable low-level ONNX loop makes the entire
pipeline portable. [Pinned sources](TWIST2-Baseline.md#runtime-and-portability)

## macOS And Linux

The bootstrap machine is Apple Silicon macOS. Documentation tools are checked
here; no robotics dependency stack is installed by this task. Upstream Linux
instructions must not be pasted into macOS unchanged.

For Linux reproduction, use a separate environment and pin the upstream commit.
Do not delete existing Conda environments to follow a README verbatim. Keep
Redis on loopback for an offline baseline. Remote streaming requires a separate
network-access decision, not disabling protected mode or the firewall.

A bounded [Quest/G1 prototype](../prototypes/quest_g1/README.md) now provides
a macOS/CPU loop and a standalone Quest runtime. It is not the full upstream
pipeline or an accepted final architecture. Follow its own setup from
`prototypes/quest_g1/`; Python dependencies and runtime assets are separate from
the documentation tooling. MuJoCo's macOS
passive viewer has specific launch requirements documented by
[MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

## Keep The Checkout Small

Store third-party checkouts beside FURRY, not inside it. Put private recordings,
models, calibration captures and participant data outside the repository;
reserved local `data/`, `recordings/`, `checkpoints/` and `third_party/` directories
are ignored as an additional guard, not permission to redistribute their contents.

The standalone prototype is an explicit local-build exception: its documented
`vendor/`, `.venv/`, `.tools/` and `outputs/` directories are ignored beneath
`prototypes/quest_g1/`. Never stage their models, SDKs or recordings.

Do not commit headset serials, local network addresses, access tokens or private
student identifiers. Obtain permission before collecting identifiable footage.
Never run `sim2real.sh`, physical-robot SDK setup or robot network instructions
as part of this project.

## First Deliverable

Before a large implementation, submit a small source audit or experiment plan:
one question, pinned sources, required inputs, success/failure criteria, and the
next decision it enables. Use the [experiment template](Contributing.md#experiment-report-template).
