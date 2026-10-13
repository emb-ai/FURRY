# Project integration, 2026-10-13

This publication consolidates previously isolated experiments; it does not
accept a new walking policy or install an APK on Quest. Baseline TWIST2 20k
remains the default. Virtual Legs swing lift is disabled by default.

## Included source

- Raw 84-joint Meta recording and the independently installable Capture package.
- Experimental HOMIE v2 and SONIC VR3 backends, pinned model identities and
  upstream licenses. ONNX files are Git LFS; run `git lfs pull` before building.
- Optional 10 cm Virtual Legs ankle lift and its paired diagnostic report.
- Committed TWIST2 adaptation source from `a130b08`, excluding dirty v4 work.
- Lower-body reference adapter through `2a7c65f`, with frozen TWIST2 and an
  experimental desired-swing reward; it is training code, not a live backend.
- Audited contact/input preparation from `0cc8679`; its unresolved GPU and
  source-data checks remain visible in the status snapshot.

Current camera leg selection, persistent registration and STAGE handling from
main are retained. Raw body recording precedes camera fusion. Existing
recordings retain their cadence and wrist-frame replay semantics.

## Evidence and limits

The HOMIE/SONIC and lift reports are diagnostic offline comparisons. Their
historical APK receipts describe earlier builds, not this integrated build.
SONIC memory, latency and usability on Quest have not been accepted.

Clearance-v2 produced two candidates passing a relative filter, including a
control candidate; there is no consistent reward advantage across seeds.
The adapter continuation with penalty x5 added one failure and did not solve
low transfer. New intent runs are still ongoing; no weights are promoted.

The DexWeave body-only code is a reconstruction, not released author training
code. The latest plan trains one shared policy from scratch on C04-C15 and
keeps C01-C03 out of training and normalization: six GPUs, 24576 environments,
2000 iterations. Its status file is a dated snapshot, not a live monitor.
Tracking termination is distinct from a physical fall; prefix averages alone
do not establish full-clip completion or general teleoperation.

## Validation

Package-one checks are recorded in `validation-package1.json`. CPU, native
backend and recording checks passed; the Linux Isaac Gym contract is a
separate environment check. Final Android build and scene checks are recorded
when complete. No headset installation is performed.

## Scene and visibility package

The second package brings cup laboratory visuals and placement feedback,
table tennis, the experimental physical 2x2 cube and controller wrist inputs
into the same runtime as HOMIE/SONIC. Robot meshes fade smoothly near the
operator camera; props remain opaque. The head mesh is identified separately
from its shared torso body. Transparent geometry does not write depth.

`validation-package2.json` records the combined build and checks. The nearby
head comparison images are desktop previews with the native shader, not
headset captures. Existing camera leg controls and registration are retained.

The Rubik scene remains a prototype. A free contact lift and a fixed-core face
fixture passed; reliable G1 bimanual quarter-turns and a manual VR solve are
not accepted. No physical robot or new trained policy is deployed.
