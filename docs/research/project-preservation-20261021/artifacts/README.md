# Archived project evidence and demo production sources

This collection preserves 215 previously untracked files from 7–16 October
2026, selected during the 21 October audit of main `68079cf`. The originals
occupy 29.5 MB. Fifteen byte-identical files already present in main were omitted.
The [manifest](manifest.json) records source paths, byte sizes, SHA256 hashes,
existing copies and omissions.

These are dated research and deployment records. A historical `running` status
is not a current process check. Reports retain their original claims and
limitations, including failed experiments. This archive does not promote a
policy, change the runtime, install an APK or establish new Quest acceptance.
Historical Markdown is stored as `.md.txt` to distinguish it from maintained
documentation. Links and absolute paths inside original reports may point to
local assets that were intentionally omitted.

## Contents

| Evidence | Preserved material |
| --- | --- |
| Camera and retargeting | Alignment report, RTMW/RGB-D summaries, segment lengths, GMR residuals and full-skeleton recording inventory |
| Locomotion | Contact/clearance experiments, recorded gait diagnostics, actor cadence and wrist audits, narrow-stance evaluation and reference-adapter results |
| Runtime and scenes | APK/source verification receipts, controller-wrist and raised-arm calibration fixes, cup/WB-Dex/Rubik scene checks and scene regression summaries |
| Operator video | Recorder/FOV results, the accepted clipping-fix frames and the original capture-property helper |
| Final presentation media | Montage selection/edit scripts and policy-comparison sources, input references and all eight resulting trajectories |

Useful entry points:

- [Camera alignment](world-skeleton/alignment-1804/report.json)
- [Full Meta skeleton inventory](full-body-recording-audit-20261010/inventory.json)
- [Retargeting segment audit](quest-walk-catch-up-20261009/all-segments.json)
- [Narrow-stance result](stance-finetune-20261013/final-b11-summary.json)
- [Reference-adapter control/treatment result](reference-adapter-20261012/intent-v1/latest-results.json)
- [Scene regression](scene-regression-20261014/summary.json)
- [Operator clipping acceptance](operator-video-fov-20261015/acceptance.json)
- [Eight-policy measured metrics](policy-comparison-20261016/metrics.json)
- [Published presentation and final videos](../../../demo/2026-10-16/README.md)

Recent cup, DexWeave cup-in-box and Rubik training experiments are preserved
separately in the sibling training archive. Existing published datasets and
policy weights retain their original locations. Duplicate raw recordings,
intermediate videos, failed/truncated PPTX downloads, device backups, APKs,
compiled libraries, vendored dependencies and caches are omitted.

## Verification and reproduction limits

Verify every original file from a checkout with Git LFS content downloaded:

```sh
python docs/research/project-preservation-20261021/artifacts/verify.py
```

The independent metric check additionally requires NumPy:

```sh
python docs/research/project-preservation-20261021/artifacts/verify.py --metrics
```

It recomputes mean pelvis XY and wrapped yaw errors from 2,000 evaluated samples
per controller, checks the common timestamps, and compares the results with the
published demo metrics. These metrics describe one 20-second native simulation
clip. They do not measure foot sliding or general policy quality.

The original rendering and collection scripts are archival source. They retain
local paths, macOS library/font assumptions and references to the exact native
snapshot used for the presentation. They are not portable command-line entry
points. Their source hashes and selected editing intervals are preserved; the
final MP4s remain in `docs/demo/2026-10-16`. Recomputing the saved tracking metrics
requires no native runtime or original video footage.
