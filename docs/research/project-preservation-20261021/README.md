# Project preservation audit — 21 October 2026

Audited upstream main: `68079cfdfa36fbaef020eb192db3a477f8fe961c`.
The audit compared 13 local FURRY checkouts, the root prototype, saved research
outputs and fetched remote branches. Most apparent uncommitted runtime changes
were older versions of work already integrated into main. They were not copied
over the newer runtime.

This publication preserves missing data, useful standalone tools, final materials
and experimental evidence. It does not establish new live Quest acceptance or
promote unsuccessful training runs to the headset policy menu.

## Added to maintained project paths

| Addition | Evidence and limits |
| --- | --- |
| [67 reviewed WB-Dex cup-to-box episodes](../../../datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/README.md) | Three additive commits from `codex/quest-cup-15-trajectories`; 1,120.67 accepted seconds, 1,470,437,128 bytes in LFS. Original outcome labels remain 38 success / 29 failure; independent terminal-state review is a separate field. No new train/validation split. |
| [Camera/Quest inference logger](../../../skeleton/inference_logger/README.md) | Raw capture, bounded logging, clock fitting and offline episode alignment. The optional native synchronization patch is supplied but not applied. |
| [Desktop camera 2D → GMR viewer](../../../skeleton/DESKTOP_2D.md) | Standalone solver/viewer and synthetic tests; live headset 2D-leg integration remains archived and disabled. |
| Optional camera arm landmarks | `pose_yolo.py --debug-arms` adds elbow/wrist diagnostics while preserving the default nine-joint packet, session identity and timing fields. |
| [Fresh presentation](../../demo/2026-10-16/README.md) | Unmodified Google export obtained on 21 October: 23 slides, complete ZIP/CRC/XML relationships. It replaces the missing export, not the Google source. The PPTX contains GIFs but no embedded video streams. |
| [DexWeave overview](../../DexWeave.md) | Preserves the branch's dated paper review and corrects the local implementation status against current main. The original state-policy proposal is already superseded by main and was not imported over it. |

The dataset's source commits are `751daac`, `20c22d7` and `d237977`.
Existing cup/bimanual datasets, current runtime sources, packaged policies,
default selection and dataset splits are retained.

## Preserved research

- [Training source, selected data and results](training/README.md): cup ACT and
  state-only experiments, native control-rate diagnostics, DexWeave cup training,
  and Rubik feedback/control/curriculum work. The manifest reuses byte-identical
  files already in main and supplies a verified workspace reconstruction tool.
- [Runtime experiments](runtime/README.md): predicted Meta wrist orientation,
  the temporal leg denoiser with six original model/data files, and the complete
  camera 2D integration patch with exact base revisions. Their limitations and
  unaccepted behavior remain explicit.
- [Project evidence and media production](artifacts/README.md): camera alignment,
  retargeting, locomotion, scenes, installation receipts, video/FOV acceptance,
  montage source and trajectories for independently recomputing the eight-policy
  comparison.

Historical reports are byte-preserved as `.md.txt`; their old absolute paths and
`running` statuses are historical evidence, not current operational instructions.
Each subarchive has a SHA256 manifest. Original local workspaces were retained.

## Important findings

- The native cup replay comparison exposes a control-rate mismatch: original
  100 Hz replay succeeded in 8/8 cases, the 30 Hz path in 5/8, and aligned 50 Hz
  replay in 8/8. These are recorded-command replay checks. The separate learned
  ACT full-reset evaluations were 0/10 at the older rate, 2/10 after 50 Hz
  alignment, and 0/10 for the measured-state variant. They are not general task
  acceptance.
- The Rubik archive contains a verified deterministic sequence and replay
  evidence, but curriculum training stopped at stage 0 with no accepted policy.
  It does not demonstrate arbitrary scramble solving.
- The denoiser's published checkpoint and historical training metrics have a
  provenance mismatch and limited validation coverage. Its weights are preserved
  for investigation, not enabled in the live camera path.
- Main already includes the committed locomotion and body-only DexWeave trainer
  work from the old training branches. Cherry-picking those whole branches would
  duplicate or regress current code. Old unrun MuJoCo v4 changes remain local.

## Remaining gaps and deliberate exclusions

The best visual policy weights named in the existing reports,
`cup-mix-imnet-100k-75k` and `bimanual-unet-h16-imnet-100k-50k`, were not found
in local project files or the known FURRY directories on the two training hosts.
The user confirmed on 21 October that these weights are held by teammates.
The reports and training/evaluation code are already in main; this publication
does not include those two checkpoints, which were not available in this workspace.

APKs, device backups, signing keys, caches, virtual environments, generated
libraries, redundant raw exports and intermediate videos were not published.
Existing source recordings and local experiments were not deleted. Corrupt
and obsolete PPTX exports were excluded; the fresh complete export is included.
The media production archive preserves selected inputs and scripts, not every
original video used by the montage.

## Verification

Fetch LFS content before verifying the archive:

```sh
git lfs pull --include="docs/research/project-preservation-20261021/**,docs/demo/2026-10-16/*.pptx,datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/*.zip" --exclude=""
python datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/verify.py
python docs/research/project-preservation-20261021/artifacts/verify.py --metrics
python docs/research/project-preservation-20261021/training/archive.py verify
```

Subarchive READMEs describe dependencies and reproduction limits. Current checks
and publication evidence are recorded in [validation.json](validation.json).
Passing archive checks confirms bytes and stated offline tests; it does not
confirm live camera alignment, a newly installed APK or policy quality.
