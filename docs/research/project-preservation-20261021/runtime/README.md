# Runtime and camera preservation audit

Baseline: `68079cfdfa36fbaef020eb192db3a477f8fe961c` on 2026-10-21.
The [manifest](manifest.json) records exact source commits, patch bases, SHA-256
digests and LFS object sizes. Historical Markdown is stored as `.md.txt` so it
remains an unchanged record rather than current project guidance.

## Integrated additions

- The standalone [inference logger](../../../../skeleton/inference_logger/README.md)
  from commit `81159c1aac822f35f782632851f058a42081dd63`: bounded raw camera,
  inference and Quest transport logs, offline clock fitting, session validation
  and episode import. Its native sync patch is supplied but not applied.
- The optional [desktop 2D/GMR viewer](../../../../skeleton/DESKTOP_2D.md), body
  solver and keypoint helpers from the camera 2D branch, with synthetic tests.
- The local `--debug-arms` camera packet extension, merged additively while
  retaining main's person tracking, timing/session identity, backend selection,
  depth constraints and recording behavior.

Twenty-five targeted tests passed: logger 7, camera transport/packet 6, body
solver 6, GMR retargeter 4 and keypoints 2. The logger patch passes
`git apply --check` against current `camera_stream.cpp`. No APK was built,
installed or tested on Quest by this audit. No new live camera run was made.

## Archived experiments

### Temporal leg denoiser

`skeleton-denoiser-v0.patch` is the complete binary Git diff from the exact
merge base to the recorded remote branch revision. The sibling
`skeleton-denoiser-v0/` contains the changed skeleton sources, historical
notes, processed datasets and model files. Six LFS objects total 43,998,167
bytes. Their original paths, full OIDs and archive locations are in the manifest.

The historical checkpoint's evaluated errors differ from the original training
log; exact training provenance has not been re-established. The data cover one
operator/session, with only five left and one right validation examples above
60 degrees of knee flexion. The model therefore remains an experiment. The
branch's camera-only leg-hold mode was based on older fusion code and is not
enabled or merged into the current Quest runtime.

### Camera 2D legs

`camera-2d-legs.patch` retains the complete remote branch change, including
headset solver, packet fields, UI and native tests. The main integration uses
only the desktop/offline pieces described above. The full branch was reported
as synthetic-only; it changes which input drives legs when 2D packets arrive.
Its native runtime changes have not been accepted on a headset.

### Predicted Meta wrist orientation

`predicted-hands-working-tree.patch` records the tracked local delta relative
to its original HEAD. `predicted-hands-supplemental/` retains differing untracked
source files; identical untracked sources already in the baseline are listed
by hash in the manifest. `predicted-hands-evidence/` retains the original
reports, tests and comparison source.

The versioned `predicted_hand_orientation` experiment preserves Meta wrist
rotation through its bind frame instead of replacing it with controller or
forearm-neutral orientation. Historical evidence records 112 passed tests,
one optional skip, 39 exact synthetic replays and six kinematic recording
comparisons. The APK was built but explicitly kept local; no installation or
live acceptance was confirmed. Existing main wrist defaults remain in effect.

This patch also contains older scene/menu/SONIC edits from the shared dirty
worktree. It is an archival snapshot, not a patch to apply on current main.
Apply any historical patch only in an isolated checkout at its manifest base;
external assets and the locally built APK are not included in this archive.

## Deliberately excluded from active runtime

Most dirty root/prototype and `furry-publish`/`furry-homie-sonic` files are older
versions of changes already integrated in main. Copying them would remove newer
camera registration, capture FOV, scene, recording and policy work. The remote
`camera-skeleton` branch is already an ancestor of this baseline.

Root floor/swing replay scripts use historical fixed recording column layouts
and are not a new accepted gait fix. The old floor-rollback build changes would
also discard newer packaging and recording functionality. Machine-specific
camera launchers, stale October 7 `RUNNING.md`, virtual environments, caches,
signing material, generated builds and APK backups are excluded.
