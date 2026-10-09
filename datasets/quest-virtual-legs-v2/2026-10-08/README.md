# Quest virtual legs v2 — 2026-10-08

Training data collected on Meta Quest 3 for the FURRY Unitree G1 simulation.
This snapshot includes the unchanged source recording, the curated GMR motion
references and the train-only mirrored augmentation used in the night-v3 run.
It contains one operator/session, with estimated Meta legs and no camera footage.

## Download and verify

ZIP and NPZ payloads are stored in Git LFS. Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/emb-ai/FURRY.git
cd FURRY
git lfs pull --include="datasets/quest-virtual-legs-v2/**"
python3 -m pip install numpy
python3 datasets/quest-virtual-legs-v2/2026-10-08/verify.py
```

GitHub's ordinary source-code ZIP is not sufficient when it contains LFS
pointers. The verifier checks payload sizes/SHA256, archive CRC and source hashes,
NPZ shapes/finite values, split boundaries and the augmentation parent links.
[release.json](release.json) records the published file inventory.

## Contents

| Location | Contents |
| --- | --- |
| [raw/source-episode.zip](raw/source-episode.zip) | Unchanged episode 1791467626074995; 11 source files, 1180.70 s total, 900 accepted s across 15 stages |
| [audit/REPORT.md](audit/REPORT.md) | Source integrity audit, pauses, tracking jumps and limitations |
| [audit/audit.json](audit/audit.json) | Accepted segments and detailed audit measurements |
| [audit/suggested_exclusions.csv](audit/suggested_exclusions.csv) | Preliminary tracking-jump exclusion windows |
| [dataset-v2/manifest.json](dataset-v2/manifest.json) | 33 Quest train clips / 713.10 s; 12 validation / 174.22 s; 4 upstream replay clips / 62.47 s |
| [night-v3/manifest.json](night-v3/manifest.json) | The same originals/validation/replay plus 33 mirrored train clips; 66 Quest train clips / 1426.20 s |
| [mirror-check.json](mirror-check.json) | G1 reflection checks and original parent IDs |

All manifest-referenced NPZ files are included, with unchanged bytes and hashes.
Shared originals appear in both directories so each manifest works directly;
Git LFS stores their common contents once. There are **82 unique NPZ payloads**
and one source ZIP, about **262 MiB** of unique binary data. Redundant pickle/YAML
exports, intermediate datasets, checkpoints and diagnostic fall recordings are
outside this training snapshot.

## Splits and preprocessing

Validation holds complete protocol stages **2, 9 and 14**, before chunking.
All other stages are train. Source recordings are about 72 Hz; processed
references are **100 Hz**, with clips of at most 30 seconds. GMR runs continuously
from each calibration before slicing. The preprocessing uses fixed anatomy
scales, a 6 Hz zero-phase Butterworth filter and a sole floor guard; see the
manifests for calibration values and curation thresholds.

The 33 mirrors are sagittal reflections of train clips, with explicit
`parent_id` links. They add no independent demonstrations. Validation bytes and
stage assignments are identical in v2 and night-v3. The legacy
`validation_fraction` field refers to the original Quest split; augmented
seconds and replay must not be treated as independent collected data.

The four author-recorded TWIST2 replay examples are a separate `replay` split,
sampled at 20% during the referenced training run. Their IDs are
`0807_yanjie_walk_{001,003,005,006}`, from revision
`b06178f19a22f2138cbd31f60c6d494bc263f67d`; the manifests retain original source
hashes. The historical replay selection used baseline survival on this
embodiment and did not use Quest validation. Failed Quest baseline rollouts are
not a reason to remove otherwise valid Quest reference clips.

## NPZ format

Load with `numpy.load(path, allow_pickle=False)` and follow the manifest's
`file` fields; no machine-specific YAML paths or pickle loading are needed.

| Key | Shape / meaning |
| --- | --- |
| `qpos` | `(N, 36)`, float64; root XYZ in metres, root quaternion **wxyz**, then 29 body joint positions in radians |
| `fps` | Scalar, 100 Hz |
| `local_body_pos` | Quest clips only: `(N, 38, 3)`; body positions minus root position, still expressed in world axes |
| `link_body_list` | Quest clips only: 38 G1 body names, matching `local_body_pos` |

Root XY starts at zero in each processed clip. The two replay keys are `qpos`
and `fps`; derive body positions through the matching G1 model if needed.
Do not interpret `local_body_pos` as vectors already rotated into root axes.
Joint order and source geometry/controller hashes are recorded in the raw
episode manifest; the 14 finger joints are absent from these 36-value references.
The raw robot snapshot has additional hand/object coordinates.

## Limits

This is one session with no independent test set. Meta virtual legs are
estimates, not measured leg ground truth. The source includes tracking jumps,
pauses and recalibration; the audit records them. Its `frames.csv` and `mimic.csv`
contain frozen calibration states of the robot and are **not motion targets**.
Processed GMR references also do not establish successful physical execution,
stable walking or completed cup/push-T tasks. `camera.jsonl` is empty.

## Provenance and licenses

The source ZIP SHA256 is
`4a77d267da0b23a2431d0dd79bb646a5e8610b716436e68cc15fcc7c2dd52414`.
The preparation implementation is preserved in FURRY's
`codex/quest-tracker-adaptation` history at revision `91b8ec7`; the payload
manifests identify the actual data by hashes, independently of later code edits.

FURRY's original Quest recording and its derivatives use the repository's
[Apache-2.0 license](../../../LICENSE). The four upstream replay examples retain
the [TWIST2 MIT notice](TWIST2_LICENSE.txt), copyright 2025 Yanjie Ze. Upstream
[example-motion provenance](https://github.com/amazon-far/TWIST2/tree/b06178f19a22f2138cbd31f60c6d494bc263f67d)
is separate from the FURRY recording. No AMASS or OMOMO data is included.
