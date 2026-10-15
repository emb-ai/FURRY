# Quest camera legs — dataset-v3 (2026-10-14)

G1 motion clips (`quest-g1-motion-v1`, 100 Hz NPZ) for fine-tuning on real leg
motion from the external RGB-D camera skeleton, with the actor command bank.
Supersedes dataset-v2 for the general camera family (`source: quest-camera`).
Kicks are a separate dataset, `../dataset-v3-kicks`, so that the training mix
can give them their own share.

- Episodes: 34, one stage per episode (101–134).
  - Stages 101–129 are the dataset-v2 episodes, in the same order. All 231 v2
    clips are byte-identical here.
  - Stages 130–134 are new: five ep13 episodes (2026-10-14 15:48–15:56, Push-T,
    observer view, catch-up off). They were recorded with the quest-camv2-250
    policy, but the actor input is the operator's command, so the policy does
    not matter.
- Train: 116 clips, 2102 s of originals plus the same again as sagittal
  mirrors.
- Validation: unchanged from v2. Stages 103 (ep7 1791882756630195) and 116
  (ep12 1791970696625609), 177 s.
- Reward target and actor input are built as in v2. The target is clean3
  camera legs re-solved by native GMR; the actor input is the live headset
  `mimic.csv`. All 116 original train clips equal the live `mimic.csv` exactly.

Shin bias: the camera was moved between ep12 and the afternoon sessions
(registration shifted by up to 12 cm), and the right shin bias changed with
it. ep13 targets therefore use the afternoon session bias,
`camera_targets.py --shin -5.2 -2.0 -4.4 3.8`. It is the median over 8,543
quiet-stance frames of ep13 and ep15; ep15 alone gives [[−5.6, −2.3], [−3.2, 4.5]],
and its two halves agree within 1°. ep12 keeps `--shin -6.5 -2.2 4.1 0.5`.

Built by `prototypes/quest_g1/training/camera_dataset.py --mirror` (pair list:
`episode_stages` in `manifest.json`). Merge with
`training/quest_contact_clearance/camera_mixture.py combine QUEST night-v3
dataset-v3 dataset-v3-kicks` and set the shares with
`author_dataset.py --quest-shares quest-meta=…,quest-camera=…,quest-camera-kicks=…`.

Limitations: as for dataset-v2. One operator; cleaned feet suppress part of
real floor slides; no camera images included.
