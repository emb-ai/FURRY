# Quest camera legs — dataset-v1 (2026-10-13)

G1 motion clips (`quest-g1-motion-v1`, 100 Hz NPZ) for fine-tuning on real leg
motion from the external RGB-D camera skeleton, with the actor command bank.

- Episodes: 5 sessions of one operator (ep5, ep6, ep7, ep9 ×2), one stage per
  episode (101–105). Validation: all of ep7 (stage 103).
- Train: 187 s of originals + 187 s of sagittal mirrors (`parent_id` links;
  mirrors are augmentation, not new data). Validation: 89 s.
- Reward target: camera legs cleaned offline (`camera_targets.py`, clean3) and
  re-solved by native GMR. Actor input (`command_bank.npz`): the live
  `mimic.csv` recorded on the headset, per train clip `[t, 35 commands]`.
- Built by `prototypes/quest_g1/training/camera_dataset.py --mirror`; merge into
  the training mixture with `training/quest_contact_clearance/camera_mixture.py`
  (see that README).

Limitations: few minutes from one operator; cleaned feet suppress part of real
floor slides (shoe AprilTag check, 2026-10-14); no camera images included.
