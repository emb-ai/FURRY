# Quest camera legs — dataset-v2 (2026-10-14)

G1 motion clips (`quest-g1-motion-v1`, 100 Hz NPZ) for fine-tuning on real leg
motion from the external RGB-D camera skeleton, with the actor command bank.
Supersedes dataset-v1: same builder and format, ~10× more data.

- Episodes: 29 episodes from 6 sessions of one operator, one stage per
  episode (101–129). dataset-v1 episodes (ep5, ep6, ep7, ep9 ×2), ep11
  (2026-10-13, AprilTag session, 1 long episode), ep12 (2026-10-14, 23
  episodes, Push-T scene, wired network).
- Train: 106 clips, 1972 s of originals + the same again as sagittal mirrors
  (`parent_id` links; mirrors are augmentation, not new data). Camera legs
  were on for 1921 s of the train originals.
- Validation: two whole episodes from different sessions, 177 s:
  ep7 1791882756630195 (stage 103, Empty scene, observer view) and
  ep12 1791970696625609 (stage 116, Push-T, kicks with both legs, no fall).
- Reward target: camera legs cleaned offline (`camera_targets.py`, clean3) and
  re-solved by native GMR. Actor input (`command_bank.npz`): the live
  `mimic.csv` recorded on the headset, per train clip `[t, 35 commands]`.
  All 106 original train clips equal the live `mimic.csv` exactly.
- Built by `prototypes/quest_g1/training/camera_dataset.py --mirror`; merge into
  the training mixture with `training/quest_contact_clearance/camera_mixture.py`
  (see that README). The steps to record and process are in
  `prototypes/quest_g1/CAMERA_RECORDING.md`.

Differences from dataset-v1:
- ep12 targets use one shin bias for the whole session
  (`camera_targets.py --shin -6.5 -2.2 4.1 0.5`, the per-frame median over
  19,674 quiet-stance frames of the session). Per-episode estimates had
  ranged from −12 to +39° on the right shin, and in one episode the target
  bent the knee 35° more than the operator. dataset-v1 episodes and ep11 keep
  the per-episode estimate, as in dataset-v1.
- ep5 and all ep12 episodes were recorded in first-person view. There GMR
  adds the HMD task: leg commands change by < 1.5° (p95), but root velocity
  commands differ from observer view (p95 |Δvx| 0.08, |Δvy| 0.14 m/s).
  Targets and commands of each clip come from the same mode, so clips are
  self-consistent. The native Meta bank is camera tracking OFF.
- ep12 usable share 0.96–0.99 per episode, against 0.66–0.85 in dataset-v1
  (no Wi-Fi stalls on the wired network).
- Four ep12 train episodes end in a robot fall (1791970225150705,
  1791970794315569, 1791971807132839, 1791971940007610). They are kept: the
  target comes from the operator, and the falls are policy failures.

Limitations: one operator; cleaned feet suppress part of real floor slides
(shoe AprilTag check, 2026-10-14). In stance, camera hips are 10–15° more
extended and knees ~10° straighter than Meta legs; no ground truth says which
is right. No camera images included.
