# TWIST2 source-trainer finetuning (2026-10-09)

This experiment uses the original TWIST2 Isaac Gym environment, reward,
termination, domain randomization, motion curriculum and `DaggerPPO` at revision
`b06178f19a22f2138cbd31f60c6d494bc263f67d`. It is not the earlier MuJoCo reward port.
No improvement or headset deployment has been established.

## Data lineage and split

The released `twist2_dataset.yaml` contains 19,426 train entries: 5,841 OMOMO,
724 AMASS, 12,788 TWIST1 conversions, and 73 PICO clips. The PICO weight is 10 per
clip; other entries have weight 1. Before motion curriculum, PICO therefore gets
730 / 20,083 = 3.6349% of sampling probability. Sampling is per motion, not per
second. Curriculum subsequently changes effective probabilities.

`select_author_motions.py` deterministically selects 1,024 exact non-PICO train
members: 309 OMOMO, 38 AMASS and 677 TWIST1, totalling 8,770.12 seconds. It preserves
the original family sampling masses after subsampling. Files outside the released
train manifest and all PICO clips are excluded. There is no baseline-survival filter.

`author_dataset.py` combines these with 33 Quest train originals and 33 mirrors.
Quest replaces total PICO sampling mass (730); mirrors share their parent's weight.
The 12 held-out Quest clips (stages 2, 9, 14, 174.22 seconds) remain validation.
Do not create validation by re-splitting pretrained authors' train motions. The
existing validation is from one session and was already inspected diagnostically;
it is not an independent test set. The public train subset does not enlarge it.

The builder rejects split/stage leakage, missing augmentation parents, cross-split
identical content, changed source hashes and mismatched body lists. Source revision,
raw-file hashes and transformation metadata are retained in `manifest.json`.
Upstream YAML resolves `root_path: .` against CWD: launch from the dataset directory.

## Coordinate compatibility

Both sources use 29 G1 joints, the same 38 body names/order, Z-up robot space, meters,
radians, root quaternions XYZW in PKL. Quest is already anatomy-scaled and retargeted;
no additional body scaling is applied. Quest source rate is 100 Hz, author files
approximately 30 Hz; the original MotionLib interpolates both at simulator time.

Important boundary: historical Quest NPZ `local_body_pos` contains world-axis
`body_world - root_world`. Upstream PKL expects pelvis-oriented coordinates:
`R_root.T @ (body_world - root_world)`. `motion_coordinates.py` applies this rotation
when exporting PKL. NPZ files, including validation, stay byte-identical.
The earlier MuJoCo environment ignores this field and recomputes FK from `qpos`;
this export error does not establish the cause of earlier training/runtime falls.

Five FK frames per clip across the full mixture verified the convention. After the
rotation, Quest error is below 3e-15 m; AMASS/TWIST1 below 2e-6 m. OMOMO has a constant
1.5 cm difference at the two toe markers relative to our GMR model; original public
files are retained. Converted TWIST1 root quaternions have norm deviations up to
7.3%; upstream MotionLib does not explicitly normalize them. These author files
are preserved byte-for-byte for this source-pipeline baseline, and norm errors are
recorded, not silently corrected. This is a known inherited data limitation.

## Weights and explicit finetuning differences

Actor: published `assets/ckpts/twist2_1017_20k.onnx`, SHA256
`2d1fb3a31e4e967f70ecfefc3ad1e7b2ac491677068b89f60b565a94e7735061`.
Its actor parameters and frozen input affine normalization are recovered exactly.
FP32 GPU/ONNX parity on simulator observations was within 2.7e-6 absolute error.
The released ONNX has no critic, optimizer or exploration distribution to resume.

The adapter keeps the source actor/critic architectures and PPO objective, without
a DAgger teacher. Explicit differences needed for the small-data continuation:

- New critic with separately fitted, frozen normalization; the upstream runner's
  teacher-normalizer/critic-normalizer mismatch is not copied.
- Critic-only warmup, 500–1,000 iterations. Actor parameters are checked bit-for-bit.
  Actor unfreezes only after two 50-iteration windows with pre-update TD-lambda
  EV >= 0.7, normalized RMSE <= 0.65, and all train motions visited. This is an
  on-policy readiness proxy, not independent Monte Carlo proof near falls.
- Actor LR ramps to 1e-5 over 100 updates, fixed thereafter; critic warmup LR 3e-4.
  No upstream adaptive-LR escalation toward 1e-2 during this small-data finetune.
- Fixed action std 0.05; original learned exploration std was not released.
- Actor dropout disabled for consistent rollout/update likelihoods; FP32, TF32 off.
- Checkpoints every 100 actor updates. Stop on nonfinite statistics, a single
  measured post-update KL > 0.05, or a stop file. No new custom slip/fall penalty.

`run_author_pipeline.py` runs an independent baseline and checkpoint evaluation in
our immutable MuJoCo runtime at 100 Hz, on the exact 12 original validation clips,
seeds 0/1. These samples never enter PPO or critic normalization. Three successive
runtime regressions stop further optimization. Acceptance is reported only;
weights are not automatically installed or promoted. One training seed is a pilot,
not evidence of statistical improvement.

## Entry points

```bash
python -m unittest discover -s prototypes/quest_g1/training -p test_author_dataset.py
python prototypes/quest_g1/training/select_author_motions.py \
  --manifest twist2_dataset.yaml --index archive-index.json --output selection.json
python prototypes/quest_g1/training/fetch_author_motions.py \
  --selection selection.json --url-file public-download-url.txt --output raw
python prototypes/quest_g1/training/author_dataset.py \
  --selection selection.json --raw raw --quest quest-night-v3 \
  --author-manifest twist2_dataset.yaml --output mixed-dataset
```

The source trainer requires a separate Python 3.8 / Isaac Gym Preview 4 environment.
The prepared cluster environment is `~/furry/twist2-author-20261009` on `mipt-stud`.
Keep the dataset, actor weights and generated runs out of Git. Run frozen simulator
and disposable optimizer smoke checks before submitting `run_author_pipeline.py`.
