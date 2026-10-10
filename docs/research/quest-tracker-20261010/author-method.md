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

The first pilot `111224` incorrectly overrode source PPO settings with fixed LR
1e-5, fixed std .05, 512 environments and a whole-run KL stop. It stopped after
118 updates. That pilot is not a faithful source-settings baseline.

The corrected adapter passes `class_to_dict(tc.algorithm)` unchanged to upstream
`DaggerPPO`. Actor training uses the original initial LR 2e-4, adaptive schedule,
desired KL .008, 5 optimization epochs, 4 minibatches, 24 rollout steps, PPO clip
.2, gamma .99, lambda .95, entropy .005 and gradient norm 1. Original action std
starts at 1.0 and is learned, with `fix_action_std=False`; the separate `action_std`
list in the config is inactive in this mode. Training mode includes the original
dropout. The main allocation uses 4,096 environments as in the source config.
The unchanged source PPO reduces LR for high minibatch KL and continues all
updates; there is no added minibatch skip or whole-run stop based on KL.

Explicit remaining differences for this requested continuation:

- Published pretrained actor with its frozen input normalization, replacing random
  actor initialization. Neither original critic nor learned std was released.
- Smaller non-PICO subset + Quest replacement, with protected stage validation.
- Requested critic-only warmup, 500–1,000 iterations. Actor/std are frozen during
  this phase; critic uses source initial LR, held fixed during warmup only. Two
  50-iteration readiness windows require TD-lambda EV >= .7, normalized RMSE <=
  .65 and all motions visited. This proxy is not independent Monte Carlo proof.
- Separate frozen critic normalization, avoiding the upstream runner's mismatched
  teacher/critic normalizer objects. Source actor/critic networks and losses stay.
- FP32/TF32 off for repeatable actor parity, checkpoint interval 100, finite pilot
  budget of 1,000 actor updates rather than a 30,001-iteration from-scratch run.

`run_author_pipeline.py` independently evaluates baseline/checkpoints in the
immutable MuJoCo runtime at 100 Hz on all 12 original validation clips, seeds 0/1.
Validation never enters PPO or critic normalization. Regression reports do not
silently stop the run, and no checkpoint is automatically installed/promoted.
Explicit stop files, invalid/nonfinite data, a failed critic-readiness prerequisite,
and the declared iteration/Slurm budget can still end a run. One training seed is
not evidence of statistical improvement.

`check_upstream_kl.py` exercises the real source PPO with deliberately high KL and
checks that LR decreases while the update and following rollout both complete.

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

## Window-displacement experiment, 2026-10-09

This is an explicit new reward experiment, not an exact author reproduction.
It starts again from the published 20k actor and runs 500 actor updates with
the source PPO/std/dropout settings. No whole-run KL stop or validation stop.
The fresh critic is prefitted with actor/std frozen before actor updates; it is
not assumed to be valid after changing rewards, contacts and critic inputs.

`--window-reward` adds two negative Huber terms to the unchanged source reward:
`-sum(Huber((actual_xy_delta - reference_xy_delta) / 0.25 m))`
and `-Huber(wrap(actual_yaw_delta - reference_yaw_delta) / 0.5 rad)`.
Each coefficient is 0.25 reward units/second, multiplied by policy dt exactly once.
These scales are experimental, not author coefficients. A frozen-actor smoke
with coefficients 1 produced about 1.55–1.65 penalty units/second against about
1.34 source reward units/second. Coefficients were reduced to 0.25 before the
training run, to keep this an auxiliary objective (about 30% at initialization).
The terms are dense over consecutive 1-second windows, not a delayed endpoint
loss: score the boundary sample, then rebase. Both displacements are expressed
in their own trajectory's heading at window start. Thus world origin and an
old heading offset do not create an invisible catch-up objective. Reset/RSI
also rebases the window. This does not enforce absolute position recovery.
The actor gets no new inputs or servo. The critic gets nine additional features:
actual/reference XY displacement, sin/cos of both yaw changes, and window age.
The source velocity, pose, contact/slip rewards and termination remain present;
correct displacement alone cannot distinguish a good step from sliding.

`--sole-urdf` selects a derived copy of the source G1 URDF with only the two
cylinders per sole replaced by main's four 5 mm spheres. Inertias and other
collisions are unchanged. This aligns sole geometry, not the entire PhysX and
MuJoCo contact solvers or policy frequencies. Runtime evaluation uses the
corresponding MuJoCo sphere scene, without catch-up, for both baseline and
checkpoints. Baseline is recalculated before optimization. Existing validation
is diagnostic, not a new independent test set.

Quest references are reconstructed with main `3d0164f` Meta toe clamp from the
same continuous calibrated GMR epochs. The no-clamp reconstruction must exactly
match each original clip before accepting its derivative. Train/validation
cuts, anatomy scaling, filtering and the existing sole-floor preprocessing are
unchanged; 33 train originals are mirrored again, all 12 validation clips retain
their held-out stages (2, 9, 14). The 1,024 author train clips, sampling weights
and author/Quest initial sampling masses remain unchanged. No author train
motion enters validation. Evaluation scores both policies against this same
clamped derivative; its numbers should not be spliced into old-reference tables.

Local provenance/data: `outputs/quest-finetune-20261009/window-displacement/`.
Remote isolated root: `~/furry/twist2-window-20261009`.

## Continue from the user-selected 1000 checkpoint

The user selected `Quest: дообучение 1000` (`111260/checkpoint_001000.pt`)
after trying it in the headset. The initial published-actor window run 111484
was stopped at the user's request after 15 critic updates, zero actor updates.
The replacement lives in `~/furry/twist2-window-from1000-20261009`.

`--init-checkpoint` restores the exact actor, frozen input normalization and
learned exploration std. It does not restore the old optimizer or critic: the
critic has nine new inputs and a changed reward/contact distribution. Critic
warmup precedes 500 additional actor updates. Source PPO settings, new window
reward, geometry, toe-clamped references, sampling and holdout remain unchanged.
The comparison baseline is now checkpoint 1000 on the same new runtime inputs,
not the released actor. This is actor continuation, not an exact optimizer resume.

`--init-manifest` is mandatory with the checkpoint. Its bytes must match the
checkpoint's recorded dataset hash. `checkpoint_lineage.py` requires every
ancestor clip to keep its split, author train content to remain unchanged, and
Quest stages/mirror parents to remain fixed. It rejects old train hashes or
train stages in validation, even under new IDs. Validation contains only the
same held-out Quest stages 2, 9 and 14; all 1090 ancestor train motions remain
in train. The check runs both before baseline evaluation and before optimization.

## Training performance (2026-10-09)

`--fast-critic` (trainer or pipeline) omits actor forward/log-probability/entropy
work during the frozen-actor critic warmup. It uses the source minibatch
permutation, clipped value objective, gradient clipping and Adam updates.
It refuses trainable actor/std, a teacher, recurrent policies or adaptive LR.
Actor-phase updates still call the unmodified source `DaggerPPO.update()`.
The skipped dropout/action samples change the subsequent RNG stream, so this
is opt-in and is not a bitwise replay of a whole upstream training run.
The unevaluated surrogate is explicitly marked in metrics.

Rollout statistics now accumulate on GPU and transfer once per rollout.
Float32 reductions are accumulated in float64 to match the former Python-float
sums. `--legacy-rollout-metrics` retains the reference path for benchmarking.
This does not change reward tensors or the optimizer. The standalone timing
effect was within the variation of the short benchmark; do not claim a separate
speedup from it.

`check_training_performance.py --device cuda` runs against the pinned source
PPO. On GPU job 111526, both the tiny fixture and source ActorCriticFuture,
with clipped and unclipped value losses, produced bitwise-identical parameters,
Adam states and value loss for two updates (40 minibatches) per case.
All seven rollout metrics matched exactly for 24 steps of 4096 environments.

Sequential same-GPU benchmark 111526 used 4096 environments and six critic plus
six actor updates per variant, excluding startup/transition samples. The two
reference critic medians were 3.134/3.127 s; fast critic was 2.964 s (about 5%
less time per warmup iteration). Actor medians were 3.643/3.640 s vs 3.660 s:
this optimization does not accelerate actor training. These are short-run
measurements, not a claim about total job duration. Existing training job 111501
was not restarted or modified. Full benchmark artifacts are under
`outputs/quest-finetune-20261009/performance-optimization/` in the project workspace
and `~/furry/performance-optimization-20261009/` on MIPT.

Removing the per-substep `fetch_results` was tested only in a disposable
benchmark. It reduced short-run iteration times, but was not adopted: repeated
unmodified simulator runs already diverged on the first step, preventing the
planned full-state equivalence certification. In diagnostic job 111531, initial
root/joint/observation tensors, RNG states and first actions matched, but
post-step states did not. This does not prove the candidate changes physics;
it means the current test cannot establish its equivalence. Production keeps
the original physics loop, timestep, substeps and synchronization.
The diagnostic rerun then segfaulted during simulator creation for its third
variant, before the optimized metrics/critic code ran; its no-fetch case did
not run. Job 111526 completed all five variants and the CUDA parity checks.

## 50 Hz evaluation and gait diagnostics (2026-10-09)

`author_eval.py` and `run_author_pipeline.py` default to `--control-hz 50`.
The actor and its history update every 20 ms, with actions held between calls.
The existing MuJoCo physics stays at 1 kHz; reference frames, diagnostics and
legacy reward reporting stay at 100 Hz. Dataset files are not resampled and
source Isaac Gym training remains 50 Hz. `evaluate()` retains 100 Hz as its
backward-compatible default. Reports record actor/history/trace frequencies;
comparison rejects different frequencies, scenes, splits and diagnostic schemas.

`gait_evaluation.py` adds read-only foot contact and full-quaternion heading
measurements. Swing labels use the unchanged dataset qpos reference, >=1 cm
relative clearance and >0.3 m/s foot speed. Complete reference swings must last
at least 80 ms; partial swings at warmup/prefix boundaries are excluded. This
reference construction differs from the Meta/GMR replay study in main, so its
absolute numbers must not be spliced into that study's tables. Existing sliding,
fall and XY tracking measurements remain present. Braking uses net horizontal
floor force opposing the foot body's horizontal velocity, with a 50 N threshold.

Gait/heading/progress comparisons are recomputed on each shared survival prefix,
excluding its first second. Directed progress projects robot displacement onto
the corresponding reference path segments sampled every 0.5 s. It can be zero
for sideways motion or negative for opposite motion even when path lengths are
equal. It is separate from along-track/cross-track displacement error, heading
error and path length ratio. Static reference segments have no progress ratio.
Legacy acceptance remains diagnostic, with no new metric thresholds or early
termination of training. All per-clip metrics are available for checkpoint review.

The pipeline evaluates baseline and every 100-update checkpoint at 50 Hz, then
baseline/final at 100 Hz for compatibility. Stage 5 backward walking is a separate
TRAIN diagnostic, never a new validation slice. `--previous-training` requests
retrospective 50 Hz evaluation of the previous run's saved checkpoints. Existing
12 validation clips/stages 2, 9, 14 and all ancestor train membership are preserved
by the same lineage guards. No train-to-validation transfer, catch-up, new reward,
friction, sole or clearance change is introduced by this work.

The requested new run starts again from user-selected `111260/checkpoint_001000.pt`
with the existing displacement/turn reward, fresh critic warmup and 500 actor
updates. This is a repeat with improved evaluation and optional fast critic,
not continuation from the previous run's additional 500 updates and not a claim
that the reward changed. Previous run 111501 completed its 500 actor updates.
