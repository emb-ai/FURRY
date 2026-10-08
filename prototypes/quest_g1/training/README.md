# Quest motion-tracker adaptation

PPO adaptation of the released TWIST2 actor in the Quest MuJoCo embodiment.
This is **not** an Isaac Gym reproduction or an exact checkpoint resume.
The actor and its fixed observation normalization are recovered from ONNX;
the critic, exploration standard deviation and optimizer are new. The upstream
actor source must be pinned to TWIST2 b06178f19a22f2138cbd31f60c6d494bc263f67d.

Private recordings, motion files, weights, logs and generated assets are not
committed. The cluster bundle is copied only to the operator's account.

## Preparation

Compile `retarget_capture.cpp` with the prototype's native `gmr.cpp`,
`meta_retarget.cpp`, `simulation.cpp`, MuJoCo and ONNX Runtime. The executable
accepts assets, a strict source text stream and an output CSV. `prepare.py`
creates that stream from the capture and consumes the CSV.

```bash
python training/prepare.py EPISODE MOTIONS --assets ASSETS --retarget-bin BINARY
python training/curate.py MOTIONS CURATED
python training/add_replay.py CURATED PINNED_TWIST2
python training/policy.py POLICY_ONNX PINNED_ACTOR_SOURCE INITIAL_PT
python training/check_pipeline.py --assets EMPTY_ASSETS --dataset CURATED --policy POLICY_ONNX
```

Use the identical Quest assets with table/cup/T props removed for flat-floor
experiments. Keep collision geometry, inertias, actuator definitions and PD.
Preparation uses native fixed bind-based anatomy scaling (root/legs and arms
separately), STAGE Y-up conversion, observer mode, and the deployed sole guard.
Scales are recorded per calibration. **Run IK continuously from calibration**;
resetting the solver at each arbitrary clip start can select a wrong branch
when the subject has turned. Warm-up uses source poses only, never validation
rollout scores. State continuity across a split boundary is part of the fixed
source transformation, not a learned statistical fit.

Stages 2, 9 and 14 are validation; all their fragments stay in validation.
Other stages are train. Cleaning thresholds are fixed before training and
apply equally to both: drop measured discontinuities >25cm or >45deg/frame
with 0.5s margins, invalid inputs, and later retargeted speed bursts above
12rad/s for legs/torso or 16rad/s for arms with 0.25s margins. These are
conservative curation thresholds, not device specifications. Retain at least
2s per resulting clip. Positions/joints use a fixed 6Hz filter and rotations
SLERP; references are resampled to 100Hz and grounded after filtering. Source
files and exclusions remain available; never interpolate through a pause or
coordinate reset. This one-session validation is not an independent test.
Nine pinned public walk fixtures (excluding previously flagged 002) are first
evaluated on the same model. `curate_replay.py CURATED BASELINE_TRAIN_JSON`
keeps fixtures that complete that baseline at seed 0. In the first experiment,
001/003/005/006 qualify. These four supply 20% of training samples and are never
added to Quest validation. Keep the all-fixture baseline to expose exclusions.

## Corrected experiment (schema v2)

The first run (110651, iteration 790) regressed and was stopped. These changes
are a new experiment, not an exact resume or a claim that learning has improved.
Use a fresh output directory and the original recovered actor.

```bash
python -m unittest discover -s training -p 'test_*.py' -v
python training/check_pipeline.py --assets ASSETS --dataset CURATED --policy POLICY_ONNX
python training/train.py --assets ASSETS --dataset CURATED --initial INITIAL_PT --source PINNED_ACTOR_SOURCE --output NEW_RUN
```

Control is 100 Hz; physics and slip integration are 1000 Hz. The actor keeps
its released 1432 observations, history, 29 actions and frozen normalization.
The deployment parity check covers 100 control steps and verifies that the
shadow contact measurement does not modify physical integration state.

### Reward

`rewards.py` defines explicit experiment configuration. New slip is the sum
of contact-weighted sliding path over time, not squared speed or foot COM
velocity. Tangential velocity is evaluated at actual foot-floor contact
points with a shadow MuJoCo state. Per-foot speed is the normal-force-weighted
RMS; the force gate is Fn/(Fn+5 N). The smooth absolute speed is
sqrt(v² + 0.01²) - 0.01 m/s. Sum feet and multiply by the 0.001 s physics step,
then apply `--slide-weight` (initial candidate 0.5, not a tuned optimum).
This is a weighted path, not net foot displacement. Numerical impact peaks
have approximately linear rather than quadratic cost. No swing-height term
is added. `stumble` remains a diagnostic and has zero weight in the new flat
floor reward. `--legacy-slip` isolates the old COM-slip reward for an ablation.

The formerly saturated global-key reward uses a Cauchy kernel:
2/(1 + mean_body_squared_error/0.5²). Half reward corresponds to 0.5 m RMS
error. `--legacy-reward` restores both old slip and global-key formulas for
comparison. Local tracking and the other existing terms remain. Continuous
state reward rates already multiply by DT; do not apply a second rate correction.
Action differences and touchdown airtime events are exceptions: their rate
receives a 0.02/DT factor to preserve the upstream 50 Hz cost per physical
trajectory/event. Tests cover constant action slew and a fixed touchdown cost.
The new coefficients and formulas need learning ablations; offline correlation
alone does not establish an improvement in behavior.

### Critic readiness, before any actor updates

The new critic receives actor observations plus simulation qpos/qvel, world
and joint reference errors, orientation and key-body errors, foot contacts,
airtime, previous velocities and time remaining in the source motion. This
information is critic-only; deployment inputs do not change. Its normalization
is fitted to warm-up train rollouts and then frozen.

The actor is frozen while collecting 32 complete stochastic training rollouts
and 8 held-out rollouts with separate seeds, using std=0.05. Both sets use
training motions only; the protocol validation split never supplies gradients
or normalization. The critic holdout shares the training motion pool and is
a value-fit diagnostic, not independent motion generalization validation.
Targets are full-episode discounted Monte Carlo returns, never values from
the untrained critic. Samples are kept every 5 control steps; complete rewards
are still integrated at 100 Hz. Falls and source-motion ends terminate value.
Artificial 20 s PPO limits bootstrap the terminal pre-reset state once.
GAE never crosses resets. This explicitly corrects the old behavior that also
bootstrapped source-motion ends without a subsequent reference.

The critic is trained for at least 100 and at most 2000 minibatch updates.
Readiness requires two consecutive checks (25 updates apart): held-out
explained variance >=0.3, RMSE/target-std <=0.85 and absolute bias/target-std
<=0.2. These are configurable starting criteria, not universal guarantees.
Degenerate constant-target data cannot pass. If readiness fails, save the
report and checkpoint and stop with zero actor updates. `--warmup-only` runs
only this stage. `critic_holdout.npz` and rollout metadata preserve evidence.

### PPO and regression controls

Gamma=sqrt(0.99), GAE lambda=sqrt(0.95) preserve the original 50 Hz physical
time horizon at 100 Hz. Actor LR ramps from 1e-6 to 1e-5 over 50 updates;
critic LR is 3e-4. Actor and critic have separate optimizers and separate
norm-1 gradient clipping. Critic Adam moments are retained after warm-up. Four epochs, minibatches <=512, PPO clip=0.2 and
local KL stop=0.02 remain. A local KL stop stops actor updates only.

A frozen original teacher anchors actor outputs on current rollout states:
0.5*mean(((mu-mu_original)/0.05)²), coefficient 0.1. This is a fixed-scale mean
penalty, not a KL that can be reduced by inflating learned action std. The
public-motion replay mix remains 20%. Entropy coefficient is reduced to 1e-4;
std starts at 0.05 and is bounded [0.02,0.15]. These defaults are conservative
experiment settings; `--anchor-coef`, `--entropy-coef` and LR are configurable.

Evaluate the initial actor and candidates on complete Quest validation and
public replay clips, with the same seeds (default 0/1/2). Every 25 PPO updates,
compare tracking/slip on matched pre-termination prefixes, excluding the first
0.3 s. Reject new falls on baseline-successful trials, increased total falls,
>1% lost survival, >5% tracking regression (joint/local-key/root-position), or
>2% sliding regression, with 1e-4 absolute metric slack. To accept a candidate,
Quest validation also needs fewer falls or >2% improvement in sliding or joint
tracking; replay must not regress. Neither survival nor standing still alone
qualifies. These validation tolerances are recorded in each acceptance report.

At each evaluation, 4 fresh complete on-policy training rollouts probe critic
accuracy against Monte Carlo returns; a candidate also needs a healthy critic.
Stop after 3 consecutive baseline regressions or 2 failed critic probes.
Among accepted candidates select completion, then lower slip, then joint error.
`best.pt` exists only after acceptance. Otherwise the original policy is the
fallback. A selected candidate still needs testing on an independent recording
and real standing-to-walking transitions before headset deployment.

Logs separate raw environment reward from timeout bootstrap, policy loss,
value loss, entropy, teacher deviation, reward components, GAE-target value
metrics, clipping and complete-rollout critic probes. Save actor/critic and
both optimizer states for audit; automatic v1/v2 resume is not supported.

`run_slurm.sh` requests one RTX4500 Ada, 16 CPUs, 32 GB, up to 4 hours and
300 PPO updates. Extra trainer arguments can follow the bundle and venv paths.
It runs tests and parity checks first. Export only an accepted checkpoint,
verify PyTorch/ONNX parity, then evaluate with the recorded reward configuration.
No automatic headset installation. A run without an accepted candidate ends
with an explicit report and leaves the original policy as fallback.

### Short controlled slip ablation

Keep the same original actor, seed, data, readiness criteria and PPO settings.
Use separate output directories. Compare repaired PPO with `--legacy-slip`
against the same trainer with `--slide-weight 0.5`, initially 75 updates,
`--eval-every 25 --eval-seeds 0`. This is a pilot; accepted results still need
the full three-seed evaluation. If critic readiness fails, that arm has no
actor training result and must not be called a policy comparison.


## V3 bounded overnight adaptation

The v3 trainer adds 20 critic-only reference samples from 0.02 to 1.9 s.
The deployed actor still receives exactly 1432 values. Each future sample has
35 command values, relative root translation/rotation, and an end-validity bit;
clipped end padding has zero reference velocities. This is future conditioning,
not a claim of complete TWIST2 critic parity.

Monte Carlo warm-up visits every training motion at least twice and each
holdout motion once. Holdout means independent stochastic trajectories from
training motions, not Quest validation. Collection is reproducible across
worker scheduling. Periodic on-policy critic probes use fixed clips, initial
phases and noise seeds instead of changing their test distribution each time.

Actor LR persists across updates and adapts to analytic Gaussian KL with a
warm-up ceiling. The former KL0.02 minibatch early stop is removed; KL0.1 is
retained as an emergency stop within an update. PPO critic loss clips value
changes by 0.2. Separate actor/critic optimizers and original-action anchoring
remain. Checkpoints record the adaptive LR, and periodic evaluation also
reports paired-prefix reward on a fixed eight-clip original train subset.

`mirror_dataset.py` creates train-only G1 sagittal reflections and checks joint
limits, involution and forward-kinematic link poses. The fixed asymmetric IMU
mount is excluded from symmetry checks. Original/validation/replay files are
copied byte-for-byte. `mirror_guard.py` records baseline falls/tracking as diagnostics only.
A baseline failure does not make a reference invalid and must not exclude it.
All reflected clips passing data/geometry checks are retained. No validation
data is consulted for augmentation selection. Cached baseline rollouts can be
reused only when policy, dataset and simulation/evaluation source hashes match.
Mirror augmentation adds zero independent demonstrations. Use a fresh private
output directory; the source dataset is not edited.

`run_night.sh BUNDLE VENV` is a single eight-hour Slurm allocation. The bundle
must contain augmented `dataset`, unchanged assets/source/initial actor, and
`baseline-reference.json/.npz` from the deterministic fixed-train diagnostic.
Unit tests, physics parity, a real optimizer smoke test and the mirror physics
audit must pass before training starts. Two predeclared arms then run for at
most 3.4 hours each: old slip as control, and contact-point sliding weight2.
Each starts from the original actor; no failed checkpoint is used to initialize
the other arm. Both use32 environments x256 steps (8192 transitions),
minibatches2048, actor LR ceiling3e-6 with100-update warm-up, critic LR1e-4,
up to1500 updates, full validation/replay seeds0/1 every100 updates, and three
consecutive regressions or bad critic probes to stop. No checkpoint is deployed
automatically. Inspect `night-results.json`, per-arm acceptance reports and
`best.pt` (present only on acceptance), not just `latest.pt`.
