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

## Experiment

```bash
python training/evaluate.py --assets EMPTY_ASSETS --dataset CURATED --policy POLICY_ONNX --split validation --seeds 0 1 2 --output baseline.json
python training/train.py --assets EMPTY_ASSETS --dataset CURATED --initial INITIAL_PT --source PINNED_ACTOR_SOURCE --output RUN
```

Control is 100Hz and physics 1000Hz. Observations (1432), history ordering,
29 actions, PD and saturation match the deployed controller; the parity check
compares 100 physics steps. Episode resets use reference position/velocity
and zero history. Evaluation covers full clips; seed 0 is unperturbed and
seeds 1/2 add 0.005rad initial joint noise. This is an optimistic tracking test,
not a reproduction of live standing-to-walking transitions. Train episodes
sample random reference starts and end at 20s or a fall. Time limits bootstrap
the terminal value; falls do not. Neither crosses reset boundaries in GAE.

The reward implements the pinned TWIST2 tracking/regularization terms with
MuJoCo contact forces and 0.01s integration scaling. Unlike the upstream model,
key bodies use wrist-yaw frames and torso instead of rubber-hand/head-mocap
frames, matching the Quest embodiment. Torque-limit normalization uses the
Quest enforced PD caps. Joint errors are uniformly weighted. These explicit
model/reward adaptations prevent a claim of exact author parity. No additional
swing-clearance reward is used in this first experiment. The actual reward
terms are named in `env.py`.

PPO uses clipping 0.2, gamma sqrt(0.99) = 0.99498744,
GAE lambda sqrt(0.95) = 0.97467943, entropy coefficient 0.005,
actor LR 1e-5, critic LR 3e-4, KL early stop 0.02, 4 epochs, minibatches <=512.
Exploration starts at std=0.05. The first 10 updates warm only the fresh critic.
The discount and GAE coefficients preserve decay per second from the upstream
50Hz configuration at our 100Hz control rate. Reward terms already multiply
by DT in the environment; do not halve those coefficients again. The initial
experiment used unconverted gamma=0.99/lambda=0.95; its archived bundle remains
unchanged. This correction alone has not been validated by a new training run.
Dropout is disabled in the actor even during optimization to keep PPO ratios
well-defined. The released ONNX observation normalization is frozen; no
validation statistics are fitted. A 1000-update/16-env/128-step job collects
2,048,000 simulated transitions; it is an initial bounded experiment.

`run_slurm.sh` requests one RTX4500 Ada, 16 CPUs, 32GB and at most 8 hours.
Latest checkpoints are atomic every 10 updates. Validate every 100 updates;
choose candidate by completion rate then surviving time. Report tracking and
slip too: survival alone can favor standing. The initial policy stays intact;
`best.pt` means best candidate, not guaranteed better than baseline. Export
requires PyTorch/ONNX numerical parity. No automatic headset installation.
