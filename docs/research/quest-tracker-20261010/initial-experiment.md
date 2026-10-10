# Quest tracker adaptation — 2026-10-08

Status: submitted and observed running as Slurm job **110651**, account
`mipt-stud`, node `node211-17`, partition `AMD7V12-RTX4500-CCP-common`.
One RTX 4500 Ada GPU, 16 CPUs, 32GB, hard wall limit 8 hours.
Remote root: `~/furry/quest-adapt-20261008/`.
Run: `bundle/runs/110651/`; stdout: `slurm-110651.log`.

## Data and scaling

Source episode 1791467626074995; 900 accepted seconds, source files untouched.
Fixed GMR scaling by calibration:

| Calibration sequence | Root/legs | Arms |
| --- | ---: | ---: |
| 8720 | 0.8990447494 | 0.8167136587 |
| 30925 | 0.9111222937 | 0.8276851751 |

Continuous native GMR from each calibration, then fixed smoothing and sole
floor guard; slice after preprocessing. Splitting before IK warm-up produced
wrong mid-turn branches; that intermediate dataset/baseline is retained only
as a negative preparation check and was not sent for the full training run.

The final Quest dataset (`dataset-v2`) contains 33 train and 12 validation
clips: **713.10s train, 174.22s validation (19.63%)**. Validation holds whole
stages 2, 9 and 14; no adjacent random-frame split. There is no independent
second-session test yet. The source consists of estimated Meta legs, not
measured leg ground truth. Frozen robot recording streams are not motion targets.

Twenty percent of training sampling is public replay. Initial candidates are
nine pinned TWIST2 walking fixtures (pre-existing exclusion 002). Four survive
the baseline on this embodiment: 001, 003, 005, 006. The other five are excluded
from replay, with the original all-fixture baseline retained. No Quest
validation outcome is used to select replay.

## Initial policy and baseline

Current ONNX SHA256:
`2d1fb3a31e4e967f70ecfefc3ad1e7b2ac491677068b89f60b565a94e7735061`.
Actor weights and fixed observation normalization recovered exactly, checked
on 512 inputs: max absolute ONNX/PyTorch difference **3.8147e-6**. Critic and
optimizer are new; this is not an exact author checkpoint resume.
Observation/PD physics parity against the deployed Python controller over
100 physics steps: maximum qpos difference **0**.

Final continuous-reference baseline, same model, control 100Hz, physics 1000Hz:

- Quest validation: **33/36 trials complete**, 3 falls; seeds 0/1/2.
  Surviving time 520.19s out of 522.66s prescribed across seeds.
- Quest train: **29/33 trials complete**, 4 falls; seed 0.
- Initial all-public replay candidates: 4/9 complete; reported separately.

The reset uses reference qpos/qvel with zero history; seeds 1/2 add 0.005rad
joint noise. Thus these are controlled tracking results, not claims of live
headset performance. Clip-level completion is biased toward short clips;
report surviving/reference duration and tracking/slip as well. Baseline logs
predate replay-only manifest pruning: all Quest motion hashes are unchanged.
`manifest-before-replay-qc.json` preserves their original manifest.

## Training

MuJoCo PPO adaptation in the deployed Quest embodiment, **not Isaac Gym
parity**. Isaac Gym was not present in the account. PyTorch CUDA and MuJoCo
were installed in a separate venv; existing projects/environments untouched.
Actor: recovered TWIST2, 788105 parameters. Critic: new 512/256 MLP.
1000 updates x 16 environments x 128 steps = at most 2,048,000 transitions.
Initial 10 updates warm only the critic. PPO clipping 0.2, actor LR 1e-5,
critic LR 3e-4, gamma 0.99, lambda 0.95, entropy 0.005, KL early stop 0.02.
Released normalizer frozen; actor Dropout disabled for consistent log-probs.
Existing TWIST2-style tracking, slip, stumble, air-time and action regularizers;
no added swing-height reward. Adaptations: actual wrist/torso model frames,
uniform joint error weights, and actual Quest torque caps/contact forces.
These differences are documented in training/README.md and env.py.

Validation every 100 updates, atomic latest checkpoints every 10. Candidates
remain separate from the current policy. Final export checks ONNX parity and
reruns validation on seeds 0/1/2. Nothing is installed on the headset by this job.

## Launch incident

Job 110649 was submitted prematurely while SCP was still copying. Its
preflight failed before training and it was cancelled. After SCP completed,
the archive SHA256 and all 213 bundle file hashes passed before job 110651
was submitted. Archive SHA256:
`61dfad2d30dbbbdb140e516d7eb76ffd7dbe6704a00790ad065019dfc1750727`.

## Commands

```bash
ssh mipt-stud 'squeue -j 110651'
ssh mipt-stud 'tail -5 ~/furry/quest-adapt-20261008/slurm-110651.log'
```

Implementation lives in local FURRY worktree `.tools/furry-training`, branch
`codex/quest-tracker-adaptation`. Source/data/result hashes and exact run
configuration are in the private bundle and run directory. No participant
recordings or weights were published to GitHub.

## Launch verification snapshot

Job 110651 was still RUNNING at 4m55s, with iteration 98 completed
(200,704 transitions). Actor optimization is active after the 10-update
critic warm-up; finite PPO metrics and a 19MB latest checkpoint were observed.
Launch config, pipeline check, metrics snapshot and Slurm status are saved
in `job-110651/`. This is startup evidence, not a final training result.

Implementation committed locally as `2ba13ac` on
`codex/quest-tracker-adaptation`; working tree clean. The uploaded training
bundle is separately pinned by its manifest and archive hash. Later local
additions only cover preparation validation, replay selection and documentation;
the running trainer is unchanged. Nothing has been pushed to GitHub.

## User-requested stop and visual comparison

Training stopped cleanly after iteration **790**, on explicit user request.
Slurm job 110651 completed with exit 0:0 after 40m25s; GPU released.
The training process handled SIGTERM and saved latest.pt. The wrapper then
exported and evaluated the best candidate (iteration 100); that final validation
is distinct from the latest iteration 790 used for the requested video.
Latest ONNX export parity maximum absolute difference: 1.9073486328125e-6.

Video: comparison/before-left_after-right.mp4. Left is unchanged Quest baseline;
right is the final iteration 790, not the best validation candidate. Same
seed 0, reference reset, zero history, assets and input, 1x playback. Three
clips chosen as the first >=20s clip in each held-out stage 2/9/14, showing
the first 20s. Freeze on evaluation fall termination; no hidden resets.
Baseline survives all displayed intervals; latest falls at 4.00s, 4.63s and
2.76s. These selected intervals do not replace full-dataset evaluation.
Detailed hashes and frame states are in comparison/comparison.json and NPZs.
