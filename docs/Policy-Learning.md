# Policy Learning

[Home](Home.md) | [TWIST2 Baseline](TWIST2-Baseline.md) | [Data](Data-and-Evaluation.md)

**Status: proposed.** Written 2026-10-07. This page plans a state-based
high-level policy in the spirit of the TWIST2 Diffusion Policy. Nothing on it is
implemented or measured in FURRY yet. Statements about TWIST2 are author-reported
or inspected at the [pinned revision](TWIST2-Baseline.md).

## What TWIST2 Does

TWIST2 is hierarchical. The low-level policy is an RL motion tracker that turns
a 35-value command `p_cmd` into PD targets for 29 body joints. During teleop,
GMR retargeting produces `p_cmd`; during autonomy, a Diffusion Policy produces
the same vector. The policy's action is therefore exactly what the operator
recorded, and the tracker is not retrained.
[Paper, Section III-G](https://arxiv.org/html/2511.02832v1)

`p_cmd` is `[vx, vy, z, roll, pitch, yaw_rate, q_ref(29)]`: root velocities in
the root's own frame, height, tilt, yaw rate and all body joint targets. Hands
are commanded separately by interpolating between open and closed presets.

| Author-reported setting | Value |
| --- | --- |
| Observation | ZED Mini RGB resized to 224x224 (ResNet-18, R3M weights) plus `p_cmd` history |
| Action | Chunk of 64 `p_cmd` vectors (about 2 s), 48 executed |
| Model | Diffusion Policy with a 1D convolutional denoiser, modified from iDP3 |
| Augmentation | 10% Gaussian noise on proprioception; crop, rotation, color jitter |
| Inference | 20 Hz on one RTX 4090 |
| Results | WB-Dex: 170 demos, reaches the cup but often misses the grasp; Kick-T: 50 demos, 6 of 7 trials succeed |

The pinned release does **not** include high-level policy training code; its
README says it will ship in a separate iDP3-derived repository. The release
does include `deploy_real/server_data_record.py`, which records RGB plus
`state_*` and `action_*` streams, where `action_body` is the 35-value `p_cmd`.
FURRY therefore builds the model from public Diffusion Policy code or LeRobot.

## FURRY Variant

The policy observes **simulator scene state, not images**. Training and
evaluation both happen in MuJoCo. Rendering is used only to inspect behavior.

| | TWIST2 | FURRY |
| --- | --- | --- |
| Observation | RGB and `p_cmd` history | Low-dimensional robot, object and goal state |
| Encoder | ResNet-18 | None; normalized state conditions the denoiser |
| Action | `p_cmd` chunk and hand presets | `p_cmd` chunk and `grip` |
| Evaluation | A few hardware trials | Many seeded simulator rollouts with automatic success checks |

This is the low-dimensional Diffusion Policy setting. The state is privileged:
results do not transfer to camera input without separate work, and reports must
say so.

## Preliminary State Layout

Accepted as a starting point, to be revised by experiment:

- **Robot:** root height and roll/pitch, root linear and angular velocity in the
  root frame, and the 29 body joint angles.
- **Objects:** cup or T-box position and orientation in the robot pelvis frame,
  not world coordinates.
- **Goal:** `t_target` position and orientation in the same pelvis frame.
- **Hands:** current `grip`.
- **Encoding:** orientations as a 6D rotation or sin/cos of yaw, not raw
  quaternions or Euler angles.
- **Contract:** a fixed, named and versioned layout read by joint and body name,
  independent of the scene's `qpos` order.

Comparing robot joint state against TWIST2's `p_cmd`-history proprioception is
a planned ablation.

## Where It Plugs In

| Need | Existing FURRY piece | State |
| --- | --- | --- |
| Low-level tracker | [`g1_sim/controller.py`](../prototypes/quest_g1/g1_sim/controller.py), CPU port of TWIST2 at 100 Hz | Locally reproduced |
| Command interface | `Controller.step(data, command, grip)` with a 35-value `command` | Same contract as TWIST2 |
| Actions in recordings | Quest `mimic.csv` (`p_cmd`) and `frames.csv` (`grip_left`, `grip_right`); desktop NPZ `command` and `grip` | Recorded |
| Scene state | `qpos` and `qvel` include the robot and the `cup_free` and `t_box_free` bodies from [`g1_sim/scene.py`](../prototypes/quest_g1/g1_sim/scene.py) | Recorded as flat scene-dependent vectors |
| Tasks | `cup` (similar to WB-Dex) and `push_t` (similar to Kick-T) scenes | No success criteria yet |

The policy becomes a third command source next to the scripted arm demo and the
standing command in [`g1_sim/__main__.py`](../prototypes/quest_g1/g1_sim/__main__.py).

## Open Work

- **Converter:** recordings to versioned state, `p_cmd` and `grip`, resampled
  from the 100 Hz record to the action rate. Training exports are derived data;
  raw episodes stay untouched ([ADR-005](Architecture-and-Decisions.md#decision-register)).
- **Demonstrations:** Quest demos of cup pickup and Kick-T are not yet
  validated. Debug the whole loop first on scripted simulator demos.
- **Tasks:** success criteria, timeouts, fall detection, randomized starts and
  seeded resets for `cup` and `push_t`.
- **Runtime:** interpolate chunk steps to the 100 Hz tracker, start smoothly
  from the current pose and blend consecutive chunks. Root velocity errors
  accumulate, so the policy must observe the robot's pose relative to objects.
- **Hands:** keep `grip` presets for now; finger control belongs to ADR-004.
- **Evaluation:** fixed tracker, scene and seeds; splits by operator and
  session as in [Data And Evaluation](Data-and-Evaluation.md).

## First Steps

1. Run `g1_sim`, record an NPZ with `--demo` and inspect its fields.
2. Implement the state layout and the recording converter.
3. Script a simple reach task, generate demonstrations, train a small
   low-dimensional Diffusion Policy and close the loop through `Controller.step`.
4. Add success criteria and randomization to `cup` and `push_t`, then move to
   Quest demonstrations.

## Background

- Chi et al., *Diffusion Policy*, RSS 2023, and its `*_lowdim` configurations
  in [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).
- [iDP3](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy), the base of
  the unreleased TWIST2 high-level code.
- [LeRobot](https://github.com/huggingface/lerobot), whose Diffusion Policy
  accepts environment state.
