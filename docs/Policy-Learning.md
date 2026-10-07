# Policy Learning

[Home](Home.md) | [TWIST2 Baseline](TWIST2-Baseline.md) | [Data](Data-and-Evaluation.md)

**Status: proposed.** Written 2026-10-07. This page plans a state-based
high-level policy in the spirit of the TWIST2 Diffusion Policy. The policy
itself is not implemented. The [desktop NPZ contract](#desktop-recording-contract)
and the [state contract](#state-contract) below are locally reproduced. Statements about TWIST2 are author-reported
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
- **Encoding:** object and goal orientations as a 6D rotation, not raw
  quaternions or Euler angles. Root roll and pitch stay the TWIST2 Euler pair.
- **Contract:** a fixed, named and versioned layout read by joint and body name,
  independent of the scene's `qpos` order. The raw NPZ is not this vector;
  the converter builds it from the [recording contract](#desktop-recording-contract).
  Implemented widths, the 6D order and the 20 Hz export are in
  [State Contract](#state-contract).

Comparing robot joint state against TWIST2's `p_cmd`-history proprioception is
a planned ablation.

## Where It Plugs In

| Need | Existing FURRY piece | State |
| --- | --- | --- |
| Low-level tracker | [`g1_sim/controller.py`](../prototypes/quest_g1/g1_sim/controller.py), CPU port of TWIST2 at 100 Hz | Locally reproduced |
| Command interface | `Controller.step(data, command, grip)` with a 35-value `command` | Same contract as TWIST2 |
| Actions in recordings | Quest `mimic.csv` (`p_cmd`) and `frames.csv` (`grip_left`, `grip_right`); desktop NPZ `command` and `grip` | Recorded |
| Scene state | `qpos` and `qvel` include the robot and the `cup_free` and `t_box_free` bodies from [`g1_sim/scene.py`](../prototypes/quest_g1/g1_sim/scene.py) | Recorded as flat scene-dependent vectors; read them by joint name |
| Policy observation | [`g1_sim/state.py`](../prototypes/quest_g1/g1_sim/state.py) builds `state-v1` from a desktop NPZ | Derived export; the raw episode stays untouched |
| Tasks | `cup` (similar to WB-Dex) and `push_t` (similar to Kick-T) scenes | No success criteria yet |

The policy becomes a third command source next to the scripted arm demo and the
standing command in [`g1_sim/__main__.py`](../prototypes/quest_g1/g1_sim/__main__.py).

## Desktop Recording Contract

**Status: locally reproduced** on 2026-10-07. Host: Linux, Python 3.12.5,
MuJoCo 3.3.7, ONNX Runtime 1.23.2, NumPy 2.2.6, policy
`twist2_1017_20k.onnx` at TWIST2 `b06178f`. The NPZ stays in gitignored
`prototypes/quest_g1/outputs/`. Repeat the probe from that directory:

```sh
.venv/bin/python -m g1_sim --headless --scene lab --demo --seconds 10 --record outputs/demo_lab.npz
```

`--demo` waves the arms. It does not pick up the cup or kick the T-box. Over
10 s the pelvis stayed between 0.782 m and 0.793 m, tilt stayed at 2.4 degrees,
and MuJoCo reported no warnings. The cup moved 2.3 cm and the T-box 1.4 mm.
`grip` stayed 0.

### Fields

The file has 1000 rows. `time` runs from 0.010 s to 10.000 s in steps of
0.010 s. Physics is 1000 Hz and the tracker is 100 Hz. A row is written after
`Controller.step` when `step_index` is a multiple of the policy period, so the
first sample is at 0.010 s rather than 0.

| Field | Shape on this run | Role |
| --- | --- | --- |
| `time` | `(1000,)` | Simulator time after the physics step |
| `qpos` | `(1000, 64)` | Scene pose. Width depends on the scene and on whether hands articulate |
| `qvel` | `(1000, 61)` | Velocities in the same joint order as `qpos` |
| `ctrl` | `(1000, 43)` | Actuator torques, not position targets |
| `command` | `(1000, 35)` | `p_cmd` passed into the step that produced this row |
| `grip` | `(1000,)` | One scalar for both hands. Quest recordings instead store `grip_left` and `grip_right` |
| `scene` | scalar | `"lab"` |
| `scene_xml` | string | Model XML for replay. Mesh paths inside it are absolute |
| `metadata` | JSON string | Run statistics, not an observation tensor |

### Named pose layout

`qpos` follows the scene's joint order. That order is not the order of
`command`, and the addresses move when the scene or `--fixed-hands` changes.
The converter resolves them with `MjModel.joint(name).qposadr` loaded from
`scene_xml`. These addresses were read from `make_model` on the same date;
only the `lab` row was also checked against a recorded episode.

With articulated hands, each hand's seven joints sit immediately after that
arm, so the 29 body joints are not one contiguous block. A free body appends
seven values: `xyz` and a MuJoCo `wxyz` quaternion.

| Build | `nq` | Free joint at `qpos` address |
| --- | --- | --- |
| `stand`, hands | 50 | none |
| `cup`, hands | 57 | `cup_free` at 50 |
| `push_t`, hands | 57 | `t_box_free` at 50 |
| `lab`, hands | 64 | `cup_free` at 50, `t_box_free` at 57 |
| `lab`, fixed hands | 50 | `cup_free` at 36, `t_box_free` at 43 |

On the recorded `lab` episode the pelvis occupies `0:7`, the left hand
`29:36`, the right arm `36:43` and the right hand `43:50`. `t_target` has no
joint, so its pose exists only in `scene_xml`.

`command` is `[vx, vy, z, roll, pitch, yaw_rate, q_ref(29)]` in the `JOINTS`
order from [`g1_sim/controller.py`](../prototypes/quest_g1/g1_sim/controller.py):
both legs, waist, left arm, right arm. Fingers are not in it. On this episode
the root command stayed at the standing value (`z` is 0.8; velocities and tilt
are zero). Only shoulder pitch and elbow of both arms varied.

`ctrl` follows actuator order. On the right hand that order lists the index
finger before the middle finger, which is the opposite of `qpos`. Joint
position is not stored in `ctrl`.

### Command time

The stored `command` was evaluated one physics step before the recorded
timestamp, at `time - 0.001` s. Every row matches `arm_command(time - 0.001)`.
Compared with `arm_command(time)`, the largest absolute difference on this
episode is `1.83e-4` rad, which is the slope of the arm cosine across that
millisecond.

The tracker reads `command` at the beginning of a step whose `step_index` is a
multiple of the policy period. The file stores the command from the later step
that lands on the 10 ms mark. For this smooth demo those two commands differ by
the same millisecond. The converter keeps the stored `command` next to the
post-step state and does not rebuild the action from `time`.

## State Contract

**Status: locally reproduced** on 2026-10-07, from the same `lab` episode as
the recording contract. Host and versions are those listed there. The export
stays in gitignored `prototypes/quest_g1/outputs/`. Repeat from that directory:

```sh
.venv/bin/python -m g1_sim.state outputs/demo_lab.npz outputs/demo_lab_state.npz
.venv/bin/python -m unittest tests.test_state -v
```

[`g1_sim/state.py`](../prototypes/quest_g1/g1_sim/state.py) writes schema
`state-v1`. It does not open the source for writing. On this episode the
export has 200 rows and 66 state values. Exported `command` matches the stored
`command` on the kept rows exactly, `grip` stays 0, and root height stays
between 0.783 m and 0.793 m.

### Layout

The vector is a robot prefix, then each present scene body, then `grip`.
Body order is fixed: `cup`, `t_box`, `t_target`. A scene that lacks a body
omits that block. The contract is these field names, resolved through joint
and body names.

| Block | Fields |
| --- | --- |
| Root | Height, roll and pitch. Roll and pitch are the TWIST2 Euler pair |
| Root velocity | Linear velocity rotated into the pelvis frame, and the free-joint angular velocity, which is already in that frame |
| Joints | The 29 body angles in the `JOINTS` order from [`g1_sim/controller.py`](../prototypes/quest_g1/g1_sim/controller.py) |
| Each body | Position in the pelvis frame, then a 6D orientation |
| Hands | The recorded `grip` scalar |

The 6D orientation is the first two columns of the pelvis-frame rotation
matrix, stored row-major: `R00, R01, R10, R11, R20, R21`. An identity rotation
is `1, 0, 0, 1, 0, 0`. Root yaw is not a separate component. Heading relative
to the robot is carried by the object and goal poses.

| Scene | State width | Bodies after the robot prefix |
| --- | --- | --- |
| `stand` | 39 | none |
| `cup` | 48 | cup |
| `push_t` | 57 | T-box, `t_target` |
| `lab` | 66 | cup, T-box, `t_target` |

The width is 38 robot values, plus 9 per present body, plus `grip`. `lab` with
`--fixed-hands` keeps these names. Only the raw addresses change. The same
logical pose encodes to the same vector from both hand models.

`t_target` has no joint. Its pose is the body pose in `scene_xml`, expressed
in the pelvis frame. `command` is a separate array. Its names are `vx`, `vy`,
`z`, `roll`, `pitch`, `yaw_rate`, then `q_` prefixed to each body joint.

### Action rate

The record is 100 Hz. `state-v1` keeps every fifth row, which is 20 Hz.
TWIST2's action chunk is about 32 Hz and its inference is 20 Hz. This export
uses 20 Hz because 20 divides 100, so each kept `command` and `grip` is a
stored sample. A rate that does not divide 100 Hz, including 32 Hz, is
rejected instead of interpolated.

Rows are taken from the first sample. On the 10 s episode that is `t = 0.010`
through `t = 9.960` s, in steps of 0.050 s. The last four 100 Hz samples are
not on that grid. The kept `command` is still the stored value, which leads
`time` by one physics step.

The file contains `state`, `state_names`, `command`, `command_names`, `grip`,
`time`, `source_index`, `scene`, `schema` and a JSON `metadata` string.
`source_index` is the row each export sample copied from the raw episode.

## Open Work

- **Converter:** desktop NPZ to [`state-v1`](#state-contract) is locally
  reproduced. Quest CSV is not. Training exports are derived data; raw
  episodes stay untouched
  ([ADR-005](Architecture-and-Decisions.md#decision-register)).
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

1. Done. A desktop NPZ was recorded with `--demo` and its fields, named pose
   addresses and one-step command offset are in
   [Desktop Recording Contract](#desktop-recording-contract).
2. Done. [`g1_sim/state.py`](../prototypes/quest_g1/g1_sim/state.py) defines
   `state-v1` and converts a desktop NPZ without modifying it. See
   [State Contract](#state-contract).
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
