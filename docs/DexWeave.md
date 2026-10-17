# DexWeave

[Home](Home.md) | [Literature](Literature-Review.md#dexweave-2026) | [TWIST2 audit](TWIST2-Baseline.md)

Snapshot: **2026-10-13**. Sources: [project page](https://dexweave.github.io/),
[arXiv:2609.34724](https://arxiv.org/abs/2609.34724),
[paper PDF](https://dexweave.github.io/dexweave.pdf) and
[code repository](https://github.com/WU-CVGL/DexWeave). Paper claims below are
**author-reported**. FURRY's own work is a reconstruction; see
[What FURRY Has](#what-furry-has).

## What It Is

Sun, Shen et al. (Westlake University, Shanghai Jiao Tong University) learn
dexterous humanoid loco-manipulation from human body, hand and object
demonstrations. The robot is a Unitree G1 with Inspire hands. There are two
parts: interaction-consistent retargeting, then an anatomy-aware Transformer
policy trained with PPO. The paper describes offline imitation of recorded
clips. It does not describe live VR teleoperation.

## How It Relates To GMR And TWIST2

| | GMR | TWIST2 | DexWeave |
| --- | --- | --- | --- |
| Role | Kinematic retargeter. | Teleoperation and data collection system. | Retargeter plus per-clip tracking policy. |
| Input | Human skeleton per frame. | PICO headset and calf trackers, retargeted by GMR. | Recorded body, hand and object motion. |
| Method | Scale to the robot, then two-pass damped least-squares IK under joint limits. | One general RL tracking policy follows a streamed 35-value reference. | Two-stage retargeting, then PPO from scratch per reference clip. |
| Hands | Not modeled. | Grasp presets from controllers. | Twelve independent drivers across both Inspire hands. |
| Physics | None, so feet can slide or penetrate. | In the policy. | Contact-aware retargeting and the policy. |
| Live teleoperation | Yes, as a front end. | Yes. | Not shown. |

## Retargeting

Stage 1 solves the body and the hands separately. The body solve tracks
landmarks with soft collision clearance, temporal regularization, joint limits
and support constraints. Foot contacts are inferred from source foot kinematics
to suppress sliding. Each wrist target blends a GMR-style morphology-preserving
target with a world-aligned target, weighted towards the world during sustained
hand-object proximity. Hands start from the nearest entries of a codebook of
feasible configurations.

Stage 2 refines only the arms, wrists and finger drivers. It weights fingertip
positions (thumb and index highest), palm orientation, collision clearance and
temporal smoothness. The wrist position is softened so error can be split
between the wrist and the fingers. Leg joints keep their Stage 1 values.

Reported against GMR on OMOMO: penetration duration 0.002 versus 0.916, and
contact distance 2.944 cm versus 7.677 cm. On LAFAN1 foot skating is 0 versus
0.010 for GMR.

## Policy

The actor sees 238 values: 211 robot and 27 object. The policy uses eight
anatomical tokens: base, two legs, waist, two arms and two hands. Masked
attention lets body tokens exchange information but never attend to hands;
each hand attends to the waist, both arms and itself. Object state is fused
only into the waist, arm and hand tokens, so legs stay object-independent.
Seven heads output 41 joint-position targets for PD control. Training is a
single PPO stage with no pretrained tracker, distillation or residual policy,
and the authors report about twice the convergence speed of MLP baselines.

In MuJoCo sim-to-sim body tracking the authors report 100% completion and
2.62 cm body error, against BeyondMimic's 100% and 2.82 cm. The paper notes that
this table trains a separate policy per reference sequence. Real G1 results are
video only.

## Use For Quest Teleoperation

DexWeave is not a drop-in replacement for the TWIST2 policy. A per-clip policy
is trained on a known reference, so it cannot follow an unseen operator stream
in real time. Its usable ideas for FURRY are:

- Inferring foot support from motion and anchoring the stance foot during
  retargeting, applied before the tracking policy sees the reference.
- Leaving legs fixed while refining the hands, so hand contact work does not
  disturb balance.
- Masked attention between body-part tokens for a future shared policy that
  includes Dex3-1 hands.

For camera-assisted legs, including the 2D keypoint experiment, the stance-foot
anchor is a cheap constraint. 2D leg fits are weakest along the camera's depth
axis, and a pinned stance foot removes much of that ambiguity. DexWeave has no
camera input, so it does not inform the reprojection residual itself.

## What FURRY Has

[`prototypes/quest_g1/training/dexweave`](../prototypes/quest_g1/training/dexweave/README.md)
is an independent reconstruction of the body-only experiment, using Meta
virtual legs and the FURRY GMR reference. The public repository had no
released training code or weights when it was checked on 2026-10-12. The
reconstruction now also trains one shared policy across clips; see the
[integration record](research/project-integration-20261013/README.md). No
DexWeave policy is promoted to the Quest runtime.
