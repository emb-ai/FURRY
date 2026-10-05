# Compressed Literature Review

[Home](Home.md) | [TWIST2 audit](TWIST2-Baseline.md) | [Research design](Data-and-Evaluation.md)

Snapshot: **2026-10-05**. This is a selective working map, not an exhaustive SOTA
survey. Paper claims are **author-reported**; public code links establish
availability, not a successful FURRY reproduction. Check the exact revision,
weights, data licenses and causal timing assumptions before adopting a method.

## Comparison Matrix

| Work | Sensing and method | Embodiment and implementation | Relevance and limit for FURRY |
| --- | --- | --- | --- |
| [TWIST](#twist-2025) | Full-body mocap references; retargeting and RL plus behavior-cloning tracking. | G1; [released training/deployment code](https://github.com/YanjieZe/TWIST). | Understand the controller lineage; does not remove the need for sparse-input reconstruction. |
| [TWIST2](#twist2-2025) | PICO plus calf trackers; GMR-derived retargeting, tracking policy and data collection. | G1; [released code](https://github.com/amazon-far/TWIST2) includes ONNX policies and walking clips. | Starting baseline; Quest sensing and working dexterous-hand simulation are separate extensions. |
| [GMR](#gmr-2025) | Requires human skeletal motion, measured or estimated elsewhere; optimization-based retargeting. | Multiple humanoids; [code](https://github.com/YanjieZe/GMR). | Strong initial retargeter candidate; input pose quality and dynamic feasibility still matter. |
| [OmniH2O](#omnih2o-2024) | Sparse VR or RGB-derived pose goals; teacher/student motion-tracking policies. | Humanoid whole-body control; [code](https://github.com/LeCAR-Lab/human2humanoid). | Directly relevant sparse-control baseline; reproduce its sensing/training assumptions, not only the demo interface. |
| [HOVER](#hover-2024) | Mode-dependent position, joint or root commands, including sparse head/hand modes; policy distillation. | Humanoid; [Isaac Lab release](https://github.com/NVlabs/HOVER). | Suggests flexible command interfaces; not evidence that arbitrary unobserved human motion is identifiable. |
| [SONIC](#sonic-2025) | Motion references or VR/planner interfaces; scaled tracking with a shared token representation. | G1 among release targets; [code and checkpoint instructions](https://github.com/NVlabs/GR00T-WholeBodyControl). | Useful comparator, not the required starting stack. Account for model-specific lookahead and deployment requirements. |
| [AvatarPoser](#avatarposer-2022) | Head/hand motion -> learned full-body pose, with IK refinement. | Human avatar, not robot dynamics; [official code](https://github.com/eth-siplab/AvatarPoser). | Candidate sparse-completion baseline; human reconstruction and robot execution need separate tests. |
| [QuestSim](#questsim-2022) | Sparse HMD/controller signals -> physics-based avatar motion using learned control. | Simulated human avatar; paper available, official runnable code not verified in this audit. | Physics can regularize completion, but avatar feasibility is not G1 feasibility or proof of actual leg recovery. |
| [XRoboToolkit](#xrobotoolkit-2025) | XR pose streams, IK and visual feedback; multiple tracking modalities. | [Framework project](https://xr-robotics.github.io/) and [Quest client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest). | Evaluate reuse first; inspect Quest-specific feature gaps rather than assuming parity with PICO. |
| [Open-TeleVision](#open-television-2024) | Head/hand teleoperation with active stereoscopic robot feedback. | Upper-body/dexterous humanoid manipulation; [code](https://github.com/OpenTeleVision/TeleVision). | Feedback and operator-interface reference, not a complete balance/locomotion solution. |
| [OPEN TEACH](#open-teach-2024) | Quest 3 hand gestures/poses and visual feedback for manipulation. | Arms, hands and mobile manipulation; [official code](https://github.com/aadhithya14/Open-Teach). | Particularly relevant to controller-free control and data collection; whole-body G1 tracking remains separate. |
| [AnyTeleop](#anyteleop-2023) | Vision-based arm/hand teleoperation across morphologies and camera setups. | Arm-hand systems; [author project](https://yzqin.github.io/anyteleop/) and [hand-retargeting component](https://github.com/dexsuite/dex-retargeting). | Reuse hand-mapping ideas before writing another solver; a component release is not a complete G1 integration. |

## What We Can Test

FURRY's proposed contribution is a measured low-sensor simulation/data-collection
system, not the unqualified claim of being the first headset teleoperator.
Existing sparse-tracking work already makes that framing too broad.

Three useful hypotheses are: uncertainty-aware completion improves usable
camera-free demonstrations; a calibrated RGB-D view improves selected ambiguous
motions enough to justify setup cost; and recording provenance plus achieved
body/hand state improves replay and data quality. Each needs a matched baseline
and a failure analysis, not just a new interface.

Use SDK body estimates as an explicit baseline. Compare estimated pose quality,
retargeting quality and dynamic execution separately. A learned prior may trade
faithfulness for plausible balance; neither should be hidden by a single score.

## Platform And Data References

- [Meta IOBT and generated legs](https://developers.meta.com/vr/blog/inside-out-body-tracking-and-generative-legs/): distinguish observed upper-body signals from generated lower-body motion.
- [Meta Movement samples](https://github.com/oculus-samples/Unity-Movement) and [native OpenXR samples](https://github.com/meta-quest/Meta-OpenXR-SDK): verify current device/API availability and permissions.
- [RealSense SDK](https://github.com/realsenseai/librealsense) and [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker): candidate acquisition/pose building blocks, not a prevalidated fused skeleton or ground-truth system.
- [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/): authoritative simulation, state and viewer reference.
- [AMASS](https://amass.is.tue.mpg.de/): possible offline human-motion resource; access and constituent dataset terms require separate review before download or redistribution.

No datasets or model weights were downloaded for this review. SONIC's release
distinguishes source-code and weight licensing, and describes checkpoint-specific
reference horizons. Compare latency and reuse terms per artifact, not per project
name. [Release documentation](https://github.com/NVlabs/GR00T-WholeBodyControl)

## Bibliography

### TWIST (2025)

Ze et al. *TWIST: Teleoperated Whole-Body Imitation System*.
[arXiv:2505.02833](https://arxiv.org/abs/2505.02833).

### TWIST2 (2025)

Ze et al. *TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System*.
[arXiv:2511.02832](https://arxiv.org/abs/2511.02832).

### GMR (2025)

Araujo et al. *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking*.
[arXiv:2510.02252](https://arxiv.org/abs/2510.02252).

### OmniH2O (2024)

He et al. *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning*.
[arXiv:2406.08858](https://arxiv.org/abs/2406.08858).

### HOVER (2024)

He et al. *HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots*.
[arXiv:2410.21229](https://arxiv.org/abs/2410.21229).

### SONIC (2025)

Luo et al. *SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control*.
[arXiv:2511.07820](https://arxiv.org/abs/2511.07820).

### AvatarPoser (2022)

Jiang et al. *AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing*.
[arXiv:2207.13784](https://arxiv.org/abs/2207.13784).

### QuestSim (2022)

Winkler, Won and Ye. *QuestSim: Human Motion Tracking from Sparse Sensors with Simulated Avatars*.
[arXiv:2209.09391](https://arxiv.org/abs/2209.09391).

### XRoboToolkit (2025)

Zhao et al. *XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation*.
[arXiv:2508.00097](https://arxiv.org/abs/2508.00097), revised November 2025.

### Open-TeleVision (2024)

Cheng et al. *Open-TeleVision: Teleoperation with Immersive Active Visual Feedback*.
[arXiv:2407.01512](https://arxiv.org/abs/2407.01512).

### OPEN TEACH (2024)

Iyer et al. *OPEN TEACH: A Versatile Teleoperation System for Robotic Manipulation*.
[arXiv:2403.07870](https://arxiv.org/abs/2403.07870).

### AnyTeleop (2023)

Qin et al. *AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System*.
[arXiv:2307.04577](https://arxiv.org/abs/2307.04577).
