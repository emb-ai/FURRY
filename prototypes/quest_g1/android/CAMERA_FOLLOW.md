# First-person camera calibration

First person uses a fixed scene-to-Quest STAGE transform established by A.
Objects remain stationary while the headset moves. Left stick click toggles
view without moving this world. Standing eye-height calibration changes visual
scale, not physics dimensions or inertia.

GMR receives a soft absolute HMD camera goal; a bounded outer pose loop keeps
requesting motion toward the goal after the user stops. The robot's physical
lag remains visible. Full convergence and exact hand correspondence are not
guaranteed by the pretrained policy.

See [CAMERA_STREAM.md](CAMERA_STREAM.md) for the camera/Meta fusion, pose servo,
coordinate frames, limits and verification. This replaces the previous live-HMD
scene attachment, which moved stationary props in physical space.
