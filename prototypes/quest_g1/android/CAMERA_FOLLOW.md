# First-person camera calibration

First person uses a fixed scene-to-Quest STAGE transform established by A.
Objects remain stationary while the headset moves. Left stick click toggles
view without moving this world. Standing eye-height calibration changes visual
scale, not physics dimensions or inertia.

In first person GMR first solves the body reference without the camera task.
It then translates all fourteen task positions, including feet, by the
horizontal difference between the achieved reference camera and the HMD goal.
The kinematic free-root warm start receives the same translation. Joint angles,
relative limb geometry and floor height are preserved. A final solve retains
the soft camera height/orientation objective.

The translation uses the reference camera, not the lagging physical camera;
adding the latter's error would count the already absolute world travel twice.
No physical state is overwritten. Tests cover a one-metre translation and a
stationary repeated goal without accumulated displacement.

A bounded outer pose loop keeps requesting motion toward the goal after the
user stops. The robot's physical
lag remains visible. Full convergence and exact hand correspondence are not
guaranteed by the pretrained policy.

See [CAMERA_STREAM.md](CAMERA_STREAM.md) for the camera/Meta fusion, pose servo,
coordinate frames, limits and verification. This replaces the previous live-HMD
scene attachment, which moved stationary props in physical space.

A synthetic stationary-leg test separates reference placement from locomotion:
GMR reaches the translated camera target within 0.01 mm, while the physical
policy reduces a 30 cm error to only 27 cm over 15 seconds. Raising the GMR
camera position weight from 10 to 1000 does not improve that physical result.
Reference alignment is therefore not validation of autonomous catch-up gait.
