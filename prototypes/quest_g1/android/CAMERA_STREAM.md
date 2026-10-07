# External camera legs and a stationary VR world

The Quest connects directly to the camera server over LAN WebSocket. Networking
uses pinned IXWebSocket, independently of physics and rendering. Run
`scripts/fetch_camera_transport.sh` before the first Android build. Set
`android/camera_stream_url.txt` to the server's reachable address, for example
`ws://<server-ip>:8765`. `0.0.0.0` is a listening address, not a client address.
The repository default is blank (network disabled); the locally built APK uses
the configured server. The protocol follows `skeleton/skeleton_server.py`; native parsing uses JsonCpp.
Both dependency licenses are packaged with the APK. No Unity or desktop runtime
is required by the Quest client.

## Alignment

A resets alignment. Stand facing the camera, look forward and make a short
translation/crouch movement while remaining fully visible. Shoulders across time
provide a metric rigid transform (Horn/Kabsch equivalent, scale fixed to one).
Static shoulders alone are collinear: the client waits for sufficient motion
instead of inventing a pitch angle. At least 40 shoulder observations, transverse
spread above 2 cm and shoulder fit RMS below 40 mm are required. Calibration
locks once accepted; A or a STAGE recenter resets it.

The current server has no wrist landmarks. Optional `lwrist` / `rwrist` anchors
are supported if added later, matched to Meta wrists rather than controller
origins. If the nose is visible, centred nose/HMD trajectories contribute to
rotation estimation while head orientation stays within 0.15 rad of its initial
orientation. Centre subtraction removes the nose/HMD positional offset; nose is
never treated as the HMD origin. Headset occlusion can disable this extra anchor.

Camera packets use `frame=camera`, absolute camera coordinates. Calibrated
`frame=pelvis-relative` packets must already use Quest STAGE axes. Unknown frames,
invalid numbers, oversized JSON, missing required landmarks and reordered
packets are rejected. The client sends HMD/controller poses back to the server.

Camera epoch timestamps are matched to a 3 s Meta history (nearest body pose
within 60 ms), rather than the latest predicted HMD pose. Source age and arrival
age both gate use. Server and Quest clocks must agree; automatic clock-offset
estimation is not implemented yet. Persistent clock skew prevents fusion rather
than silently mixing mismatched motion.

## Fusion

Upper body and pelvis remain Meta-controlled. Only trustworthy depth `window`
measurements (confidence at least 0.55) of an entire hip/knee/ankle chain are used;
`line` and `missing` points do not become measured ground truth. Large correction
vectors above 45 cm are rejected.

The camera corrects pelvis-relative knee and ankle positions in the historical
Meta sample. A 60 ms exponential filter smooths the **correction**, preserving
current high-frequency Meta motion rather than delaying the whole skeleton.
Confidence weights fade over 100 ms; each leg expires against its last good
measurement, even when empty heartbeat packets continue. A two-bone projection
preserves Meta segment lengths and hip anchors. Hip/knee
orientations receive swing rotations matching the new bone directions. Ankle
orientation/twist remain Meta-estimated: the current camera has no toe/heel
landmarks, so camera-derived foot pitch and yaw are not claimed.

At more than 150 ms the HUD shows STALE and the correction weight decays;
at 500 ms the result is exactly the original Meta skeleton. Invalid new frames
expire old corrections. A fixed packet is never applied forever.

HUD shows CAM OFF/ALIGN/FUSE/STALE, age, fit error and contributing leg count.
CSV appends connection/state/age/fit/leg-count/weight metrics. Y recordings
include `camera.jsonl` with raw camera packets, receive epoch and associated
input sequence, alongside unmodified Meta body/input streams and applied mimic
commands. View switches are recorded as events. Participant recordings remain
local and excluded from Git.

## First-person world and robot feedback

The scene-to-STAGE transform is computed from the A-calibration camera pose,
HMD position, level axes and visual scale, then **held fixed**. Head motion changes
only the eye view; a stationary table/T object has unchanged STAGE coordinates.
A 0.5 m HMD displacement is a 0.5 m displacement in STAGE, independent of robot
lag. Recalibration intentionally changes the alignment. Physical dimensions do
not change when visuals are scaled.

GMR's soft camera goal uses the HMD's absolute displacement from calibration in
that same transform. In first person, whole-world pelvis travel uses this
visual-calibration scale too; anatomical limb/height scales stay separate.
Position feedback adds a bounded local velocity correction
(maximum 0.4 m/s) to the velocity-conditioned TWIST2 input. The resulting planar
command is limited to 0.8 m/s; heading correction is bounded at 0.8 rad/s and the
combined yaw command at 1.2 rad/s. This is an outer pose servo, not a retrained
policy or a physical-state teleport. Pitch/roll follow the soft IK task; this G1
has no actuated neck. The controller cannot promise exact head/hand alignment
when camera goals conflict with foot support, body tracking or policy capability.

## Verification and limits

- 37 Python model/recording/geometry tests pass.
- Native synthetic tests cover metric rigid fit, degenerate static alignment,
  fixed bone lengths, invalid heartbeat expiry and the 150/500 ms transitions.
- Pose-servo tests cover correction direction, zero command at goal, yaw sign
  and velocity limits.
- Native stereo/world checks keep a prop fixed in STAGE across HMD movement.
- Native WebSocket client receives the existing live server stream.
- A 15 s physical smoke test stays upright, but with static leg references the
  pose servo makes only limited progress. Walking convergence is not established.
- Live camera-to-Meta alignment still requires a visible, tracked person; empty
  server heartbeat frames verify transport only.
