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

Alignment runs whenever Quest body tracking is valid, even while robot control
is paused or awaiting calibration. A resets alignment. Face the camera and walk
a little (forward/back and sideways, ~0.5 m) while remaining fully visible.
Shoulders across time provide a metric rigid transform (Horn, scale fixed to
one). The HUD shows progress and what is missing.

The PC can lock that transform first (`skeleton/marker_frame.py`): a printed
AprilTag 36h11 of known size, posed from the color image, and the same marker posed by a
controller touch in STAGE. It then sends `frame=pelvis-relative` joints already
in STAGE axes. Shoulder fitting does not run for that frame.
Static shoulders alone are collinear: the client waits for sufficient motion
instead of inventing a pitch angle. Until 2026-10-12 the fit required shoulder
RMS below 40 mm and never locked in any recorded episode: YOLO shoulder
keypoints with D435 depth sit 6–9 cm RMS from Meta's shoulder joints (surface
vs joint, 30 vs 37 cm width, edge depth), so every earlier "camera leg" was
Meta's generated legs. Registration now:

- buffers up to 300 packets with both shoulders measured and the shoulder line
  within ~35° of facing the camera (side-on depth hits the arm or the edge);
- runs RANSAC over whole packets (inlier: every landmark within 15 cm), so a
  second person walking through view cannot dominate the fit;
- accepts when ≥40 inlier packets, ≥50 % inliers, inlier RMS < 90 mm, second
  spread axis ≥ 6 cm, and the rotations of the older and newer halves of the
  consensus agree within 5°;
- keeps refining every 10th packet after locking, with the same acceptance.

Replayed on the three recorded episodes it locks after 17, 47 and 23 s; the
halves then agree within 1–3°. A or a STAGE recenter resets it.

The current server has no wrist landmarks. Optional `lwrist` / `rwrist` anchors
are supported if added later, matched to Meta wrists rather than controller
origins. The nose is not used for registration (the earlier nose/HMD term was
dropped with the 40 mm gate).

Camera packets use `frame=camera`, absolute camera coordinates. Calibrated
`frame=pelvis-relative` packets must already use Quest STAGE axes. Unknown frames,
invalid numbers, oversized JSON, missing required landmarks and reordered
packets are rejected. The client sends HMD/controller poses back to the server.

Camera epoch timestamps are matched to a 3 s Meta history (nearest body pose
within 60 ms), rather than the latest predicted HMD pose. Registration accepts
matched historical packets younger than 1 s. Driving physical legs still
requires packets younger than 150 ms; delayed calibration never bypasses that
gate. Source age and arrival age both gate use. The client sends JSON ping messages and uses the server's `server_t` pong
field to estimate clock offset from the minimum-RTT exchange. Offset and RTT
are exposed in CSV and camera recordings. Existing servers can add this field
with `scripts/enable_camera_clock.py /path/to/skeleton_server.py`, then restart.
Without it the HUD shows SYNC; clock-skewed packets cannot enter fusion.

## Fusion

Upper body and pelvis remain Meta-controlled. Only trustworthy depth `window`
measurements (confidence at least 0.55) of an entire hip/knee/ankle chain are used;
`line` and `missing` points do not become measured ground truth. A registered
camera pelvis more than 35 cm from Meta's pelvis is another person (or a broken
registration) and contributes nothing. There is no correction-size limit: the
former 45 cm limit rejected exactly the high knee lifts the camera is for.

The camera supplies thigh and shin **directions**; knee and ankle targets are
laid out from Meta's hip with Meta's bone lengths in the historical Meta sample.
Camera positions put knees 5–7 cm and ankles 10–14 cm high (hip-midpoint pelvis,
short 2D shins), which would bend standing legs. A residual remains: the right
ankle sits ~7 cm behind on these recordings (occlusion). A 60 ms exponential filter smooths the **correction**, preserving
current high-frequency Meta motion rather than delaying the whole skeleton.
Confidence weights fade over 100 ms; each leg expires against its last good
measurement, even when empty heartbeat packets continue. A two-bone projection
preserves Meta segment lengths and hip anchors. Hip/knee
orientations receive swing rotations matching the new bone directions. Ankle
orientation/twist remain Meta-estimated: the current camera has no toe/heel
landmarks, so camera-derived foot pitch and yaw are not claimed.

At more than 150 ms the HUD shows STALE. The correction keeps full weight
until 400 ms and fades linearly to zero at 500 ms, where the result is exactly
the original Meta skeleton. Age delays the legs but does not shrink them: the
correction is relative to the time-matched Meta pose. The former 150→500 ms
ramp left a 5 Hz CPU camera (250–440 ms old) at ~0.44 weight, so the robot
moved only about half way towards the camera legs. Invalid new frames
expire old corrections. A fixed packet is never applied forever.

The always-visible status panel has a leg-source row: green `Ноги: камера
(обе/одна)` (with the delay when stale), otherwise red `Ноги: Meta. …` with the
reason — no link, clock sync, `Калибровка N%: <hint>` (stand in view / face the
camera / walk facing the camera / one person in view), no fresh data, or the
camera does not see the legs. Hidden when no camera URL is configured or in
human capture. Debug stats show CAM OFF/ALIGN/FUSE/STALE, progress, age, fit
error and contributing leg count.
CSV appends connection/state/age/fit/leg-count/weight/fit-progress/fit-hint metrics. Y recordings
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
  fixed bone lengths, invalid heartbeat expiry, the 150/500 ms transitions, a
  second person in view, a distant person after locking and a high knee lift.
- Pose-servo tests cover correction direction, zero command at goal, yaw sign
  and velocity limits.
- Native stereo/world checks keep a prop fixed in STAGE across HMD movement.
- Native WebSocket client receives the existing live server stream.
- A 15 s physical smoke test stays upright, but with static leg references the
  pose servo makes only limited progress. Walking convergence is not established.
- Live camera-to-Meta alignment still requires a visible, tracked person; empty
  server heartbeat frames verify transport only.

## Camera skeleton overlay

The miniature body display overlays measured camera landmarks in pink on the
raw Meta skeleton (orange estimated legs). Both use the same STAGE coordinates,
projection and display scale. The camera overlay uses the calibrated rigid
rotation and translation, without re-centering its pelvis on Meta; registration
errors remain visible. Leg fusion separately uses pelvis-relative corrections.
Camera nose is a nose landmark, not an HMD origin. Missing elbows are not drawn.
Only measured points with confidence at least 0.55 are shown; absent joints and
uncalibrated camera-space packets are hidden. The overlay fades after 150 ms
and disappears at 500 ms, including when physics publishing stalls.

The same samples are also drawn on the operator in the room. Bones use STAGE
metres and the eye view directly; they are not moved by the robot scene anchor.
Meta's upper body is cyan and its estimated legs are orange. Camera landmarks
are a slightly thicker pink shell, so a registration error sits beside the Meta
bone instead of being hidden inside it. Measured camera wrists are markers
only; no shoulder-to-wrist bone is invented. The neck stops 18 cm short of the
headset, because that joint is the HMD pose. A point farther than 2.5 m from
the pelvis is dropped. Where the ego robot covers the body, the hidden part of
a bone stays visible at half opacity. The Meta skeleton disappears when the
body sample is older than 200 ms. Camera fade and the 500 ms cutoff match the
miniature, including a stalled physics publish.
