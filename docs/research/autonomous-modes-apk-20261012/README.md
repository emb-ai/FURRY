# HOMIE / SONIC Capture APK — 2026-10-12

Built `furry-capture-autonomous.apk`, package `org.g1lab.quest.capture`, 824158109 bytes.
SHA256: `e819d228d64e59fb4ff9a393c4b6a4da933e9eb343839658b719399ae3c5a3e8`.

Menu: **Политика → HOMIE v2: автономная ходьба / SONIC: автономные ноги**.
Both are selectable runtime backends. Selection persists, resets physics/history/
calibration, and is disabled while recording. TWIST2 stays the default.
Full 84-joint raw capture and the latest camera-skeleton camera-weight fix are retained.

SONIC uses the same official three models and GMR FK upper targets as the desktop
comparison. Its planner runs on a separate CPU thread on Android; encoder/decoder
run at 50 Hz. Initial planning and model loading remain synchronous. Exhausted
plans produce a runtime fault. Planner latency and age are in telemetry.

Verification:

- Android ARM64 build and APK signature verified.
- ZIP CRC, all nine policy catalogue entries, all ONNX SHA256 and embedded source
  hashes verified; no desktop test recordings included in the package.
- 100 unit tests: 99 passed, one optional test skipped.
- Native SONIC vs Python: 100 updates, max qpos difference 4.3143267541845276e-7.
- Native leg-input invariance, failed loading preserving the active mode, reset,
  switching TWIST2/HOMIE/SONIC, stand/cup/push-T smoke, record/replay model identity,
  and asynchronous planner reset passed.

Not installed. Quest inference speed, memory use and live behavior are unverified.
The desktop benchmark used synchronous planning; its metrics are not a measurement
of this Android asynchronous schedule. Each SONIC recording includes the large
planner model. Actual states support playback; exact dynamic restart is not
supported because asynchronous plan history is not recorded.

Source and model revisions/licenses are under `prototypes/quest_g1/policies/sonic-vr3`.
Models are stored as Git LFS objects; a fresh checkout needs `git lfs pull`.
Build with `G1_CAPTURE_APP=1 android/build_apk.sh` from `prototypes/quest_g1`.
