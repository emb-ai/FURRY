# Raw Meta full-body capture

Based on `camera-skeleton` commit `b8db09975d94678b59bdc847288a38faf139404a`.

Build with `G1_CAPTURE_APP=1 android/build_apk.sh`. This produces the separate
`org.g1lab.quest.capture` application, labelled **FURRY Capture**, signed with
the local development key. It can coexist with `org.g1lab.quest`; preferences
and recordings are separate. The default build command retains the original
application identity.

Each recorded episode now contains `body_full.bin` (84 raw SDK joint poses and
64-bit flags per input sample) and `body_skeleton.bin` (versioned bind skeletons).
Streams preserve sequence IDs and OpenXR timestamps. Poses use OpenXR STAGE
metres and wxyz quaternions, before camera fusion. Missing/inactive samples are
explicitly represented; invalid pose values must not be treated as observations.
Both streams have schema version 1 and explicit little-endian serialization.

Existing 14-joint CSV, camera recording, camera fusion, GMR retargeting, policies
and capture controls remain in place. The full-body payload is immutable and
shared across frame copies. Recording more joints does not make Virtual Legs
measured foot tracking, change actor inputs or introduce the full-foot retargeter.

Use the existing **Сбор траекторий → Человек → Обучение: 15 мин** or separate
**Тест: 5 минут** guided capture. Before a long capture, record a short hardware
probe and verify all expected streams, valid joint flags and bind skeletons.
The existing capture timer still uses the legacy tracking-validity criteria;
completion alone does not certify all 84 joints as valid.

After exporting an episode:

```sh
python scripts/inspect_full_body.py /path/to/episode
python scripts/inspect_full_body.py /path/to/episode --csv raw-joints.csv
```

The checker validates stream structure, ordering and bind versions and reports
per-joint valid frame counts. Keep existing `inspect_capture.py` checks as well.

Local validation on 2026-10-12 covers mocked SDK capture, shuffled bind IDs,
immutable samples, invalid/inactive tracking, binary round trips, truncation,
and actual EpisodeRecorder start/stop/abort. The Android APK builds and its
signature/package are checked. Installation and headset runtime validation
have not been performed for this build.
