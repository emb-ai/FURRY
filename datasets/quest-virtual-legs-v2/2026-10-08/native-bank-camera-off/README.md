# Native command bank, camera tracking OFF (night-v3 Meta clips)

Actor inputs for the 66 night-v3 Quest TRAIN clips (33 originals + 33 mirrors):
native `MetaRetargeter::Solve` commands replayed from the raw source episode,
`[t, 35 commands]` per clip, causal zero-order hold at body-event times.
Entries match clips by original NPZ SHA-256 (`target_sha256`).

Rebuilt on Linux on 2026-10-14 with `training/quest_contact_clearance/`
`meta_epochs.py`, `build_native_export.py`, `export_native_commands`,
`build_command_bank.py` from the main native source. `audit.json` equals the
original camera-OFF bank statistics (legs mean 0.00554, p95 0.02367, max
0.18578 rad vs the filtered targets). `night-v3-quest.manifest.json` is the
night-v3 manifest with `source`/`npz_file` added, which the bank was built
against; `continuous.json` holds the rebuilt run origins and epochs.
Not a measurement of human legs; `long_run_ready` stays false.
