# Preserved training and manipulation research

This archive preserves source and selected evidence that were absent from main at
`68079cf`, inspected on 2026-10-21. It does not change the Quest runtime, its policy
menu, or the accepted status of any experiment. Historical JSON files retain
their original timestamps and sometimes describe an earlier running state.

## What the evidence establishes

| Experiment | Latest evidence included | Result and limits |
| --- | --- | --- |
| Cup ACT and visual diffusion | Native 50 Hz experiment, data export, native bridges, trainers, evaluation protocols and summaries | In the same eight recorded-command continuations: 100 Hz 8/8, 30 Hz 5/8, aligned 50 Hz 8/8. Three separate Linux full-reset evaluations: old ACT at 30 Hz 0/10, ACT at 50 Hz 2/10, ACT with measured state 0/10. The new variants also received more updates and have different temporal horizons, so this does not isolate frequency alone. |
| Cup state imitation | 59-episode 50 Hz dataset, BC-RNN and diffusion source and best checkpoints | Both completed 12,000 updates. BC-RNN full-reset evaluation was 0/10; diffusion evaluation has no completed local receipt. Checkpoints are preserved for analysis, not deployment. |
| DexWeave cup-to-box | Four reference exports, exact physical asset/URDF conversion, manipulation trainer, full-reset evaluator and results | Checkpoint 3500 failed its own reference rollout. A later checkpoint 5900 was evaluated from ten full resets: five falls and five timeouts; no candidate goal-settle event. The evaluator does not implement the complete native success detector (`task_success_evaluated=false`). |
| Rubik physical sequences | Original native snapshot code, wrist/Grip planners and verifiers, full `second-other-face.npz` and earlier dependencies | Nine turns of one face, whole-cube X rotation and two turns of the adjacent face were replayed continuously: 28,367 frames, exact qpos/userdata, no warnings. The adjacent-face return exceeds the fixed grasp tolerances. This is one deterministic offline trajectory, not a general R/U/F controller or a Quest test. |
| Rubik curriculum | Environment, trainer, demonstrations, BC initializer, stage validation and final status | Stopped at stage 0 after 65,536 PPO steps, no completed curriculum stages and no accepted policy. The short warm-start grasp completions are not complete turns. |
| Stance finetuning | Retarget/reference preparation and later DexWeave resume source | The deployed policy bytes already exist in main. This preserves missing preparation/resume code; outcome evidence is in the sibling artifacts archive. |

The working TWIST2 and body-only DexWeave code from the committed training branch
was already present in main, despite different commit ancestry. Its dirty v4
MuJoCo trainer was an unfinished, older experiment and is not promoted here.

The successful cup `pi_high` model reported elsewhere in main (23/50 fresh
layouts) is a different trainer and protocol. The ACT/state-only results here
do not supersede that result. Its checkpoint was not present among these local
artifacts; this archive does not claim to contain it.

## Layout and integrity

[manifest.json](manifest.json) maps every preserved dependency to its original
workspace path, absolute source location, exact SHA256 and size. New files are
stored under `workspace/outputs/`. Byte-identical files already in main are
referenced by repository path, including frozen HOMIE weights and archived
MuJoCo models. This avoids duplicate datasets and model copies.

Historical Markdown is stored with `.md.txt` suffix to preserve its exact bytes
and obsolete local links without presenting it as current repository navigation.
`archive.py restore` restores the original filename. No algorithm was edited.
Selected NPZ, PyTorch and ZIP files are Git LFS objects. The two PyTorch
checkpoints are local experimental weights, not a claim of task success.

The archive helper requires Python 3.11 or newer (for `hashlib.file_digest`).
From this directory:

```sh
python archive.py verify
python archive.py verify --with-references
python archive.py restore --destination /tmp/furry-training-reconstruction --relocate
```

Materialize the referenced Git LFS files before running `--with-references` or
`restore`; a missing pointer produces an error. Verification checks hashes,
parses Python source without executing it, and checks ZIP/NPZ CRCs. Restoration
requires a new destination, verifies all selected bytes before writing, and
never alters the archive. Optional `--relocate` replaces only the historical
absolute workspace prefixes in source files; the changed paths are listed in
`RESTORE.json`. Use `--group rubik`, `--group cup-native50` or another original
output-directory prefix to select a smaller set.

## Replaying the preserved Rubik trajectory

`replay_rubik.py` compiles the original native simulation and replays the stored
wrist/Grip commands without the historical planner scripts or a prebuilt shared
library. Install MuJoCo 3.3.7, NumPy, SciPy and ONNX Runtime in the Python
environment, and use a C++17 compiler. Materialize the main files named by the
manifest, then run from this archive directory:

```sh
python archive.py restore --group rubik --destination /tmp/furry-rubik
python replay_rubik.py --workspace /tmp/furry-rubik --frames 100 --output /tmp/prefix.json
python replay_rubik.py --workspace /tmp/furry-rubik --output /tmp/full-replay.json
```

The short prefix command checks the reconstruction. The full command covers all
28,367 frames. The replay helper reports state differences and fails if they
exceed 1e-8 or native warnings occur; exact results on a different architecture
are not assumed. The archived 2026-10-15 receipt covers the complete trajectory.

## Reproduction boundaries

This is an auditable research archive, not one portable training application.
Related scripts keep their original sibling-directory structure and historical
runtime snapshots. Do not substitute current main runtime code for the archived
native snapshot when reproducing an exact replay. Historical report/source
files may mention unsuccessful trial NPZs that were not selected for this
archive; the manifest is authoritative about available files.

- Native replay and export require MuJoCo 3.3.7, NumPy, SciPy and ONNX Runtime.
  Source snapshots include the original native implementation and ONNX headers;
  compiled `.dylib`/`.so` files are deliberately not archived.
- The Rubik lower controller is frozen HOMIE Standing. Only wrist poses and the
  original scalar Grip commands belong to the accepted sequence. Dedicated independent-finger control drivers were excluded. Historical
  shared wrappers retain unused optional branches; the original replay bridge
  exposes no independent-finger API.
- Cup diffusion includes the small pinned iDP3 U-Net/EMA subset and its license.
  ACT additionally needs the pinned LeRobot dependency documented by its source
  metadata/setup script. Large failed visual-model checkpoints and repeat RGB
  datasets are not included. Published raw/RGB cup datasets remain in
  `datasets/quest-cup/`.
- DexWeave manipulation requires Isaac Lab, RSL-RL and the BeyondMimic
  `whole_body_tracking` package used by the recorded launch. The four reference
  files and physical assets are preserved; GPU training was not restarted.
- Rubik learning additionally requires Gymnasium 1.4.0 and Stable Baselines3
  2.9.0. Installed dependency caches, launch processes and transient logs are not
  part of the archive.

No new task-performance or hardware acceptance follows from the archive checks.
