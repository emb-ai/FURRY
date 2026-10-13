# DexWeave body-only / Quest virtual legs

Independent reconstruction of the body-only experiment in [DexWeave v1](https://arxiv.org/html/2609.34724v1). The public DexWeave repository has no released training implementation or weights as checked on 2026-10-12. This module must not be described as a port of released author code.

The source is Meta virtual legs plus the existing continuous FURRY GMR reference. Finger tracking and objects are outside this experiment. Nothing here changes the Quest runtime, installs an APK or deploys a policy.

## Components

- `reference.py`: checks source SHA256 and frozen stage splits; exports named 50 Hz robot states, FK links, source timing/flags and ankle-based support proposals. No contact-confidence number is invented.
- `project_support.py`: fixes world XY toe anchors during proposed stance; adjusts translation and legs subject to joint limits and floor clearance. Root orientation and all upper joints are preserved exactly. This is a kinematic adaptation; dynamic feasibility remains an evaluation question.
- `network.py`: six anatomical tokens, two attention blocks, five action heads; 157 actor observations, 286 critic observations, 29 body actions. `spec.py` records unresolved author details and explicit implementation choices.
- `isaac_env.py`, `train_isaac.py`: pinned BeyondMimic/IsaacLab environment integration and RSL-RL PPO from scratch, one policy per reference. No validation-based early stop or extra whole-run KL stop.
- `export.py`: exports empirical normalization, including its epsilon, inside ONNX and checks inference parity.
- `evaluate_mujoco.py`: five matched first-frame trials per reference at 50 Hz control / 1000 Hz physics, maximum 10 seconds. TWIST2 and the new policy retain their respective observation/PD contracts. Physical falls, paper-style tracking terminations, world drift and contact-point slip are separate metrics.

## Pinned training dependencies

- BeyondMimic: `cd65172032893724b445448818c34165846d847d`.
- IsaacLab: tag `v2.1.0`; Isaac Sim `4.5.0`; Python 3.10.
- RSL-RL: `2.3.1`; resolved server Torch `2.5.1+cu124`.
- G1 model and PD constants: pinned BeyondMimic body model. This is a different hand embodiment from G1–Inspire in DexWeave; Inspire payload randomization and hand-load compensation are omitted.
- Server experiment root: `/home/jovyan/vlad/FURRY/dexweave-body-20261012` on `andrei_gpu`. Only this experiment's isolated environment is changed.

H100 lacks RTX rendering support. Headless physics was tested separately. Isaac Sim's URDF importer GUI must be skipped in this experiment's environment; parsing and physics stay active. The original extension file and before/after hashes are retained in server `evidence/`. Check `native-contract.json` for the actual PhysX backend, named joint order and control periods.

## Running

Run from the directory containing the `dexweave` package, or add that directory to `PYTHONPATH`. Check free GPUs before choosing `CUDA_VISIBLE_DEVICES`; the selected physical GPU becomes `cuda:0` inside the process.

```bash
CUDA_VISIBLE_DEVICES=7 FURRY_HEADLESS_URDF=1 \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
env/bin/python -m dexweave.train_isaac \
  --reference pilot-reference.npz --output runs/stage01 \
  --num-envs 4096 --iterations 8000 --device cuda:0
```

Before a full run, use a separate `--smoke --iterations 2` output directory. Successful setup, FK parity, finite losses and changed actor weights are required. A smoke checkpoint is not a trained-quality result. Runtime status is `runs/<name>/status.json`; metrics and checkpoint files are stored locally on the server without cloud logging.

```bash
python -m pytest dexweave/test_contract.py -q
python -m dexweave.export runs/stage01/checkpoint-08000.pt stage01.onnx
```

Do not evaluate a per-reference policy on other references as if it were a general motion tracker. Preserve held-out stages 2/9/14. Mirrors and prepared replays are not independent demonstrations. The study averages the final checkpoint window; do not report one selected checkpoint as reproducing that aggregation.

## Known reconstruction choices

Ankle-based support inference substitutes for unavailable source toe poses. Thresholds, support masks, original states and solver outcomes remain inspectable. Waist/arm head routing and Transformer normalization placement are explicit choices where the paper is incomplete. Sampled-start pre-roll is interpreted as starting 0.5–1.5 seconds before the sampled phase, within the same episode limit. Exact source-model equivalence, live teleoperation and Quest deployment are not claimed.

## Shared-policy continuation of the study

The current source also accepts a multi-clip train manifest with balanced
sampling (`motion_library.py`) and synchronized multi-GPU PPO
(`distributed.py`). One policy learns all train clips. Train/test membership
is explicit and disjoint; test data is excluded from normalization.
Actor and critic normalizers aggregate training samples across all ranks.
`evaluate_mujoco.py --full-reference` evaluates the complete clip; inspect
tracking termination, physical falls and the executed fraction together.
The earlier per-reference and ten-second protocols above are historical.
See [the integration record](../../../../docs/research/project-integration-20261013/README.md)
for the current six-GPU 12/3 experiment and its dated evidence.
