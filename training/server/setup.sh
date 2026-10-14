#!/bin/bash
# Build the isolated training root on the GPU server. Idempotent; copies, never edits, the source root.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SOURCE=${FURRY_SOURCE_ROOT:?set FURRY_SOURCE_ROOT in server.env}
UV=$(command -v uv)
mkdir -p "$FURRY_ROOT"/{checkpoints,runs,torch-extensions,cuda-cache,matplotlib-cache,xdg-cache}
cd "$FURRY_ROOT"

for item in isaacgym TWIST2 assets dataset cuda-compat runtime-assets clamp-mirrored g1_sim \
            comparison-baseline.pt locomotion-ancestor-manifest.json; do
  [[ -e $item ]] || cp -a "$SOURCE/$item" .
done
[[ -e checkpoints/111586-500 ]] || cp -a "$SOURCE/stance-finetune-20261013/initial" checkpoints/111586-500
# Pilot configs and the phase table are generated on the server, not tracked in git.
mkdir -p "$FURRY_CODE"
for item in pilot-A.json pilot-B.json pilot-C.json phase.npz phase.json phase-v2.npz phase-v2.json; do
  [[ -e $FURRY_CODE/$item ]] || cp -a "$SOURCE/quest-contact-clearance-20261012/$item" "$FURRY_CODE/"
done

grep -v '^-e ' "$SOURCE/environment-training.txt" > requirements-training.txt
grep -v '^-e ' "$SOURCE/environment-evaluation.txt" > requirements-evaluation.txt
"$UV" python install 3.8.20 3.12.11
[[ -x env/bin/python ]] || "$UV" venv --python 3.8.20 env
"$UV" pip install --python env/bin/python -r requirements-training.txt
"$UV" pip install --python env/bin/python --no-deps -e isaacgym/python -e TWIST2/pose -e TWIST2/rsl_rl -e TWIST2/legged_gym
[[ -x eval-env/bin/python ]] || "$UV" venv --python 3.12.11 eval-env
"$UV" pip install --python eval-env/bin/python -r requirements-evaluation.txt

sha256sum comparison-baseline.pt checkpoints/111586-500/checkpoint_000500.pt
env/bin/python -c 'import torch; print("train torch", torch.__version__, torch.cuda.is_available())'
eval-env/bin/python -c 'import torch, mujoco; print("eval torch", torch.__version__, "mujoco", mujoco.__version__)'
