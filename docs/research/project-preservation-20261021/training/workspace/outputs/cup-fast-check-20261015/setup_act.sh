#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/vlad/FURRY/cup-fast-check-20261015
UV=/home/jovyan/.local/bin/uv
git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout --detach 53feaef37b67e8f4c23b6953e880ff535e39d6fe
"$UV" venv --python 3.12 env-act
"$UV" pip install --python env-act/bin/python ./lerobot h5py pandas pyarrow
env-act/bin/python -c 'from lerobot.policies.act.modeling_act import ACTPolicy; print("ACT_IMPORT_OK")'
