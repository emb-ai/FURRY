#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/vlad/FURRY/cup-fast-check-20261015
git clone https://github.com/NVIDIA/Isaac-GR00T.git Isaac-GR00T
git -C Isaac-GR00T checkout --detach d2b7e75b937e3ec9aa5dbc798f08b89692c49734
cd Isaac-GR00T
/home/jovyan/.local/bin/uv sync --python 3.12
.venv/bin/python -c 'import gr00t; print("GROOT_IMPORT_OK")'
