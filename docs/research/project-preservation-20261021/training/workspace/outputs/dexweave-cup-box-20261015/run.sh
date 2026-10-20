#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
base=/home/jovyan/vlad/FURRY/dexweave-body-20261012
export FURRY_HEADLESS_URDF=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$base/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$base/env/bin/python" -m dexcup.train "$@"
