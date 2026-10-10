#!/bin/zsh
set -eu
G1_WIFI_ROOT="${0:A:h}"
if [[ -x "$G1_WIFI_ROOT/.venv/bin/python" ]]; then
  G1_WIFI_PYTHON="$G1_WIFI_ROOT/.venv/bin/python"
else
  G1_WIFI_PYTHON="$(command -v python3)"
fi
exec "$G1_WIFI_PYTHON" "$G1_WIFI_ROOT/scripts/quest_wifi.py" "$@"
