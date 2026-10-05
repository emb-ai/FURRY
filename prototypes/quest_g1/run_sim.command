#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/mjpython ]]; then
  print "Environment missing. Follow README.md to install."
  exit 1
fi
# Invoke the script through Python: MuJoCo's installed shebang cannot handle
# spaces in this project's path on macOS.
G1_PYTHON_LIB_DIR="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export DYLD_FALLBACK_LIBRARY_PATH="${G1_PYTHON_LIB_DIR}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
exec .venv/bin/python .venv/bin/mjpython -m g1_sim "$@"
