# Source on the GPU server: isolated FURRY training environment. Paths come from server.env.
source "$(dirname "${BASH_SOURCE[0]}")/server.env"
: "${FURRY_ROOT:?set FURRY_ROOT in server.env}"
export FURRY_ROOT FURRY_SOURCE_ROOT FURRY_CODE=$FURRY_ROOT/code/quest_contact_clearance
export PATH=$FURRY_ROOT/env/bin:$HOME/.local/bin:$PATH
# Driver 560 (CUDA 12.6): PhysX segfaults without the 12.8 forward-compat libcuda.
export LD_LIBRARY_PATH=$FURRY_ROOT/cuda-compat/usr/local/cuda-12.8/compat:$FURRY_ROOT/python/cpython-3.8.20-linux-x86_64-gnu/lib:$FURRY_ROOT/env/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$FURRY_CODE:$FURRY_ROOT:$FURRY_ROOT/isaacgym/python:$FURRY_ROOT/TWIST2/legged_gym:$FURRY_ROOT/TWIST2/rsl_rl:$FURRY_ROOT/TWIST2/pose
export UV_CACHE_DIR=$FURRY_ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$FURRY_ROOT/python UV_LINK_MODE=copy
export TORCH_EXTENSIONS_DIR=$FURRY_ROOT/torch-extensions CUDA_CACHE_PATH=$FURRY_ROOT/cuda-cache
export MPLCONFIGDIR=$FURRY_ROOT/matplotlib-cache XDG_CACHE_HOME=$FURRY_ROOT/xdg-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=8
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=1 WANDB_MODE=disabled
