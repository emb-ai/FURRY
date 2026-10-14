# Shared GPU server

Scripts for an isolated FURRY training root on a shared multi-H100 container.
Other users' processes and files are visible there: never kill, edit or delete
anything outside the FURRY root, and only use idle GPUs.

Host and paths are not in git. Copy `server.env.example` to `server.env` and
fill it in; `sync.sh` uploads it next to `env.sh`. Everything lives inside
`FURRY_ROOT`: uv Pythons, uv cache, `env` (Python 3.8, torch 2.4.1 cu121,
Isaac Gym, TWIST2 editable), `eval-env` (Python 3.12, MuJoCo), CUDA caches
and runs.

```sh
training/server/sync.sh                      # local: push git-tracked code + server.env
ssh "$FURRY_HOST"
bash "$FURRY_ROOT/code/server/setup.sh"      # once; idempotent
bash "$FURRY_ROOT/code/server/free_gpu.sh"   # idle H100 UUIDs
bash "$FURRY_ROOT/code/server/smoke.sh" 20   # pilot B, 20 actor updates
```

`setup.sh` copies data from the earlier root `FURRY_SOURCE_ROOT` without
changing it: Isaac Gym, the pinned TWIST2 snapshot, the sphere-sole URDF, the
1,090-motion dataset, `comparison-baseline.pt`, checkpoint 111586/500, pilot
configs and phase tables.

The driver is 560 (CUDA 12.6). Isaac Gym PhysX segfaults unless `env.sh` puts
the extracted `cuda-compat-12-8` libcuda first on `LD_LIBRARY_PATH`. Run long
jobs inside `tmux` so they survive SSH drops.
