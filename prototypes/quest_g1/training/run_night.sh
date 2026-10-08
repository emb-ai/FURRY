#!/bin/bash
#SBATCH --job-name=furry-quest-night-v3
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
TASK_DIR="${1:?bundle directory}"
TASK_VENV="${2:?venv directory}"
cd "$TASK_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
TASK_OUT="runs/$SLURM_JOB_ID"
mkdir -p "$TASK_OUT"
"$TASK_VENV/bin/python" -m unittest discover -s training -p 'test_*.py' -v
"$TASK_VENV/bin/python" training/check_pipeline.py --assets assets --dataset dataset --policy assets/policy.onnx > "$TASK_OUT/pipeline-check.json"
"$TASK_VENV/bin/python" training/smoke_updates.py --assets assets --dataset dataset --initial initial_actor.pt --source upstream/actor_critic_future.py --device cuda > "$TASK_OUT/optimizer-smoke.json"
"$TASK_VENV/bin/python" training/mirror_guard.py --assets assets --dataset dataset --initial initial_actor.pt --source upstream/actor_critic_future.py --baseline baseline-reference.json --output "$TASK_OUT/preflight"
exec "$TASK_VENV/bin/python" training/night_experiment.py --output "$TASK_OUT"
