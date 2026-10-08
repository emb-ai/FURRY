#!/bin/bash
#SBATCH --job-name=furry-quest-ppo
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120
set -euo pipefail
TASK_DIR="${1:?Pass the uploaded bundle directory}"
VENV_DIR="${2:?Pass the isolated venv directory}"
cd "$TASK_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
RUN_DIR="runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
"$VENV_DIR/bin/python" training/check_pipeline.py --assets assets --dataset dataset --policy assets/policy.onnx > "$RUN_DIR/pipeline-check.json"
"$VENV_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
srun "$VENV_DIR/bin/python" training/train.py --assets assets --dataset dataset --initial initial_actor.pt --source upstream/actor_critic_future.py --output "$RUN_DIR" --envs 16 --horizon 128 --iterations 1000 --eval-every 100
"$VENV_DIR/bin/python" training/export_policy.py "$RUN_DIR/best.pt" upstream/actor_critic_future.py "$RUN_DIR/best.onnx"
"$VENV_DIR/bin/python" training/evaluate.py --assets assets --dataset dataset --policy "$RUN_DIR/best.onnx" --split validation --seeds 0 1 2 --output "$RUN_DIR/final-validation.json"
