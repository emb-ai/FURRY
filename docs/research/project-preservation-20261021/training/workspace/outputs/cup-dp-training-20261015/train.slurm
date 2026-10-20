#!/bin/bash
#SBATCH --job-name=furry-cup-r3m
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$HOME/furry/cup-dp-20261015"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi
resume_args=()
if [[ -f run-seed0/latest.pt ]]; then resume_args+=(--resume); fi
env/bin/python train.py --out run-seed0 --epochs 301 --workers 4 "${resume_args[@]}"
