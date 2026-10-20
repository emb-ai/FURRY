#!/bin/bash
#SBATCH --job-name=furry-cup-smoke
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$HOME/furry/cup-dp-20261015"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi
env/bin/python train.py --data gpu-smoke.hdf5 --out gpu-smoke-run --max-steps 3 --workers 4
