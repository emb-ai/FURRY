#!/bin/bash
#SBATCH --job-name=furry-cup-one-check
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$HOME/furry/cup-dp-20261015"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
env/bin/python pipeline-check/one_episode.py --out pipeline-check/single-run --steps 3000
