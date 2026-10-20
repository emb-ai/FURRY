#!/bin/bash
#SBATCH --job-name=furry-cup-checkpoints
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$HOME/furry/cup-dp-20261015"
env/bin/python pipeline-check/compare_checkpoints.py
