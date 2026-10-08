#!/bin/bash
#SBATCH --job-name=furry-quest-ppo-v2
#SBATCH --partition=AMD7V12-RTX4500-CCP-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@120
set -euo pipefail
TASK_DIR="${1:?Pass the uploaded bundle directory}"
VENV_DIR="${2:?Pass the isolated venv directory}"
shift 2
cd "$TASK_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
RUN_DIR="runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
"$VENV_DIR/bin/python" -m unittest discover -s training -p 'test_*.py' -v
"$VENV_DIR/bin/python" training/check_pipeline.py --assets assets --dataset dataset --policy assets/policy.onnx > "$RUN_DIR/pipeline-check.json"
"$VENV_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
srun "$VENV_DIR/bin/python" training/train.py --assets assets --dataset dataset --initial initial_actor.pt --source upstream/actor_critic_future.py --output "$RUN_DIR" --envs 16 --horizon 128 --iterations 300 --eval-every 25 "$@"
if [[ -f "$RUN_DIR/best.pt" ]]; then
  "$VENV_DIR/bin/python" training/export_policy.py "$RUN_DIR/best.pt" upstream/actor_critic_future.py "$RUN_DIR/best.onnx"
  "$VENV_DIR/bin/python" - "$RUN_DIR" <<'PY'
import json,sys
sys.path.insert(0,'training')
from evaluate import evaluate
from rewards import RewardConfig
p=sys.argv[1];config=json.load(open(p+'/config.json'))
evaluate('assets','dataset',p+'/best.onnx',split='validation',seeds=config['eval_seeds'],output=p+'/final-validation.json',reward_config=RewardConfig(**config['reward_config']))
PY
else
  echo 'No checkpoint passed acceptance; retain original policy. See result.json.'
fi
