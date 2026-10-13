#!/usr/bin/env bash
# Each run owns one reference/GPU and writes only inside its output directory.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=$1
clip=$2
label=$3
run="runs/$label"
mkdir -p "$run"
phase=training
trap 'printf "{\"status\":\"failed\",\"phase\":\"%s\"}\n" "$phase" > "$run/pipeline-state.json"' ERR
printf '{"status":"running","phase":"training"}\n' > "$run/pipeline-state.json"
export CUDA_VISIBLE_DEVICES="$gpu"
export FURRY_HEADLESS_URDF=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
env/bin/python -m dexweave.train_isaac --reference "support-projected/train/$clip.npz" \
  --output "$run" --num-envs 4096 --iterations 8000 --device cuda:0 > "$run/train.log" 2>&1

# Training has released its GPU; all exports and sim-to-sim evaluations are CPU.
export CUDA_VISIBLE_DEVICES=""
phase=evaluation
printf '{"status":"running","phase":"evaluation"}\n' > "$run/pipeline-state.json"
mkdir -p "$run/evaluation"
env/bin/python -m dexweave.evaluate_mujoco --model assets/furry-20261008.mjb \
  --references support-projected --policy assets/twist2-20k.onnx --backend twist2 \
  --clip-id "$clip" --output "$run/evaluation/twist2.json" > "$run/evaluation/twist2.log" 2>&1
for iteration in 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000; do
  checkpoint=$(printf '%05d' "$iteration")
  env/bin/python -m dexweave.export "$run/checkpoint-$checkpoint.pt" "$run/evaluation/policy-$checkpoint.onnx" \
    > "$run/evaluation/export-$checkpoint.log" 2>&1
  env/bin/python -m dexweave.evaluate_mujoco --model assets/furry-20261008.mjb \
    --references support-projected --policy "$run/evaluation/policy-$checkpoint.onnx" --backend dexweave \
    --clip-id "$clip" --output "$run/evaluation/checkpoint-$checkpoint.json" \
    > "$run/evaluation/checkpoint-$checkpoint.log" 2>&1
done
env/bin/python -m dexweave.summarize "$run/evaluation" > "$run/comparison.json.tmp"
mv "$run/comparison.json.tmp" "$run/comparison.json"
printf '{"status":"completed","phase":"evaluation"}\n' > "$run/pipeline-state.json"
