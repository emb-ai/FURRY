#!/bin/bash
# 20-update pilot of condition B on one idle H100 (smoke only, not critic readiness).
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU=$("$(dirname "$0")/free_gpu.sh" | head -1)
[[ -n $GPU ]] || { echo 'No idle H100' >&2; exit 1; }
OUT=$FURRY_ROOT/runs/smoke-B-$(date +%Y%m%d-%H%M%S)
echo "gpu $GPU output $OUT"
cd "$FURRY_ROOT"
CUDA_VISIBLE_DEVICES=$GPU env/bin/python "$FURRY_CODE/cycle_runner.py" --root "$FURRY_ROOT" --pilot B --updates "${1:-20}" --output "$OUT"
tail -1 "$OUT/metrics.jsonl"
