#!/bin/bash
# Metrics for every replayed variant, ONE process at a time (each loads a ~270 MB CSV).
# Swing variants are classified against the unlifted reference of the clamp run.
set -uo pipefail
Q=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$Q/outputs/foot_study}
PY=$Q/.venv/bin/python
HERE=$(dirname "$0")
NAMES=${*:-clamp hz50 swing20 swing30 swing50 capsule fric10 train ft500 ft1000 ft500hz50 ft1000hz50}
for v in $NAMES; do
  src=$OUT/q_$v.csv; [ "${v:0:5}" = swing ] && src=$OUT/q_clamp.csv
  nice -n 10 "$PY" "$HERE/track.py" "$OUT/q_$v.csv" "$src" > "$OUT/m_$v.json" || echo "FAIL track $v"
  nice -n 10 "$PY" "$HERE/root.py" "$OUT/q_$v.csv" > "$OUT/r_$v.json" || echo "FAIL root $v"
done
cd "$OUT" && "$PY" "$Q/scripts/foot_study/compare.py" $NAMES
