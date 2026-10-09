#!/bin/bash
# Replays the 15-minute Quest capture for every variant: q_<name>.csv/.log in $OUT.
# At most two replays at once: the study machine has 7 GB RAM and froze with eight.
# usage: run.sh [name ...]   (default: all)
set -uo pipefail
Q=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$Q/outputs/foot_study}
export LD_LIBRARY_PATH=$OUT/lib
cd "$Q"
declare -A SPEC=(  # decimation assets toe-mode
  [clamp]="10 android/assets clamp"        [hz50]="20 android/assets clamp"
  [swing20]="10 android/assets swing20"    [swing30]="10 android/assets swing30"
  [swing50]="10 android/assets swing50"    [capsule]="10 $OUT/assets_capsule clamp"
  [fric10]="10 $OUT/assets_fric10 clamp"   [train]="20 $OUT/assets_train clamp"
  [ft500]="10 $OUT/assets_p500 clamp"      [ft1000]="10 $OUT/assets_p1000 clamp"
  [ft500hz50]="20 $OUT/assets_p500 clamp"  [ft1000hz50]="20 $OUT/assets_p1000 clamp")
NAMES=${*:-clamp hz50 swing20 swing30 swing50 capsule fric10 train ft500 ft1000 ft500hz50 ft1000hz50}
n=0
for v in $NAMES; do
  read -r dec assets mode <<< "${SPEC[$v]}"
  ( "$OUT/build/replay_dec$dec" "$assets" stand "$OUT/episode-1791467626074995" "$OUT/segments.csv" "$mode" \
      "$OUT/q_$v.csv" > "$OUT/q_$v.log" 2>&1 || echo "FAIL $v" ) &
  (( ++n % 2 )) || wait
done
wait
for v in $NAMES; do echo "$v: $(tail -n1 "$OUT/q_$v.log")"; done
