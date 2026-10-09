#!/bin/bash
# Prepares the Quest foot-tripping study (FOOT_TRIPPING_QUEST.md) in $OUT:
# episode + segments, replay binaries at 100/50 Hz policy rate, scene and policy variants.
set -euo pipefail
Q=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$Q/../../datasets/quest-virtual-legs-v2/2026-10-08
OUT=${OUT:-$Q/outputs/foot_study}
P=$Q/.venv/lib/python3.12/site-packages
mkdir -p "$OUT/build" "$OUT/lib"
cd "$Q"

[ -d "$OUT/episode-1791467626074995" ] || unzip -q "$DATA/raw/source-episode.zip" -d "$OUT"
python3 -c "import json,sys; print('stage,first,last'); [print(f\"{s['stage']},{s['first_sequence']},{s['last_sequence']}\") for s in json.load(open(sys.argv[1]))['accepted_segments']]" \
  "$DATA/audit/audit.json" > "$OUT/segments.csv"

# Runtime sonames for the venv's MuJoCo and ONNX Runtime.
ln -sfn "$P/mujoco/libmujoco.so.3.3.7" "$OUT/lib/libmujoco.so.3.3.7"
ln -sfn "$P/onnxruntime/capi/libonnxruntime.so.1.23.2" "$OUT/lib/libonnxruntime.so.1"

# replay_meta_skeleton + toe mode swingNN (toe clamp and G1_SWING_CLEARANCE of NN mm);
# simulation.cpp with the policy every G1_POLICY_DECIMATION physics steps (10 = 100 Hz, 20 = 50 Hz).
sed -e 's/if(mode!="clamp"&&mode!="noclamp")/int swingMm=mode.rfind("swing",0)==0?std::stoi(mode.substr(5)):0;if(mode!="clamp"\&\&mode!="noclamp"\&\&!swingMm)/' \
    -e 's|meta.EnableToeClamp(mode=="clamp");|meta.EnableToeClamp(mode!="noclamp");if(swingMm)meta.EnableSwingClearance(swingMm/1000.);|' \
    android/native/replay_meta_skeleton.cpp > "$OUT/build/replay.cpp"
sed 's/steps%10==0/steps%G1_POLICY_DECIMATION==0/' android/native/simulation.cpp > "$OUT/build/simulation.cpp"
grep -q EnableSwingClearance "$OUT/build/replay.cpp" && grep -q G1_POLICY_DECIMATION "$OUT/build/simulation.cpp" \
  || { echo "patch did not apply to current sources" >&2; exit 1; }
for d in 10 20; do
  g++ -O2 -std=c++17 -DG1_POLICY_DECIMATION=$d -Iandroid/native -Ivendor/mujoco/include -Ivendor/onnxruntime-android/headers \
    "$OUT/build/replay.cpp" android/native/gmr.cpp android/native/meta_retarget.cpp "$OUT/build/simulation.cpp" \
    "$P/mujoco/libmujoco.so.3.3.7" "$P/onnxruntime/capi/libonnxruntime.so.1.23.2" -o "$OUT/build/replay_dec$d"
done

# Asset variants: android/assets symlinked, one file replaced.
variant() { # name file source
  rm -rf "$OUT/assets_$1"; mkdir -p "$OUT/assets_$1"
  for f in android/assets/*; do [ "$(basename "$f")" = "$2" ] || ln -s "$Q/$f" "$OUT/assets_$1/"; done
}
scene() { # name capsules(0/1) friction
  variant "$1" scene-stand.xml
  python3 - "$2" "$3" android/assets/scene-stand.xml "$OUT/assets_$1/scene-stand.xml" <<'PY'
import re, sys
capsules, friction, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
s = open(src).read()
if capsules:  # TWIST2 training foot: two R 2 cm capsules (URDF cylinders) instead of four 5 mm pads
    for side in ('left', 'right'):
        pads = re.findall(r'<geom type="sphere" size="0.005" pos="[^"]*" name="%s_foot(\d)_collision"([^>]*)/>' % side, s)
        assert [p[0] for p in pads] == ['1', '2', '3', '4'], side
        rest = pads[0][1]
        caps = ''.join('<geom type="capsule" size="0.02" fromto="-0.025 %s -0.02 0.125 %s -0.02" name="%s_foot%d_collision"%s/>'
                       % (y, y, side, k + 1, rest) for k, y in enumerate(('0.02', '-0.02')))
        s = re.sub(r'<geom type="sphere" size="0.005" pos="[^"]*" name="%s_foot1_collision".*?name="%s_foot4_collision"[^>]*/>'
                   % (side, side), caps, s, count=1)
s, n = re.subn(r'(<geom name="floor"[^>]*friction=")[^"]*"', r'\g<1>%s"' % friction, s)
assert n == 1
open(dst, 'w').write(s)
PY
}
scene capsule 1 0.6
scene fric10 0 1.0
scene train 1 1.0
for c in 500 1000; do
  variant p$c policy.onnx
  ln -s "$Q/policies/quest-111260/checkpoint_$(printf %06d $c).onnx" "$OUT/assets_p$c/policy.onnx"
done
echo "prepared $OUT"
